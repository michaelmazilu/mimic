from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import compute, db, ingest
from .config import Settings, get_settings
from .models import (
    ClusterSummary,
    HealthResponse,
    MarketDetailResponse,
    MarketSummary,
    RefreshResponse,
    StateResponse,
    TradeItem,
    WalletTrades,
    WalletStats,
    WalletStatsResponse,
    WalletsListResponse,
)


def _utc_ts() -> int:
    return int(time.time())


def _clamp_int(value: int, *, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


@dataclass
class CachedState:
    markets: list[MarketSummary]
    clusters: list[ClusterSummary]
    wallet_count: int
    trade_count: int
    last_refresh_ts: int | None
    next_refresh_earliest_ts: int | None
    updated_at: int


class StateCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state: CachedState | None = None

    async def set(self, state: CachedState) -> None:
        async with self._lock:
            self._state = state

    async def get(self) -> CachedState | None:
        async with self._lock:
            return self._state


class RefreshManager:
    def __init__(self, settings: Settings, cache: StateCache) -> None:
        self.settings = settings
        self.cache = cache
        self._lock = asyncio.Lock()
        self._in_progress = False

    @property
    def in_progress(self) -> bool:
        return self._in_progress

    def _next_earliest(self, last_refresh_ts: int | None) -> int | None:
        if last_refresh_ts is None:
            return None
        return int(last_refresh_ts) + int(self.settings.refresh_interval_sec)

    def _load_state_from_db(self) -> CachedState:
        with db.db_conn(self.settings.db_path) as conn:
            last_refresh_ts = db.get_last_refresh_ts(conn)
            markets_raw = db.read_market_state(conn)
            clusters_raw = db.read_clusters(conn)
            wallet_count = db.count_wallets(conn)
            trade_count = db.count_trades(conn)

        markets = [
            MarketSummary(
                conditionId=r["condition_id"],
                title=r.get("title"),
                leadingOutcome=r.get("leading_outcome"),
                consensusPercent=float(r.get("consensus_percent") or 0.0),
                weightedConsensusPercent=float(r.get("weighted_consensus_percent") or 0.0),
                totalParticipants=int(r.get("total_participants") or 0),
                participants=int(r.get("participants") or 0),
                weightedParticipants=float(r.get("weighted_participants") or 0.0),
                bandMin=r.get("band_min"),
                bandMax=r.get("band_max"),
                meanEntry=r.get("mean_entry"),
                stddev=r.get("stddev"),
                tightBand=bool(r.get("tight_band")),
                midpoint=r.get("midpoint"),
                cooked=bool(r.get("cooked")),
                priceUnavailable=bool(r.get("price_unavailable")),
                ready=bool(r.get("ready")),
                confidenceScore=float(r.get("confidence_score") or 0.0),
                endDate=r.get("end_date"),
                isClosed=bool(r.get("is_closed")),
                isActive=bool(r.get("is_active", True)),
                updatedAt=r.get("updated_at"),
            )
            for r in markets_raw
        ]

        clusters = [
            ClusterSummary(
                clusterId=c["cluster_id"],
                walletCount=int(c.get("size") or 0),
                wallets=[str(w) for w in (c.get("wallets") or [])],
                exampleWallets=[str(w) for w in (c.get("wallets") or [])[:3]],
            )
            for c in clusters_raw
        ]

        return CachedState(
            markets=markets,
            clusters=clusters,
            wallet_count=wallet_count,
            trade_count=trade_count,
            last_refresh_ts=last_refresh_ts,
            next_refresh_earliest_ts=self._next_earliest(last_refresh_ts),
            updated_at=_utc_ts(),
        )

    async def refresh_if_due(self, *, n_wallets: int | None, trades_limit: int | None) -> RefreshResponse:
        async with self._lock:
            self._in_progress = True
            try:
                now = _utc_ts()
                with db.db_conn(self.settings.db_path) as conn:
                    last_refresh_ts = db.get_last_refresh_ts(conn)
                    next_earliest = self._next_earliest(last_refresh_ts)
                    if next_earliest is not None and now < next_earliest:
                        return RefreshResponse(
                            status="skipped",
                            nowTs=now,
                            lastRefreshTs=last_refresh_ts,
                            nextRefreshEarliestTs=next_earliest,
                            refreshed=False,
                            nWallets=n_wallets,
                            tradesLimit=trades_limit,
                        )

                effective_n_wallets = _clamp_int(
                    n_wallets if n_wallets is not None else self.settings.n_wallets,
                    lo=1,
                    hi=500,
                )
                effective_trades_limit = _clamp_int(
                    trades_limit if trades_limit is not None else self.settings.trades_limit,
                    lo=1,
                    hi=200,
                )
                outbound_concurrency = _clamp_int(
                    self.settings.outbound_concurrency, lo=1, hi=5
                )  # strict cap
                sem = asyncio.Semaphore(outbound_concurrency)

                wallet_upserts = 0
                trade_inserts = 0

                timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    leaderboard = await ingest.fetch_leaderboard(
                        self.settings,
                        client=client,
                        sem=sem,
                        target_count=effective_n_wallets,
                    )
                    leaderboard_top = leaderboard[:effective_n_wallets]
                    wallets = []
                    for entry in leaderboard_top:
                        w = entry.get("proxyWallet") or entry.get("wallet") or entry.get("address")
                        if w:
                            wallets.append(str(w).lower())
                    # de-dupe preserving order
                    seen = set()
                    wallets = [w for w in wallets if not (w in seen or seen.add(w))]

                    trade_tasks = [
                        ingest.fetch_trades_for_wallet(
                            w,
                            self.settings,
                            client=client,
                            sem=sem,
                            limit=effective_trades_limit,
                        )
                        for w in wallets
                    ]
                    trade_results = await asyncio.gather(*trade_tasks, return_exceptions=True)

                    with db.db_conn(self.settings.db_path) as conn:
                        wallet_upserts = db.upsert_wallets(conn, leaderboard_top)
                        for w, result in zip(wallets, trade_results, strict=True):
                            if isinstance(result, Exception):
                                continue
                            trade_inserts += db.insert_trades(conn, w, list(result))

                        await compute.compute_and_store(
                            conn,
                            self.settings,
                            wallets=wallets,
                            client=client,
                            sem=sem,
                        )
                        db.set_last_refresh_ts(conn, now)

                state = self._load_state_from_db()
                await self.cache.set(state)

                return RefreshResponse(
                    status="refreshed",
                    nowTs=now,
                    lastRefreshTs=state.last_refresh_ts,
                    nextRefreshEarliestTs=state.next_refresh_earliest_ts,
                    refreshed=True,
                    walletUpserts=wallet_upserts,
                    tradeInserts=trade_inserts,
                    nWallets=effective_n_wallets,
                    tradesLimit=effective_trades_limit,
                )
            except Exception as e:
                now = _utc_ts()
                # best-effort cache update from db
                try:
                    await self.cache.set(self._load_state_from_db())
                except Exception:
                    pass
                return RefreshResponse(
                    status="error",
                    nowTs=now,
                    lastRefreshTs=None,
                    nextRefreshEarliestTs=None,
                    refreshed=False,
                    nWallets=n_wallets,
                    tradesLimit=trades_limit,
                    error=str(e),
                )
            finally:
                self._in_progress = False

    async def periodic_loop(self) -> None:
        while True:
            try:
                with db.db_conn(self.settings.db_path) as conn:
                    last_refresh_ts = db.get_last_refresh_ts(conn)
                now = _utc_ts()
                next_earliest = self._next_earliest(last_refresh_ts)
                if next_earliest is None:
                    sleep_for = 1
                else:
                    sleep_for = max(1, next_earliest - now)
                await asyncio.sleep(sleep_for)
                await self.refresh_if_due(n_wallets=None, trades_limit=None)
            except asyncio.CancelledError:
                return
            except Exception:
                await asyncio.sleep(5)


settings = get_settings()
app = FastAPI(title="Mimic Backend", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

cache = StateCache()
refresh_manager = RefreshManager(settings, cache)


@app.on_event("startup")
async def _startup() -> None:
    db.init_db(settings.db_path)
    # Warm cache from disk (if any), then start periodic refresh.
    await cache.set(refresh_manager._load_state_from_db())
    asyncio.create_task(refresh_manager.periodic_loop())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = await cache.get()
    now = _utc_ts()
    last_refresh_ts = state.last_refresh_ts if state else None
    next_earliest = state.next_refresh_earliest_ts if state else None
    return HealthResponse(
        status="ok",
        nowTs=now,
        lastRefreshTs=last_refresh_ts,
        nextRefreshEarliestTs=next_earliest,
        refreshInProgress=refresh_manager.in_progress,
    )


@app.get("/refresh", response_model=RefreshResponse)
async def refresh(
    n_wallets: int | None = Query(default=None, ge=1, le=500),
    trades_limit: int | None = Query(default=None, ge=1, le=500),
) -> RefreshResponse:
    return await refresh_manager.refresh_if_due(n_wallets=n_wallets, trades_limit=trades_limit)


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    now = _utc_ts()
    cached = await cache.get()
    if cached is None:
        cached = refresh_manager._load_state_from_db()
        await cache.set(cached)
    return StateResponse(
        nowTs=now,
        lastRefreshTs=cached.last_refresh_ts,
        nextRefreshEarliestTs=cached.next_refresh_earliest_ts,
        refreshIntervalSec=int(settings.refresh_interval_sec),
        refreshInProgress=refresh_manager.in_progress,
        walletCount=cached.wallet_count,
        tradeCount=cached.trade_count,
        markets=cached.markets,
        clusters=cached.clusters,
    )


@app.get("/market/{conditionId}", response_model=MarketDetailResponse)
async def market_detail(conditionId: str) -> MarketDetailResponse:
    with db.db_conn(settings.db_path) as conn:
        trades = db.get_trades_for_condition(conn, conditionId)

    if not trades:
        raise HTTPException(status_code=404, detail="Unknown conditionId")

    title = None
    grouped: dict[str, dict[str, list[TradeItem]]] = {}
    for t in trades:
        wallet = str(t["wallet"]).lower()
        outcome = str(t["outcome"])
        title = title or t.get("title")

        raw_obj: dict[str, Any] | None = None
        raw = t.get("raw_json")
        if isinstance(raw, str) and raw:
            try:
                raw_obj = json.loads(raw)
            except Exception:
                raw_obj = None

        item = TradeItem(
            wallet=wallet,
            side=str(t.get("side")),
            outcome=outcome,
            price=t.get("price"),
            size=t.get("size"),
            timestamp=t.get("timestamp"),
            txHash=t.get("tx_hash"),
            assetId=t.get("asset_id"),
            raw=raw_obj,
        )
        grouped.setdefault(wallet, {}).setdefault(outcome, []).append(item)

    wallets_out = [WalletTrades(wallet=w, byOutcome=by_outcome) for w, by_outcome in grouped.items()]
    wallets_out.sort(key=lambda x: x.wallet)

    return MarketDetailResponse(conditionId=conditionId, title=title, wallets=wallets_out)


@app.get("/wallets", response_model=WalletsListResponse)
async def list_wallets(
    order_by: str = Query(default="recent_accuracy_7d"),
    limit: int = Query(default=100, ge=1, le=500),
) -> WalletsListResponse:
    """Get all wallets with their performance stats, ordered by specified field."""
    with db.db_conn(settings.db_path) as conn:
        wallet_stats_raw = db.get_all_wallet_stats(conn, order_by=order_by, limit=limit)
        total_count = db.count_wallets(conn)

    wallets = [
        WalletStats(
            wallet=w["wallet"],
            rank=w.get("rank"),
            userName=w.get("user_name"),
            leaderboardPnl=w.get("leaderboard_pnl"),
            totalTrades=w.get("total_trades", 0),
            wonTrades=w.get("won_trades", 0),
            lostTrades=w.get("lost_trades", 0),
            pendingTrades=w.get("pending_trades", 0),
            winRate=w.get("win_rate", 0.0),
            totalPnl=w.get("total_pnl", 0.0),
            avgRoi=w.get("avg_roi", 0.0),
            recentTrades7d=w.get("recent_trades_7d", 0),
            recentWon7d=w.get("recent_won_7d", 0),
            recentAccuracy7d=w.get("recent_accuracy_7d", 0.0),
            recentTrades30d=w.get("recent_trades_30d", 0),
            recentWon30d=w.get("recent_won_30d", 0),
            recentAccuracy30d=w.get("recent_accuracy_30d", 0.0),
            streak=w.get("streak", 0),
            lastTradeTimestamp=w.get("last_trade_timestamp"),
            updatedAt=w.get("updated_at"),
        )
        for w in wallet_stats_raw
    ]

    return WalletsListResponse(wallets=wallets, totalCount=total_count)


@app.get("/wallet/{address}", response_model=WalletStatsResponse)
async def get_wallet(address: str) -> WalletStatsResponse:
    """Get detailed stats and recent outcomes for a specific wallet."""
    with db.db_conn(settings.db_path) as conn:
        stats_raw = db.get_wallet_stats(conn, address)
        outcomes_raw = db.get_wallet_outcomes(conn, address, limit=50)

    stats = None
    if stats_raw:
        stats = WalletStats(
            wallet=stats_raw["wallet"],
            totalTrades=stats_raw.get("total_trades", 0),
            wonTrades=stats_raw.get("won_trades", 0),
            lostTrades=stats_raw.get("lost_trades", 0),
            pendingTrades=stats_raw.get("pending_trades", 0),
            winRate=stats_raw.get("win_rate", 0.0),
            totalPnl=stats_raw.get("total_pnl", 0.0),
            avgRoi=stats_raw.get("avg_roi", 0.0),
            recentTrades7d=stats_raw.get("recent_trades_7d", 0),
            recentWon7d=stats_raw.get("recent_won_7d", 0),
            recentAccuracy7d=stats_raw.get("recent_accuracy_7d", 0.0),
            recentTrades30d=stats_raw.get("recent_trades_30d", 0),
            recentWon30d=stats_raw.get("recent_won_30d", 0),
            recentAccuracy30d=stats_raw.get("recent_accuracy_30d", 0.0),
            streak=stats_raw.get("streak", 0),
            lastTradeTimestamp=stats_raw.get("last_trade_timestamp"),
            updatedAt=stats_raw.get("updated_at"),
        )

    return WalletStatsResponse(
        wallet=address.lower(),
        stats=stats,
        recentOutcomes=outcomes_raw,
    )

