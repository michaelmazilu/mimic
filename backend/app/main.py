from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import backtest, compute, db, ingest, wallet_selection
from .config import Settings, get_settings
from .models import (
    BacktestConfigModel,
    BacktestRunResponse,
    BacktestTradeModel,
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
                    hi=2500,
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
                    candidate_count = min(2500, max(effective_n_wallets, effective_n_wallets * 3))
                    leaderboard = await ingest.fetch_leaderboard(
                        self.settings,
                        client=client,
                        sem=sem,
                        target_count=candidate_count,
                    )
                    entries_by_wallet: dict[str, dict[str, Any]] = {}
                    wallets: list[str] = []
                    seen = set()
                    for entry in leaderboard:
                        w = entry.get("proxyWallet") or entry.get("wallet") or entry.get("address")
                        if not w:
                            continue
                        w_lc = str(w).lower()
                        if w_lc in seen:
                            continue
                        seen.add(w_lc)
                        wallets.append(w_lc)
                        entries_by_wallet[w_lc] = entry

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
                    trades_by_wallet: dict[str, list[dict[str, Any]]] = {}
                    for w, result in zip(wallets, trade_results, strict=True):
                        if isinstance(result, Exception):
                            continue
                        trades_by_wallet[w] = list(result)

                    with db.db_conn(self.settings.db_path) as conn:
                        cached_resolutions = db.get_all_resolutions(conn)

                    condition_ids = sorted(
                        {
                            str(t.get("conditionId") or t.get("condition_id"))
                            for trades in trades_by_wallet.values()
                            for t in trades
                            if t.get("conditionId") or t.get("condition_id")
                        }
                    )
                    missing_ids = [cid for cid in condition_ids if cid not in cached_resolutions]
                    fetched_resolutions: dict[str, str] = {}
                    if missing_ids:
                        fetched_resolutions = await ingest.fetch_market_resolutions(
                            missing_ids,
                            self.settings,
                            client=client,
                            sem=sem,
                        )
                    resolutions = dict(cached_resolutions)
                    resolutions.update(fetched_resolutions)

                    if fetched_resolutions:
                        with db.db_conn(self.settings.db_path) as conn:
                            db.bulk_upsert_resolutions(
                                conn,
                                [
                                    {
                                        "condition_id": cid,
                                        "winning_outcome": winner,
                                        "resolved_at": now,
                                    }
                                    for cid, winner in fetched_resolutions.items()
                                ],
                            )

                    selection_cfg = wallet_selection.SelectionConfig()
                    wallet_metrics = wallet_selection.compute_wallet_metrics(
                        trades_by_wallet,
                        resolutions,
                        now_ts=now,
                        config=selection_cfg,
                    )
                    selected_wallets = wallet_selection.select_wallets_by_accuracy(
                        wallet_metrics.values(),
                        limit=effective_n_wallets,
                        config=selection_cfg,
                    )

                    if len(selected_wallets) < effective_n_wallets:
                        for w in wallets:
                            if w in selected_wallets:
                                continue
                            selected_wallets.append(w)
                            if len(selected_wallets) >= effective_n_wallets:
                                break

                    selected_entries = [
                        entries_by_wallet[w] for w in selected_wallets if w in entries_by_wallet
                    ]

                    with db.db_conn(self.settings.db_path) as conn:
                        wallet_upserts = db.upsert_wallets(conn, selected_entries)
                        for w in selected_wallets:
                            trades = trades_by_wallet.get(w)
                            if not trades:
                                continue
                            trade_inserts += db.insert_trades(conn, w, trades)

                        stats_rows = []
                        for w in selected_wallets:
                            m = wallet_metrics.get(w)
                            if not m:
                                continue
                            resolved = m.resolved_trades
                            stats_rows.append(
                                {
                                    "wallet": w,
                                    "total_trades": m.total_trades,
                                    "won_trades": m.wins,
                                    "lost_trades": resolved - m.wins,
                                    "pending_trades": m.total_trades - resolved,
                                    "win_rate": m.win_rate,
                                    "total_pnl": m.total_pnl,
                                    "avg_roi": m.avg_roi,
                                    "recent_trades_7d": m.recent_trades_7d,
                                    "recent_won_7d": m.recent_won_7d,
                                    "recent_accuracy_7d": m.recent_accuracy_7d,
                                    "recent_trades_30d": m.recent_trades_30d,
                                    "recent_won_30d": m.recent_won_30d,
                                    "recent_accuracy_30d": m.recent_accuracy_30d,
                                    "streak": 0,
                                    "last_trade_timestamp": m.last_trade_ts,
                                }
                            )
                        if stats_rows:
                            db.bulk_upsert_wallet_stats(conn, stats_rows)

                        await compute.compute_and_store(
                            conn,
                            self.settings,
                            wallets=selected_wallets,
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
    allow_methods=["GET", "POST", "OPTIONS"],
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
    n_wallets: int | None = Query(default=None, ge=1, le=2500),
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


# ============================================================================
# Backtest Endpoints
# ============================================================================


@app.post("/backtest", response_model=BacktestRunResponse)
async def run_backtest_endpoint(
    config: BacktestConfigModel | None = None,
) -> BacktestRunResponse:
    """
    Run a backtest with the specified configuration.
    If no config provided, uses defaults (80% confidence, scaled betting, 180 days).
    """
    # Build config from request or use defaults
    bt_config = backtest.BacktestConfig(
        min_confidence=config.minConfidence if config else 0.80,
        bet_sizing=config.betSizing if config else "scaled",
        base_bet=config.baseBet if config else 100.0,
        max_bet=config.maxBet if config else 500.0,
        lookback_days=config.lookbackDays if config else 180,
        min_participants=config.minParticipants if config else 2,
    )
    
    sem = asyncio.Semaphore(settings.outbound_concurrency)
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Use a separate connection with longer timeout to avoid locking issues
        conn = db.connect(settings.db_path, timeout=60.0)
        try:
            result = await backtest.run_backtest(
                conn,
                settings,
                bt_config,
                client=client,
                sem=sem,
            )
            conn.commit()
        finally:
            conn.close()
    
    return BacktestRunResponse(
        runId=result.run_id,
        status="completed",
        config=BacktestConfigModel(
            minConfidence=bt_config.min_confidence,
            betSizing=bt_config.bet_sizing,
            baseBet=bt_config.base_bet,
            maxBet=bt_config.max_bet,
            lookbackDays=bt_config.lookback_days,
            minParticipants=bt_config.min_participants,
        ),
        totalTrades=result.total_trades,
        winningTrades=result.winning_trades,
        losingTrades=result.losing_trades,
        pendingTrades=result.pending_trades,
        winRate=result.win_rate,
        totalPnl=result.total_pnl,
        totalInvested=result.total_invested,
        roi=result.roi,
        maxDrawdown=result.max_drawdown,
        sharpeRatio=result.sharpe_ratio,
        profitFactor=result.profit_factor,
        trades=[
            BacktestTradeModel(
                conditionId=t.condition_id,
                title=t.title,
                signalTimestamp=t.signal_timestamp,
                predictedOutcome=t.predicted_outcome,
                confidenceScore=t.confidence_score,
                betSize=t.bet_size,
                entryPrice=t.entry_price,
                actualOutcome=t.actual_outcome,
                pnl=t.pnl,
                won=t.won,
            )
            for t in result.trades
        ],
        equityCurve=result.equity_curve,
    )


@app.get("/backtest/{run_id}", response_model=BacktestRunResponse)
async def get_backtest_run(run_id: str) -> BacktestRunResponse:
    """Get results for a specific backtest run."""
    with db.db_conn(settings.db_path) as conn:
        run = db.get_backtest_run(conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        
        trades = db.get_backtest_trades(conn, run_id, limit=500)
    
    config_dict = run.get("config", {})
    
    return BacktestRunResponse(
        runId=run["run_id"],
        status=run.get("status", "unknown"),
        config=BacktestConfigModel(
            minConfidence=config_dict.get("min_confidence", 0.80),
            betSizing=config_dict.get("bet_sizing", "scaled"),
            baseBet=config_dict.get("base_bet", 100.0),
            maxBet=config_dict.get("max_bet", 500.0),
            lookbackDays=config_dict.get("lookback_days", 180),
            minParticipants=config_dict.get("min_participants", 2),
        ),
        totalTrades=run.get("total_trades", 0),
        winningTrades=run.get("winning_trades", 0),
        losingTrades=run.get("losing_trades", 0),
        pendingTrades=run.get("total_trades", 0) - run.get("winning_trades", 0) - run.get("losing_trades", 0),
        winRate=run.get("win_rate", 0.0),
        totalPnl=run.get("total_pnl", 0.0),
        totalInvested=run.get("total_invested", 0.0),
        roi=run.get("roi", 0.0),
        maxDrawdown=run.get("max_drawdown", 0.0),
        sharpeRatio=run.get("sharpe_ratio", 0.0),
        profitFactor=0.0,  # Not stored in DB
        trades=[
            BacktestTradeModel(
                conditionId=t["condition_id"],
                title=t.get("title"),
                signalTimestamp=t["signal_timestamp"],
                predictedOutcome=t["predicted_outcome"],
                confidenceScore=t["confidence_score"],
                betSize=t["bet_size"],
                entryPrice=t.get("entry_price"),
                actualOutcome=t.get("actual_outcome"),
                pnl=t.get("pnl"),
                won=t.get("won"),
            )
            for t in trades
        ],
        equityCurve=[],  # Would need to rebuild from trades
    )


@app.get("/backtest/latest", response_model=BacktestRunResponse | None)
async def get_latest_backtest() -> BacktestRunResponse | None:
    """Get the most recent completed backtest run."""
    with db.db_conn(settings.db_path) as conn:
        run = db.get_latest_backtest_run(conn)
        if not run:
            return None
        
        trades = db.get_backtest_trades(conn, run["run_id"], limit=500)
    
    config_dict = run.get("config", {})
    
    # Build equity curve from trades
    equity_curve = []
    equity = 0.0
    for t in sorted(trades, key=lambda x: x["signal_timestamp"]):
        if t.get("won") is not None:
            equity += t.get("pnl") or 0
            equity_curve.append({
                "timestamp": t["signal_timestamp"],
                "equity": equity,
                "pnl": t.get("pnl") or 0,
            })
    
    return BacktestRunResponse(
        runId=run["run_id"],
        status=run.get("status", "unknown"),
        config=BacktestConfigModel(
            minConfidence=config_dict.get("min_confidence", 0.80),
            betSizing=config_dict.get("bet_sizing", "scaled"),
            baseBet=config_dict.get("base_bet", 100.0),
            maxBet=config_dict.get("max_bet", 500.0),
            lookbackDays=config_dict.get("lookback_days", 180),
            minParticipants=config_dict.get("min_participants", 2),
        ),
        totalTrades=run.get("total_trades", 0),
        winningTrades=run.get("winning_trades", 0),
        losingTrades=run.get("losing_trades", 0),
        pendingTrades=run.get("total_trades", 0) - run.get("winning_trades", 0) - run.get("losing_trades", 0),
        winRate=run.get("win_rate", 0.0),
        totalPnl=run.get("total_pnl", 0.0),
        totalInvested=run.get("total_invested", 0.0),
        roi=run.get("roi", 0.0),
        maxDrawdown=run.get("max_drawdown", 0.0),
        sharpeRatio=run.get("sharpe_ratio", 0.0),
        profitFactor=0.0,
        trades=[
            BacktestTradeModel(
                conditionId=t["condition_id"],
                title=t.get("title"),
                signalTimestamp=t["signal_timestamp"],
                predictedOutcome=t["predicted_outcome"],
                confidenceScore=t["confidence_score"],
                betSize=t["bet_size"],
                entryPrice=t.get("entry_price"),
                actualOutcome=t.get("actual_outcome"),
                pnl=t.get("pnl"),
                won=t.get("won"),
            )
            for t in trades
        ],
        equityCurve=equity_curve,
    )
