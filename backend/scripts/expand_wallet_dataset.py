#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import time
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app import db, ingest  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402


def _build_settings(
    base: Settings,
    *,
    db_path: str | None,
    trades_limit: int | None,
    outbound_concurrency: int | None,
) -> Settings:
    return Settings(
        db_path=db_path or base.db_path,
        n_wallets=base.n_wallets,
        trades_limit=trades_limit if trades_limit is not None else base.trades_limit,
        refresh_interval_sec=base.refresh_interval_sec,
        outbound_concurrency=(
            outbound_concurrency if outbound_concurrency is not None else base.outbound_concurrency
        ),
        enable_pricing=base.enable_pricing,
        data_api_base=base.data_api_base,
        clob_base=base.clob_base,
        user_agent=base.user_agent,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch top wallets, store trades, and optionally backfill resolutions."
    )
    parser.add_argument("--db-path", default=None, help="Path to SQLite DB (overrides MIMIC_DB_PATH).")
    parser.add_argument("--top-wallets", type=int, default=10000)
    parser.add_argument(
        "--wallet-source",
        choices=["leaderboard", "trades"],
        default="leaderboard",
        help="Where to source wallet addresses from.",
    )
    parser.add_argument(
        "--trades-limit",
        type=int,
        default=None,
        help="Max trades per wallet (single-page mode) or cap for paginated mode.",
    )
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--trades-page-limit",
        type=int,
        default=100,
        help="Page size for trade pagination (global or per-wallet when paging).",
    )
    parser.add_argument(
        "--trades-page-delay",
        type=float,
        default=0.3,
        help="Delay between paged trade requests (seconds).",
    )
    parser.add_argument(
        "--trades-max-pages",
        type=int,
        default=None,
        help="Max pages to fetch when paging trades (global or per-wallet).",
    )
    parser.add_argument(
        "--paginate-wallet-trades",
        action="store_true",
        help="Page through all wallet trades until the lookback window is exhausted.",
    )
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--fetch-resolutions", action="store_true", default=True)
    parser.add_argument("--no-fetch-resolutions", dest="fetch_resolutions", action="store_false")
    return parser.parse_args()


def _trade_timestamp(trade: dict[str, Any]) -> int | None:
    ts = trade.get("timestamp")
    try:
        return int(ts) if ts is not None else None
    except (TypeError, ValueError):
        return None


def _is_descending(values: list[int]) -> bool:
    return all(a >= b for a, b in zip(values, values[1:], strict=False))


def _trade_key(trade: dict[str, Any]) -> tuple[Any, ...]:
    return (
        trade.get("transactionHash")
        or trade.get("tx_hash")
        or trade.get("transaction_hash"),
        trade.get("conditionId") or trade.get("condition_id"),
        trade.get("outcome"),
        trade.get("side"),
        trade.get("timestamp"),
        trade.get("price"),
        trade.get("size"),
        trade.get("asset") or trade.get("asset_id") or trade.get("token_id") or trade.get("tokenId"),
    )


async def _fetch_trades_page_for_wallet(
    wallet: str,
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limit: int,
    offset: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    url = f"{settings.data_api_base}/trades"
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.get(
                    url,
                    params={"user": wallet, "limit": int(limit), "offset": int(offset)},
                    headers={"User-Agent": settings.user_agent},
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception:
            if attempt >= max_retries - 1:
                return []
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return []


async def _fetch_trades_for_wallet(
    wallet: str,
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limit: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    return await _fetch_trades_page_for_wallet(
        wallet,
        settings=settings,
        client=client,
        sem=sem,
        limit=limit,
        offset=0,
        max_retries=max_retries,
    )


async def _fetch_trades_for_wallet_paginated(
    wallet: str,
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    page_limit: int,
    max_retries: int,
    lookback_start: int,
    max_pages: int | None,
    page_delay: float,
    max_total_trades: int | None,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    offset = 0
    pages = 0
    last_first_key: tuple[Any, ...] | None = None

    while True:
        page = await _fetch_trades_page_for_wallet(
            wallet,
            settings=settings,
            client=client,
            sem=sem,
            limit=page_limit,
            offset=offset,
            max_retries=max_retries,
        )
        if not page:
            break

        first_key = _trade_key(page[0])
        if last_first_key is not None and first_key == last_first_key:
            break
        last_first_key = first_key

        timestamps = []
        page_recent = []
        for trade in page:
            ts = _trade_timestamp(trade)
            if ts is not None:
                timestamps.append(ts)
            if ts is None or ts < lookback_start:
                continue
            page_recent.append(trade)

        if page_recent:
            trades.extend(page_recent)
            if max_total_trades is not None and len(trades) >= max_total_trades:
                trades = trades[:max_total_trades]
                break

        if timestamps:
            max_ts = max(timestamps)
            min_ts = min(timestamps)
            if max_ts < lookback_start:
                break
            if min_ts < lookback_start and _is_descending(timestamps):
                break

        if len(page) < page_limit:
            break

        offset += page_limit
        pages += 1
        if max_pages is not None and pages >= max_pages:
            break
        if page_delay > 0:
            await asyncio.sleep(page_delay)

    return trades


async def _fetch_global_trades_page(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    offset: int,
    limit: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    url = f"{settings.data_api_base}/trades"
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.get(
                    url,
                    params={"limit": int(limit), "offset": int(offset)},
                    headers={"User-Agent": settings.user_agent},
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = max(backoff, float(retry_after))
                    except ValueError:
                        pass
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception:
            if attempt >= max_retries - 1:
                return []
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return []


async def _collect_wallets_from_trades(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    target_count: int,
    page_limit: int,
    page_delay: float,
    max_pages: int | None,
    max_retries: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    wallets: list[str] = []
    seen: set[str] = set()
    offset = 0
    pages = 0

    while len(wallets) < target_count:
        trades = await _fetch_global_trades_page(
            settings=settings,
            client=client,
            sem=sem,
            offset=offset,
            limit=page_limit,
            max_retries=max_retries,
        )
        if not trades:
            break
        for t in trades:
            wallet = t.get("proxyWallet") or t.get("wallet") or t.get("address")
            if not wallet:
                continue
            wallet_lc = str(wallet).lower()
            if wallet_lc in seen:
                continue
            seen.add(wallet_lc)
            wallets.append(wallet_lc)
            entries.append({"wallet": wallet_lc})
            if len(wallets) >= target_count:
                break
        offset += page_limit
        pages += 1
        print(f"wallet_source=trades wallets_collected={len(wallets)} offset={offset}")
        if max_pages is not None and pages >= max_pages:
            break
        if page_delay > 0:
            await asyncio.sleep(page_delay)

    return entries, wallets


async def _run() -> int:
    args = _parse_args()
    settings = _build_settings(
        get_settings(),
        db_path=args.db_path,
        trades_limit=args.trades_limit,
        outbound_concurrency=args.concurrency,
    )
    db.init_db(settings.db_path)
    lookback_start = int(time.time()) - (args.lookback_days * 24 * 60 * 60)

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.wallet_source == "leaderboard":
            leaderboard = await ingest.fetch_leaderboard(
                settings,
                client=client,
                sem=sem,
                target_count=args.top_wallets,
            )

            entries = []
            wallets = []
            seen: set[str] = set()
            for entry in leaderboard:
                wallet = entry.get("proxyWallet") or entry.get("wallet") or entry.get("address")
                if not wallet:
                    continue
                wallet_lc = str(wallet).lower()
                if wallet_lc in seen:
                    continue
                seen.add(wallet_lc)
                entries.append(entry)
                wallets.append(wallet_lc)
        else:
            entries, wallets = await _collect_wallets_from_trades(
                settings=settings,
                client=client,
                sem=sem,
                target_count=args.top_wallets,
                page_limit=max(1, args.trades_page_limit),
                page_delay=max(0.0, args.trades_page_delay),
                max_pages=args.trades_max_pages,
                max_retries=args.max_retries,
            )

        with db.db_conn(settings.db_path) as conn:
            existing_wallets = {w.lower() for w in db.get_all_wallets(conn)}
            wallet_upserts = db.upsert_wallets(conn, entries)

        if args.refresh_existing:
            wallets_to_fetch = wallets
        else:
            wallets_to_fetch = [w for w in wallets if w not in existing_wallets]

        print(f"wallet_source={args.wallet_source} wallets_found={len(wallets)}")
        print(f"wallets_upserted={wallet_upserts}")
        print(f"wallets_to_fetch={len(wallets_to_fetch)} refresh_existing={args.refresh_existing}")

        trade_inserts = 0
        condition_ids: set[str] = set()

        if wallets_to_fetch:
            batch_size = max(1, args.batch_size)
            page_limit = max(1, args.trades_page_limit)
            page_delay = max(0.0, args.trades_page_delay)
            for offset in range(0, len(wallets_to_fetch), batch_size):
                batch = wallets_to_fetch[offset:offset + batch_size]
                if args.paginate_wallet_trades:
                    tasks = [
                        _fetch_trades_for_wallet_paginated(
                            w,
                            settings=settings,
                            client=client,
                            sem=sem,
                            page_limit=page_limit,
                            max_retries=args.max_retries,
                            lookback_start=lookback_start,
                            max_pages=args.trades_max_pages,
                            page_delay=page_delay,
                            max_total_trades=args.trades_limit,
                        )
                        for w in batch
                    ]
                else:
                    tasks = [
                        _fetch_trades_for_wallet(
                            w,
                            settings=settings,
                            client=client,
                            sem=sem,
                            limit=settings.trades_limit,
                            max_retries=args.max_retries,
                        )
                        for w in batch
                    ]
                results = await asyncio.gather(*tasks)
                with db.db_conn(settings.db_path) as conn:
                    for wallet, trades in zip(batch, results, strict=True):
                        filtered = []
                        for t in trades:
                            ts = _trade_timestamp(t)
                            if args.paginate_wallet_trades and (ts is None or ts < lookback_start):
                                continue
                            filtered.append(t)
                            if ts is None or ts < lookback_start:
                                continue
                            cid = t.get("conditionId") or t.get("condition_id")
                            if cid:
                                condition_ids.add(str(cid))
                        trade_inserts += db.insert_trades(conn, wallet, filtered)
                done = min(offset + batch_size, len(wallets_to_fetch))
                print(f"fetched_wallets={done}/{len(wallets_to_fetch)}")

        if args.fetch_resolutions and condition_ids:
            with db.db_conn(settings.db_path) as conn:
                cached_resolutions = db.get_all_resolutions(conn)
            missing_ids = [cid for cid in condition_ids if cid not in cached_resolutions]
            print(f"resolution_candidates={len(condition_ids)} missing_resolutions={len(missing_ids)}")
            if missing_ids:
                fetched = await ingest.fetch_market_resolutions(
                    missing_ids,
                    settings,
                    client=client,
                    sem=sem,
                )
                if fetched:
                    with db.db_conn(settings.db_path) as conn:
                        db.bulk_upsert_resolutions(
                            conn,
                            [
                                {"condition_id": cid, "winning_outcome": outcome}
                                for cid, outcome in fetched.items()
                            ],
                        )

    print(f"trade_inserts={trade_inserts}")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
