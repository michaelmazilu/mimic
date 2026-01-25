#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app import backtest, db  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402


def _build_settings(base: Settings, db_path: str | None) -> Settings:
    if not db_path:
        return base
    return Settings(
        db_path=db_path,
        n_wallets=base.n_wallets,
        trades_limit=base.trades_limit,
        refresh_interval_sec=base.refresh_interval_sec,
        outbound_concurrency=base.outbound_concurrency,
        enable_pricing=base.enable_pricing,
        data_api_base=base.data_api_base,
        clob_base=base.clob_base,
        user_agent=base.user_agent,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mimic backtest with configurable sizing.")
    parser.add_argument("--db-path", default=None, help="Path to SQLite DB (overrides MIMIC_DB_PATH).")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--bet-sizing", default="bankroll", choices=["flat", "scaled", "kelly", "bankroll"])
    parser.add_argument("--base-bet", type=float, default=100.0)
    parser.add_argument("--max-bet", type=float, default=500.0)
    parser.add_argument("--starting-bankroll", type=float, default=200.0)
    parser.add_argument("--bet-fraction", type=float, default=0.02)
    parser.add_argument("--min-participants", type=int, default=2)
    parser.add_argument("--min-total-participants", type=int, default=2)
    parser.add_argument("--weighted-consensus-min", type=float, default=0.0)
    parser.add_argument("--min-weighted-participants", type=float, default=0.0)
    parser.add_argument("--ev-min", type=float, default=-1.0)
    parser.add_argument("--entry-min", type=float, default=0.0)
    parser.add_argument("--entry-max", type=float, default=1.0)
    parser.add_argument("--require-tight-band", action="store_true")
    parser.add_argument("--include-sports", action="store_true")
    parser.add_argument(
        "--no-single-perfect",
        dest="allow_single_perfect",
        action="store_false",
        help="Disable single-perfect override.",
    )
    parser.add_argument("--perfect-accuracy", type=float, default=1.0)
    parser.add_argument("--min-perfect-resolved-trades", type=int, default=20)
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    settings = _build_settings(get_settings(), args.db_path)
    config = backtest.BacktestConfig(
        min_confidence=args.min_confidence,
        bet_sizing=args.bet_sizing,
        base_bet=args.base_bet,
        max_bet=args.max_bet,
        starting_bankroll=args.starting_bankroll,
        bet_fraction=args.bet_fraction,
        weighted_consensus_min=args.weighted_consensus_min,
        min_weighted_participants=args.min_weighted_participants,
        ev_min=args.ev_min,
        allow_single_perfect=args.allow_single_perfect,
        perfect_accuracy=args.perfect_accuracy,
        min_perfect_resolved_trades=args.min_perfect_resolved_trades,
        lookback_days=args.lookback_days,
        min_participants=args.min_participants,
        min_total_participants=args.min_total_participants,
        entry_min=args.entry_min,
        entry_max=args.entry_max,
        require_tight_band=args.require_tight_band,
        exclude_sports=not args.include_sports,
    )

    sem = asyncio.Semaphore(settings.outbound_concurrency)
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        conn = db.connect(settings.db_path, timeout=60.0)
        try:
            result = await backtest.run_backtest(
                conn,
                settings,
                config,
                client=client,
                sem=sem,
            )
            conn.commit()
        finally:
            conn.close()

    print("Backtest run completed.")
    print(f"run_id={result.run_id}")
    print(f"trades={result.total_trades}")
    print(f"win_rate={result.win_rate:.3f}")
    print(f"roi={result.roi:.3f}")
    print(f"pnl={result.total_pnl:+.2f}")
    print(f"sharpe={result.sharpe_ratio:.3f}")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
