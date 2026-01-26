#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app import db, wallet_selection  # noqa: E402
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
    parser = argparse.ArgumentParser(
        description="Sweep min_accuracy and min_avg_roi thresholds for wallet selection."
    )
    parser.add_argument("--db-path", default=None, help="Path to SQLite DB (overrides MIMIC_DB_PATH).")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--top-wallets", type=int, default=10000)
    parser.add_argument("--accuracy-step", type=float, default=0.001)
    parser.add_argument("--roi-step", type=float, default=0.001)
    parser.add_argument("--accuracy-min", type=float, default=0.0)
    parser.add_argument("--accuracy-max", type=float, default=1.0)
    parser.add_argument("--roi-min", type=float, default=0.0)
    parser.add_argument("--roi-max", type=float, default=None)
    parser.add_argument(
        "--min-resolved-trades",
        type=int,
        default=0,
        help="Minimum resolved trades required for a threshold combo to be considered.",
    )
    parser.add_argument(
        "--optimize-for",
        choices=["pnl", "roi"],
        default="roi",
        help="Objective for ranking threshold combinations.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of best combos to print.")
    return parser.parse_args()


def _scale_for_step(step: float) -> int:
    scale = int(round(1.0 / step))
    if scale <= 0 or abs(step * scale - 1.0) > 1e-9:
        raise ValueError(f"step {step} must evenly divide 1.0")
    return scale


def _grid_steps(min_value: float, max_value: float, scale: int) -> int:
    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")
    return int(math.floor((max_value - min_value) * scale + 1e-9)) + 1


def _to_index(value: float, min_value: float, max_value: float, scale: int) -> int | None:
    if value < min_value:
        return None
    if value > max_value:
        value = max_value
    return int(math.floor((value - min_value) * scale + 1e-9))


def _suffix_sum(arr: np.ndarray) -> np.ndarray:
    rev = arr[::-1, ::-1]
    rev = rev.cumsum(axis=0).cumsum(axis=1)
    return rev[::-1, ::-1]


def main() -> int:
    args = _parse_args()
    settings = _build_settings(get_settings(), args.db_path)
    accuracy_scale = _scale_for_step(args.accuracy_step)
    roi_scale = _scale_for_step(args.roi_step)

    now = int(time.time())
    lookback_start = now - (args.lookback_days * 24 * 60 * 60)

    conn = db.connect(settings.db_path, timeout=60.0)
    try:
        top_wallets = db.get_all_wallets(conn, limit=args.top_wallets)
        wallet_set = {str(w).lower() for w in top_wallets if w}
        all_trades = db.get_trades_in_timerange(conn, start_ts=lookback_start, side="BUY")
        resolutions = db.get_all_resolutions(conn)
    finally:
        conn.close()

    trades_by_wallet: dict[str, list[dict]] = defaultdict(list)
    total_trades = 0
    for t in all_trades:
        wallet = str(t.get("wallet") or "").lower()
        if wallet_set and wallet not in wallet_set:
            continue
        if not wallet:
            continue
        trades_by_wallet[wallet].append(t)
        total_trades += 1

    selection_cfg = wallet_selection.SelectionConfig()
    metrics = wallet_selection.compute_wallet_metrics(
        trades_by_wallet,
        resolutions,
        now_ts=now,
        config=selection_cfg,
    )

    eligible: list[wallet_selection.WalletMetrics] = []
    for m in metrics.values():
        if m.is_bot:
            continue
        if m.sports_ratio > selection_cfg.max_sports_ratio:
            continue
        if m.resolved_trades < selection_cfg.min_resolved_trades:
            continue
        eligible.append(m)

    if not eligible:
        print("No eligible wallets after filters.")
        return 1

    roi_min = args.roi_min
    roi_max = args.roi_max
    if roi_max is None:
        roi_max = max(roi_min, max(m.avg_roi for m in eligible))

    acc_min = args.accuracy_min
    acc_max = args.accuracy_max

    acc_steps = _grid_steps(acc_min, acc_max, accuracy_scale)
    roi_steps = _grid_steps(roi_min, roi_max, roi_scale)

    pnl_grid = np.zeros((acc_steps, roi_steps), dtype=np.float64)
    count_grid = np.zeros((acc_steps, roi_steps), dtype=np.int32)
    resolved_grid = np.zeros((acc_steps, roi_steps), dtype=np.int32)

    skipped_roi = 0
    for m in eligible:
        if m.avg_roi < roi_min:
            skipped_roi += 1
            continue
        acc_idx = _to_index(m.accuracy_lb, acc_min, acc_max, accuracy_scale)
        roi_idx = _to_index(m.avg_roi, roi_min, roi_max, roi_scale)
        if acc_idx is None or roi_idx is None:
            continue
        pnl_grid[acc_idx, roi_idx] += m.total_pnl
        count_grid[acc_idx, roi_idx] += 1
        resolved_grid[acc_idx, roi_idx] += int(m.resolved_trades)

    pnl_suffix = _suffix_sum(pnl_grid)
    count_suffix = _suffix_sum(count_grid)
    resolved_suffix = _suffix_sum(resolved_grid)
    roi_suffix = np.full_like(pnl_suffix, -np.inf, dtype=np.float64)
    roi_mask = resolved_suffix >= max(1, args.min_resolved_trades)
    roi_suffix[roi_mask] = pnl_suffix[roi_mask] / resolved_suffix[roi_mask]

    objective_grid = pnl_suffix if args.optimize_for == "pnl" else roi_suffix
    best_idx = np.unravel_index(np.argmax(objective_grid), objective_grid.shape)
    best_acc = acc_min + best_idx[0] / accuracy_scale
    best_roi = roi_min + best_idx[1] / roi_scale
    best_pnl = float(pnl_suffix[best_idx])
    best_roi_value = float(roi_suffix[best_idx]) if roi_suffix[best_idx] != -np.inf else float("nan")
    best_wallets = int(count_suffix[best_idx])
    best_resolved = int(resolved_suffix[best_idx])

    print("Sweep summary")
    print(f"lookback_days={args.lookback_days}")
    print(f"wallets_in_scope={len(wallet_set)}")
    print(f"trades_in_scope={total_trades}")
    print(f"eligible_wallets={len(eligible)} (min_resolved={selection_cfg.min_resolved_trades}, "
          f"max_sports_ratio={selection_cfg.max_sports_ratio}, bots_filtered=True)")
    print(f"accuracy_range=[{acc_min:.3f}, {acc_max:.3f}] step={args.accuracy_step}")
    print(f"roi_range=[{roi_min:.3f}, {roi_max:.3f}] step={args.roi_step}")
    if args.min_resolved_trades:
        print(f"min_resolved_trades={args.min_resolved_trades}")
    print(f"grid_size={acc_steps}x{roi_steps} ({acc_steps * roi_steps} combos)")
    if skipped_roi:
        print(f"wallets_skipped_below_roi_min={skipped_roi}")
    print("")
    print(f"Best combo (optimize_for={args.optimize_for})")
    print(
        f"min_accuracy={best_acc:.3f} min_avg_roi={best_roi:.3f} "
        f"roi={best_roi_value:.3f} total_pnl={best_pnl:+.2f} "
        f"wallets={best_wallets} resolved_trades={best_resolved}"
    )

    top_k = max(1, int(args.top_k))
    flat = objective_grid.ravel()
    if top_k > 1:
        k = min(top_k, flat.size)
        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
        print("")
        print(f"Top {k} combos by {args.optimize_for}")
        for rank, idx in enumerate(top_idx, start=1):
            acc_idx, roi_idx = np.unravel_index(idx, pnl_suffix.shape)
            acc_val = acc_min + acc_idx / accuracy_scale
            roi_val = roi_min + roi_idx / roi_scale
            pnl_val = float(pnl_suffix[acc_idx, roi_idx])
            roi_val_metric = float(roi_suffix[acc_idx, roi_idx])
            wallets_val = int(count_suffix[acc_idx, roi_idx])
            resolved_val = int(resolved_suffix[acc_idx, roi_idx])
            print(
                f"{rank:>2}. min_accuracy={acc_val:.3f} min_avg_roi={roi_val:.3f} "
                f"roi={roi_val_metric:.3f} total_pnl={pnl_val:+.2f} "
                f"wallets={wallets_val} resolved_trades={resolved_val}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
