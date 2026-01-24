#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.market_filters import is_sports_market  # noqa: E402


@dataclass(frozen=True)
class SignalRecord:
    condition_id: str
    title: str | None
    signal_timestamp: int
    predicted_outcome: str
    actual_outcome: str
    confidence_score: float
    consensus_percent: float
    weighted_consensus_percent: float
    total_participants: int
    participants: int
    weighted_participants: float
    mean_entry: float
    tight_band: bool
    forced_by_perfect: bool
    won: bool


@dataclass(frozen=True)
class Candidate:
    min_confidence: float
    weighted_consensus_min: float
    min_weighted_participants: float
    ev_min: float
    entry_min: float
    entry_max: float
    require_tight_band: bool
    min_total_participants: int

    def key(self) -> tuple[Any, ...]:
        return (
            self.min_confidence,
            self.weighted_consensus_min,
            self.min_weighted_participants,
            self.ev_min,
            self.entry_min,
            self.entry_max,
            self.require_tight_band,
            self.min_total_participants,
        )


def _month_start(ts: int) -> datetime:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _add_months(dt: datetime, months: int) -> datetime:
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _month_ranges(start_ts: int, end_ts: int) -> list[tuple[int, int]]:
    start_dt = _month_start(start_ts)
    end_dt = _month_start(end_ts)
    ranges: list[tuple[int, int]] = []
    dt = start_dt
    while dt <= end_dt:
        next_dt = _add_months(dt, 1)
        ranges.append((int(dt.timestamp()), int(next_dt.timestamp())))
        dt = next_dt
    return ranges


def _passes_filter(record: SignalRecord, cand: Candidate) -> bool:
    if not record.forced_by_perfect and record.total_participants < cand.min_total_participants:
        return False
    if not record.forced_by_perfect and record.participants < 2:
        return False
    if cand.require_tight_band and not record.tight_band:
        return False
    if record.mean_entry < cand.entry_min or record.mean_entry >= cand.entry_max:
        return False
    if record.weighted_consensus_percent < cand.weighted_consensus_min:
        return False
    if not record.forced_by_perfect and record.weighted_participants < cand.min_weighted_participants:
        return False
    if record.mean_entry <= 0:
        return False
    ev = record.weighted_consensus_percent * (1 / record.mean_entry - 1) - (
        1 - record.weighted_consensus_percent
    )
    if ev < cand.ev_min:
        return False
    if record.confidence_score < cand.min_confidence:
        return False
    return True


def _compute_metrics(records: list[SignalRecord], cand: Candidate | None) -> dict[str, float]:
    trades = 0
    wins = 0
    total_pnl = 0.0
    for r in records:
        if cand and not _passes_filter(r, cand):
            continue
        trades += 1
        if r.won:
            wins += 1
        total_pnl += calculate_pnl(1.0, r.mean_entry, r.won)
    win_rate = wins / trades if trades else 0.0
    roi = total_pnl / trades if trades else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "roi": roi,
    }


def _build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    min_confidence = [0.2, 0.3, 0.4, 0.5]
    weighted_consensus_min = [0.55, 0.60, 0.65]
    min_weighted_participants = [1.0, 1.5, 2.0]
    ev_min = [0.0, 0.02, 0.03]
    entry_mins = [0.30, 0.35]
    entry_maxs = [0.60, 0.65]
    require_tight_band = [True, False]
    min_total_participants = [2, 3]

    for mc in min_confidence:
        for wcm in weighted_consensus_min:
            for mwp in min_weighted_participants:
                for ev in ev_min:
                    for emin in entry_mins:
                        for emax in entry_maxs:
                            if emin >= emax:
                                continue
                            for tb in require_tight_band:
                                for mtp in min_total_participants:
                                    candidates.append(
                                        Candidate(
                                            min_confidence=mc,
                                            weighted_consensus_min=wcm,
                                            min_weighted_participants=mwp,
                                            ev_min=ev,
                                            entry_min=emin,
                                            entry_max=emax,
                                            require_tight_band=tb,
                                            min_total_participants=mtp,
                                        )
                                    )
    return candidates


def _format_candidate(cand: Candidate) -> str:
    return (
        f"min_conf={cand.min_confidence:.2f}, weighted_consensus>={cand.weighted_consensus_min:.2f}, "
        f"weighted_support>={cand.min_weighted_participants:.2f}, ev>={cand.ev_min:.2f}, "
        f"entry=[{cand.entry_min:.2f},{cand.entry_max:.2f}), "
        f"tight_band={cand.require_tight_band}, min_participants={cand.min_total_participants}"
    )


def _fetch_resolutions(
    condition_ids: list[str],
    *,
    settings: Any,
    existing: dict[str, str],
) -> dict[str, str]:
    resolutions = dict(existing)
    missing = [cid for cid in condition_ids if cid not in resolutions]
    if not missing:
        return resolutions

    def fetch_one(condition_id: str) -> tuple[str, str] | None:
        url = f"{settings.clob_base}/markets/{condition_id}"
        try:
            req = Request(url, headers={"User-Agent": settings.user_agent})
            with urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    return None
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            if not data.get("closed"):
                return None
            for token in data.get("tokens", []):
                if token.get("winner"):
                    return condition_id, str(token.get("outcome"))
        except (URLError, JSONDecodeError, ValueError, TimeoutError):
            return None
        return None

    max_workers = max(2, int(settings.outbound_concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(fetch_one, missing):
            if not res:
                continue
            cid, winner = res
            resolutions[cid] = winner

    return resolutions


def _compute_band(prices: list[float]) -> tuple[float, float, float, float]:
    band_min = min(prices)
    band_max = max(prices)
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    stddev = math.sqrt(variance)
    return band_min, band_max, mean, stddev


def _compute_weighted_majority(
    recs: list[dict[str, Any]],
    accuracy_map: dict[str, float],
    default_weight: float = 0.5,
) -> tuple[str | None, float, float, float, dict[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    for rec in recs:
        wallet = str(rec.get("wallet") or "").lower()
        accuracy = accuracy_map.get(wallet, default_weight)
        weight = max(accuracy, 0.1)
        total_weight += weight
        totals[str(rec.get("outcome", ""))] += weight
    if not totals:
        return None, 0.0, 0.0, 0.0, totals
    leading_outcome, leading_weight = max(totals.items(), key=lambda kv: kv[1])
    weighted_consensus = leading_weight / total_weight if total_weight else 0.0
    return leading_outcome, weighted_consensus, leading_weight, total_weight, totals


def _compute_confidence_score(
    weighted_consensus: float,
    consensus_percent: float,
    tight_band: bool,
    participants: int,
    total_participants: int,
) -> float:
    if total_participants <= 1:
        max_score = 0.15
    elif total_participants == 2:
        max_score = 0.30
    elif total_participants == 3:
        max_score = 0.45
    elif total_participants <= 5:
        max_score = 0.55
    elif total_participants <= 10:
        max_score = 0.70
    else:
        max_score = 0.85

    weighted_score = weighted_consensus * 35
    raw_score = consensus_percent * 20
    tight_score = 15 if tight_band else 0
    cooked_score = 15
    participant_score = min(15, math.log1p(participants) * 3)

    raw_total = weighted_score + raw_score + tight_score + cooked_score + participant_score
    raw_normalized = raw_total / 100.0
    return min(raw_normalized, max_score)


def compute_signal_stats_at_time(
    trades: list[dict[str, Any]],
    cutoff_timestamp: int,
    *,
    accuracy_map: dict[str, float] | None = None,
    perfect_wallets: set[str] | None = None,
) -> dict[str, Any] | None:
    relevant_trades = [
        t for t in trades
        if t.get("side") == "BUY" and (t.get("timestamp") or 0) <= cutoff_timestamp
    ]
    if not relevant_trades:
        return None

    outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    title = None
    for t in relevant_trades:
        outcome = str(t.get("outcome", ""))
        outcomes[outcome].append(t)
        if title is None and t.get("title"):
            title = str(t.get("title"))

    if not outcomes:
        return None

    accuracy_map = accuracy_map or {}
    (
        weighted_outcome,
        weighted_consensus,
        weighted_participants,
        total_weight,
        weight_totals,
    ) = _compute_weighted_majority(relevant_trades, accuracy_map)
    if not weighted_outcome:
        return None

    perfect_outcomes: set[str] = set()
    if perfect_wallets:
        for t in relevant_trades:
            wallet = str(t.get("wallet") or "").lower()
            if wallet in perfect_wallets:
                perfect_outcomes.add(str(t.get("outcome", "")))

    forced_by_perfect = len(perfect_outcomes) == 1
    if forced_by_perfect:
        leading_outcome = next(iter(perfect_outcomes))
        leading_trades = outcomes.get(leading_outcome, [])
        leading_weight = weight_totals.get(leading_outcome, 0.0)
        weighted_participants = leading_weight
        weighted_consensus = leading_weight / total_weight if total_weight else 0.0
    else:
        leading_outcome = weighted_outcome
        leading_trades = outcomes.get(leading_outcome, [])

    total_participants = len(relevant_trades)
    participants = len(leading_trades)
    consensus_percent = participants / total_participants if total_participants else 0.0

    prices = []
    for t in leading_trades:
        price = t.get("price")
        if price is None:
            continue
        try:
            prices.append(float(price))
        except (TypeError, ValueError):
            continue

    mean_entry = None
    tight_band = False
    if prices:
        band_min, band_max, mean_entry, stddev = _compute_band(prices)
        tight_band = ((band_max - band_min) <= 0.03) or (stddev <= 0.01)

    confidence_score = _compute_confidence_score(
        weighted_consensus,
        consensus_percent,
        tight_band,
        participants,
        total_participants,
    )

    return {
        "condition_id": trades[0].get("condition_id"),
        "title": title,
        "leading_outcome": leading_outcome,
        "confidence_score": confidence_score,
        "consensus_percent": consensus_percent,
        "weighted_consensus_percent": weighted_consensus,
        "participants": participants,
        "total_participants": total_participants,
        "weighted_participants": weighted_participants,
        "mean_entry": mean_entry,
        "tight_band": tight_band,
        "signal_timestamp": cutoff_timestamp,
        "forced_by_perfect": forced_by_perfect,
    }


def calculate_pnl(bet_size: float, entry_price: float, won: bool) -> float:
    if won:
        if entry_price > 0:
            return bet_size * (1 / entry_price - 1)
        return bet_size
    return -bet_size


def _build_signal_records(
    trades: list[dict[str, Any]],
    *,
    resolutions: dict[str, str],
    min_participants: int,
    exclude_sports: bool,
    accuracy_map: dict[str, float],
    perfect_wallets: set[str],
) -> list[SignalRecord]:
    trades_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        cid = t.get("condition_id")
        if not cid:
            continue
        trades_by_condition[str(cid)].append(t)

    records: list[SignalRecord] = []
    for condition_id, cond_trades in trades_by_condition.items():
        if condition_id not in resolutions:
            continue
        sorted_trades = sorted(cond_trades, key=lambda x: x.get("timestamp") or 0)
        if not sorted_trades:
            continue
        if exclude_sports:
            event_slug = None
            slug = None
            title = None
            for t in sorted_trades:
                if title is None and t.get("title"):
                    title = str(t.get("title"))
                if event_slug is None and t.get("event_slug"):
                    event_slug = str(t.get("event_slug"))
                if slug is None and t.get("slug"):
                    slug = str(t.get("slug"))
                if title and event_slug and slug:
                    break
            if is_sports_market(event_slug=event_slug, slug=slug, title=title):
                continue

        buy_trades = [t for t in sorted_trades if t.get("side") == "BUY"]
        if len(buy_trades) < min_participants:
            continue
        signal_ts = buy_trades[min_participants - 1].get("timestamp") or 0
        signal = compute_signal_stats_at_time(
            sorted_trades,
            signal_ts,
            accuracy_map=accuracy_map,
            perfect_wallets=perfect_wallets,
        )
        if not signal:
            continue
        mean_entry = signal.get("mean_entry")
        if mean_entry is None:
            continue
        predicted = str(signal["leading_outcome"])
        actual = resolutions[condition_id]
        won = predicted == actual
        records.append(
            SignalRecord(
                condition_id=condition_id,
                title=signal.get("title"),
                signal_timestamp=signal_ts,
                predicted_outcome=predicted,
                actual_outcome=actual,
                confidence_score=float(signal["confidence_score"]),
                consensus_percent=float(signal["consensus_percent"]),
                weighted_consensus_percent=float(signal["weighted_consensus_percent"]),
                total_participants=int(signal["total_participants"]),
                participants=int(signal["participants"]),
                weighted_participants=float(signal["weighted_participants"]),
                mean_entry=float(mean_entry),
                tight_band=bool(signal["tight_band"]),
                forced_by_perfect=bool(signal.get("forced_by_perfect")),
                won=bool(won),
            )
        )

    return records


def _slice_records(
    records_by_month: list[list[SignalRecord]],
    indices: list[int],
) -> list[SignalRecord]:
    out: list[SignalRecord] = []
    for idx in indices:
        out.extend(records_by_month[idx])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward sweep to maximize win rate.")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--min-trades-train", type=int, default=30)
    parser.add_argument("--min-trades-val", type=int, default=10)
    parser.add_argument("--min-trades-test", type=int, default=10)
    parser.add_argument("--min-trades-constraint", type=int, default=20)
    parser.add_argument("--min-win-rate", type=float, default=0.70)
    parser.add_argument("--min-roi", type=float, default=0.0)
    parser.add_argument("--max-std", type=float, default=0.05)
    parser.add_argument("--min-splits", type=int, default=2)
    parser.add_argument("--include-sports", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    now = int(time.time())
    lookback_start = now - (args.lookback_days * 24 * 60 * 60)

    with db.db_conn(settings.db_path) as conn:
        trades = db.get_trades_in_timerange(conn, start_ts=lookback_start)
        if not trades:
            print("No trades found in lookback window.")
            return 1

        cached_resolutions = db.get_all_resolutions(conn)
        accuracy_map = db.get_wallet_accuracy_map(conn)
        perf_map = db.get_wallet_performance_map(conn)
        perfect_wallets = {
            wallet
            for wallet, stats in perf_map.items()
            if stats.get("win_rate", 0.0) >= 1.0
            and stats.get("resolved_trades", 0) >= 20
        }

    condition_ids = sorted({t.get("condition_id") for t in trades if t.get("condition_id")})
    if not condition_ids:
        print("No condition IDs found.")
        return 1

    resolutions = _fetch_resolutions(
        condition_ids,
        settings=settings,
        existing=cached_resolutions,
    )

    with db.db_conn(settings.db_path) as conn:
        missing_resolutions = [
            {"condition_id": cid, "winning_outcome": winner}
            for cid, winner in resolutions.items()
            if cid not in cached_resolutions
        ]
        if missing_resolutions:
            try:
                db.bulk_upsert_resolutions(conn, missing_resolutions)
            except sqlite3.OperationalError:
                print("Warning: DB locked while caching resolutions; continuing without DB write.")

    records = _build_signal_records(
        trades,
        resolutions=resolutions,
        min_participants=args.min_participants,
        exclude_sports=not args.include_sports,
        accuracy_map=accuracy_map,
        perfect_wallets=perfect_wallets,
    )
    if not records:
        print("No resolved signals found after filtering.")
        return 1

    min_ts = min(r.signal_timestamp for r in records)
    max_ts = max(r.signal_timestamp for r in records)
    month_ranges = _month_ranges(min_ts, max_ts)

    if len(month_ranges) < 6:
        print("Not enough monthly data for a 6-month walk-forward split.")
        return 1

    records_by_month: list[list[SignalRecord]] = [[] for _ in month_ranges]
    for r in records:
        for idx, (start_ts, end_ts) in enumerate(month_ranges):
            if start_ts <= r.signal_timestamp < end_ts:
                records_by_month[idx].append(r)
                break

    candidates = _build_candidates()
    baseline = _compute_metrics(records, None)

    aggregate: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    aggregate_trades: dict[tuple[Any, ...], dict[str, list[int]]] = {}

    for i in range(0, len(month_ranges) - 5):
        train_idx = list(range(i, i + 4))
        val_idx = [i + 4]
        test_idx = [i + 5]

        train_records = _slice_records(records_by_month, train_idx)
        val_records = _slice_records(records_by_month, val_idx)
        test_records = _slice_records(records_by_month, test_idx)

        if not train_records or not val_records or not test_records:
            continue

        for cand in candidates:
            train_metrics = _compute_metrics(train_records, cand)
            val_metrics = _compute_metrics(val_records, cand)
            test_metrics = _compute_metrics(test_records, cand)

            if (
                train_metrics["trades"] < args.min_trades_train
                or val_metrics["trades"] < args.min_trades_val
                or test_metrics["trades"] < args.min_trades_test
            ):
                continue

            key = cand.key()
            aggregate.setdefault(
                key,
                {
                    "train_win": [],
                    "val_win": [],
                    "test_win": [],
                    "train_roi": [],
                    "val_roi": [],
                    "test_roi": [],
                },
            )
            aggregate_trades.setdefault(key, {"train": [], "val": [], "test": []})

            aggregate[key]["train_win"].append(train_metrics["win_rate"])
            aggregate[key]["val_win"].append(val_metrics["win_rate"])
            aggregate[key]["test_win"].append(test_metrics["win_rate"])
            aggregate[key]["train_roi"].append(train_metrics["roi"])
            aggregate[key]["val_roi"].append(val_metrics["roi"])
            aggregate[key]["test_roi"].append(test_metrics["roi"])
            aggregate_trades[key]["train"].append(int(train_metrics["trades"]))
            aggregate_trades[key]["val"].append(int(val_metrics["trades"]))
            aggregate_trades[key]["test"].append(int(test_metrics["trades"]))

    scored: list[tuple[Candidate, dict[str, float]]] = []
    for cand in candidates:
        key = cand.key()
        if key not in aggregate:
            continue
        splits = len(aggregate[key]["test_win"])
        if splits < args.min_splits:
            continue
        avg_train = statistics.mean(aggregate[key]["train_win"])
        avg_val = statistics.mean(aggregate[key]["val_win"])
        avg_test = statistics.mean(aggregate[key]["test_win"])
        avg_train_roi = statistics.mean(aggregate[key]["train_roi"])
        avg_val_roi = statistics.mean(aggregate[key]["val_roi"])
        avg_test_roi = statistics.mean(aggregate[key]["test_roi"])
        std_test = statistics.pstdev(aggregate[key]["test_win"])

        avg_test_trades = statistics.mean(aggregate_trades[key]["test"])
        if std_test > args.max_std:
            continue
        if abs(avg_train - avg_val) > args.max_std:
            continue
        if avg_test < args.min_win_rate:
            continue
        if avg_test_roi < args.min_roi:
            continue
        if avg_test_trades < args.min_trades_constraint:
            continue
        scored.append(
            (
                cand,
                {
                    "splits": splits,
                    "avg_train": avg_train,
                    "avg_val": avg_val,
                    "avg_test": avg_test,
                    "avg_train_roi": avg_train_roi,
                    "avg_val_roi": avg_val_roi,
                    "avg_test_roi": avg_test_roi,
                    "std_test": std_test,
                    "avg_test_trades": avg_test_trades,
                },
            )
        )

    scored.sort(key=lambda x: (x[1]["avg_test"], x[1]["avg_test_trades"]), reverse=True)

    print("Baseline (no filters):")
    print(f"  trades={baseline['trades']} win_rate={baseline['win_rate']:.3f} roi={baseline['roi']:.3f}")
    print("")
    print("Top stable candidates (by avg test win rate):")
    if not scored:
        print("  No candidates met stability/trade-count constraints.")
        return 0

    for cand, metrics in scored[:10]:
        print(
            f"  {_format_candidate(cand)} | splits={metrics['splits']} "
            f"train={metrics['avg_train']:.3f} val={metrics['avg_val']:.3f} "
            f"test={metrics['avg_test']:.3f} "
            f"test_roi={metrics['avg_test_roi']:.3f} std={metrics['std_test']:.3f} "
            f"avg_test_trades={metrics['avg_test_trades']:.1f}"
        )

    best = scored[0][0]
    print("")
    print("Recommended filters:")
    print(f"  {_format_candidate(best)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
