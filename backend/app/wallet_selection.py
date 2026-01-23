from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .market_filters import is_sports_market


@dataclass(frozen=True)
class BotFilterConfig:
    min_samples: int = 20
    max_trades_per_day: float = 120.0
    min_median_gap_sec: int = 90
    max_rapid_fire_ratio: float = 0.50


@dataclass(frozen=True)
class SelectionConfig:
    min_resolved_trades: int = 10
    max_sports_ratio: float = 0.50
    bot_filters: BotFilterConfig = BotFilterConfig()


@dataclass(frozen=True)
class WalletMetrics:
    wallet: str
    total_trades: int
    resolved_trades: int
    wins: int
    win_rate: float
    total_pnl: float
    avg_roi: float
    recent_trades_7d: int
    recent_won_7d: int
    recent_accuracy_7d: float
    recent_trades_30d: int
    recent_won_30d: int
    recent_accuracy_30d: float
    last_trade_ts: int | None
    trades_per_day: float | None
    median_gap_sec: float | None
    rapid_fire_ratio: float | None
    sports_trades: int
    sports_ratio: float
    is_bot: bool


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _trade_is_sports(trade: dict[str, Any]) -> bool:
    return is_sports_market(
        event_slug=trade.get("eventSlug") or trade.get("event_slug"),
        slug=trade.get("slug"),
        title=trade.get("title"),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calculate_pnl(entry_price: float | None, *, won: bool) -> float:
    if won:
        if entry_price and entry_price > 0:
            return 1.0 * (1 / entry_price - 1)
        return 1.0
    return -1.0


def _compute_bot_metrics(
    timestamps: list[int],
    *,
    config: BotFilterConfig,
) -> tuple[float | None, float | None, float | None, bool]:
    if len(timestamps) < 2:
        return None, None, None, False
    sorted_ts = sorted(timestamps)
    span = sorted_ts[-1] - sorted_ts[0]
    trades_per_day = None
    if span > 0:
        trades_per_day = len(sorted_ts) / (span / 86400)
    else:
        trades_per_day = float("inf") if len(sorted_ts) >= config.min_samples else None

    gaps = [b - a for a, b in zip(sorted_ts, sorted_ts[1:], strict=False) if b > a]
    median_gap = _median(gaps) if gaps else None
    rapid_fire_ratio = None
    if gaps:
        rapid_fire_ratio = sum(1 for g in gaps if g <= 60) / len(gaps)

    is_bot = False
    if len(sorted_ts) >= config.min_samples:
        if trades_per_day is not None and trades_per_day >= config.max_trades_per_day:
            is_bot = True
        if median_gap is not None and median_gap <= config.min_median_gap_sec:
            is_bot = True
        if rapid_fire_ratio is not None and rapid_fire_ratio >= config.max_rapid_fire_ratio:
            is_bot = True

    return trades_per_day, median_gap, rapid_fire_ratio, is_bot


def compute_wallet_metrics(
    trades_by_wallet: dict[str, list[dict[str, Any]]],
    resolutions: dict[str, str],
    *,
    now_ts: int,
    config: SelectionConfig,
) -> dict[str, WalletMetrics]:
    metrics: dict[str, WalletMetrics] = {}
    seven_days_ago = now_ts - (7 * 24 * 60 * 60)
    thirty_days_ago = now_ts - (30 * 24 * 60 * 60)

    for wallet, trades in trades_by_wallet.items():
        buy_trades = [t for t in trades if t.get("side") == "BUY"]
        if not buy_trades:
            continue

        timestamps = [t.get("timestamp") for t in buy_trades if t.get("timestamp") is not None]
        last_trade_ts = max(timestamps) if timestamps else None

        sports_trades = 0
        resolved_trades = 0
        wins = 0
        total_pnl = 0.0
        recent_trades_7d = 0
        recent_won_7d = 0
        recent_trades_30d = 0
        recent_won_30d = 0

        for t in buy_trades:
            if _trade_is_sports(t):
                sports_trades += 1
                continue
            condition_id = t.get("conditionId") or t.get("condition_id")
            if not condition_id:
                continue
            winner = resolutions.get(str(condition_id))
            if not winner:
                continue
            resolved_trades += 1
            won = str(t.get("outcome")) == str(winner)
            if won:
                wins += 1
            entry_price = _safe_float(t.get("price"))
            total_pnl += _calculate_pnl(entry_price, won=won)
            ts = t.get("timestamp") or 0
            if ts >= thirty_days_ago:
                recent_trades_30d += 1
                if won:
                    recent_won_30d += 1
            if ts >= seven_days_ago:
                recent_trades_7d += 1
                if won:
                    recent_won_7d += 1

        total_trades = len(buy_trades)
        win_rate = wins / resolved_trades if resolved_trades else 0.0
        avg_roi = total_pnl / resolved_trades if resolved_trades else 0.0
        recent_accuracy_7d = recent_won_7d / recent_trades_7d if recent_trades_7d else 0.0
        recent_accuracy_30d = recent_won_30d / recent_trades_30d if recent_trades_30d else 0.0
        sports_ratio = sports_trades / total_trades if total_trades else 0.0

        trades_per_day, median_gap, rapid_fire_ratio, is_bot = _compute_bot_metrics(
            timestamps,
            config=config.bot_filters,
        )

        metrics[wallet] = WalletMetrics(
            wallet=wallet,
            total_trades=total_trades,
            resolved_trades=resolved_trades,
            wins=wins,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_roi=avg_roi,
            recent_trades_7d=recent_trades_7d,
            recent_won_7d=recent_won_7d,
            recent_accuracy_7d=recent_accuracy_7d,
            recent_trades_30d=recent_trades_30d,
            recent_won_30d=recent_won_30d,
            recent_accuracy_30d=recent_accuracy_30d,
            last_trade_ts=last_trade_ts,
            trades_per_day=trades_per_day,
            median_gap_sec=median_gap,
            rapid_fire_ratio=rapid_fire_ratio,
            sports_trades=sports_trades,
            sports_ratio=sports_ratio,
            is_bot=is_bot,
        )

    return metrics


def select_wallets_by_accuracy(
    metrics: Iterable[WalletMetrics],
    *,
    limit: int,
    config: SelectionConfig,
) -> list[str]:
    metrics_list = list(metrics)

    def is_eligible(m: WalletMetrics) -> bool:
        return (
            not m.is_bot
            and m.sports_ratio <= config.max_sports_ratio
            and m.resolved_trades >= config.min_resolved_trades
        )

    def sort_key(m: WalletMetrics) -> tuple[float, int, float, int]:
        return (
            m.win_rate,
            m.resolved_trades,
            m.recent_accuracy_30d,
            m.total_trades,
        )

    eligible = [m for m in metrics_list if is_eligible(m)]
    eligible.sort(key=sort_key, reverse=True)

    if len(eligible) >= limit:
        return [m.wallet for m in eligible[:limit]]

    # Fallback: include non-bots with any resolved trades.
    fallback = [
        m for m in metrics_list
        if not m.is_bot and m.sports_ratio <= config.max_sports_ratio and m.resolved_trades > 0
        and m.wallet not in {e.wallet for e in eligible}
    ]
    fallback.sort(key=sort_key, reverse=True)
    combined = eligible + fallback

    if len(combined) >= limit:
        return [m.wallet for m in combined[:limit]]

    # Final fallback: any non-bot wallets sorted by resolved trades then total trades.
    tail = [
        m for m in metrics_list
        if not m.is_bot and m.wallet not in {c.wallet for c in combined}
    ]
    tail.sort(key=lambda m: (m.resolved_trades, m.total_trades), reverse=True)
    combined.extend(tail)

    return [m.wallet for m in combined[:limit]]
