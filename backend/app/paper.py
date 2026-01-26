from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time

from . import backtest, db
from .config import Settings


def _utc_ts() -> int:
    return int(time.time())


def _meta_get_int(conn: Any, key: str) -> int | None:
    raw = db.meta_get(conn, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _meta_get_float(conn: Any, key: str) -> float | None:
    raw = db.meta_get(conn, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _meta_set_float(conn: Any, key: str, value: float) -> None:
    db.meta_set(conn, key, f"{float(value)}")


@dataclass(frozen=True)
class PaperConfig:
    min_confidence: float
    min_participants: int
    min_total_participants: int
    weighted_consensus_min: float
    min_weighted_participants: float
    entry_min: float
    entry_max: float
    require_tight_band: bool
    ev_min: float
    bet_sizing: str
    base_bet: float
    max_bet: float
    starting_bankroll: float
    bet_fraction: float
    window_days: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "PaperConfig":
        return cls(
            min_confidence=float(settings.paper_min_confidence),
            min_participants=int(settings.paper_min_participants),
            min_total_participants=int(settings.paper_min_total_participants),
            weighted_consensus_min=float(settings.paper_weighted_consensus_min),
            min_weighted_participants=float(settings.paper_min_weighted_participants),
            entry_min=float(settings.paper_entry_min),
            entry_max=float(settings.paper_entry_max),
            require_tight_band=bool(settings.paper_require_tight_band),
            ev_min=float(settings.paper_ev_min),
            bet_sizing=str(settings.paper_bet_sizing or "flat"),
            base_bet=float(settings.paper_base_bet),
            max_bet=float(settings.paper_max_bet),
            starting_bankroll=float(settings.paper_starting_bankroll),
            bet_fraction=float(settings.paper_bet_fraction),
            window_days=int(settings.paper_window_days),
        )


def _calculate_bet_size(probability: float, config: PaperConfig) -> float:
    if config.bet_sizing == "flat":
        return config.base_bet
    if config.bet_sizing == "scaled":
        edge = max(0.0, (probability - 0.5) * 2)
        return config.base_bet + (config.max_bet - config.base_bet) * edge
    if config.bet_sizing == "kelly":
        kelly_fraction = probability - (1 - probability)
        kelly_bet = config.max_bet * max(0, min(0.25, kelly_fraction))
        return max(config.base_bet, kelly_bet)
    return config.base_bet


def _entry_price(market: dict[str, Any]) -> float | None:
    midpoint = market.get("midpoint")
    if midpoint is not None:
        try:
            return float(midpoint)
        except (TypeError, ValueError):
            pass
    mean_entry = market.get("mean_entry")
    if mean_entry is None:
        return None
    try:
        return float(mean_entry)
    except (TypeError, ValueError):
        return None


def _eligible_market(market: dict[str, Any], config: PaperConfig) -> bool:
    if market.get("is_closed") or not market.get("is_active", True):
        return False
    if not market.get("ready"):
        return False
    if market.get("confidence_score", 0.0) < config.min_confidence:
        return False
    participants = int(market.get("participants") or 0)
    total_participants = int(market.get("total_participants") or 0)
    if participants < config.min_participants:
        return False
    if total_participants < config.min_total_participants:
        return False
    if config.require_tight_band and not market.get("tight_band"):
        return False
    if market.get("weighted_consensus_percent", 0.0) < config.weighted_consensus_min:
        return False
    if market.get("weighted_participants", 0.0) < config.min_weighted_participants:
        return False
    entry = _entry_price(market)
    if entry is None:
        return False
    if entry < config.entry_min or entry >= config.entry_max:
        return False
    if config.ev_min > -1 and entry > 0:
        prob = float(market.get("weighted_consensus_percent") or 0.0)
        ev = prob * (1 / entry - 1) - (1 - prob)
        if ev < config.ev_min:
            return False
    return True


def _get_or_init_start_ts(conn: Any, now_ts: int) -> int:
    start_ts = _meta_get_int(conn, "paper_start_ts")
    if start_ts is None:
        db.meta_set(conn, "paper_start_ts", str(int(now_ts)))
        start_ts = now_ts
    return start_ts


def _window_allows_new_trades(conn: Any, config: PaperConfig, now_ts: int) -> bool:
    if config.window_days <= 0:
        return True
    start_ts = _get_or_init_start_ts(conn, now_ts)
    window_end = start_ts + config.window_days * 86400
    return now_ts <= window_end


def _get_bankroll(conn: Any, config: PaperConfig) -> float:
    bankroll = _meta_get_float(conn, "paper_bankroll")
    if bankroll is None:
        bankroll = config.starting_bankroll
        _meta_set_float(conn, "paper_bankroll", bankroll)
    return bankroll


def _set_bankroll(conn: Any, bankroll: float) -> None:
    _meta_set_float(conn, "paper_bankroll", bankroll)


def get_paper_meta(conn: Any, settings: Settings) -> dict[str, Any]:
    config = PaperConfig.from_settings(settings)
    start_ts = _meta_get_int(conn, "paper_start_ts")
    if start_ts is not None and config.window_days > 0:
        window_end = start_ts + config.window_days * 86400
    else:
        window_end = None
    bankroll = _meta_get_float(conn, "paper_bankroll")
    if bankroll is None:
        bankroll = config.starting_bankroll
    return {
        "start_ts": start_ts,
        "window_end_ts": window_end,
        "bankroll": bankroll,
    }


def process_paper_trades(
    conn: Any,
    settings: Settings,
    *,
    market_rows: list[dict[str, Any]],
    resolutions: dict[str, str],
    now_ts: int | None = None,
) -> dict[str, int]:
    if not settings.paper_enabled:
        return {"opened": 0, "resolved": 0}

    config = PaperConfig.from_settings(settings)
    now = now_ts if now_ts is not None else _utc_ts()

    open_trades = db.get_open_paper_trades(conn)
    open_conditions = {t["condition_id"] for t in open_trades}

    # Resolve existing paper trades first.
    resolved_count = 0
    bankroll = None
    for trade in open_trades:
        winner = resolutions.get(str(trade["condition_id"]))
        if not winner:
            continue
        predicted = str(trade["predicted_outcome"])
        won = predicted == str(winner)
        pnl = backtest.calculate_pnl(float(trade["bet_size"]), trade.get("entry_price"), won)
        status = "won" if won else "lost"
        db.resolve_paper_trade(
            conn,
            int(trade["id"]),
            status=status,
            actual_outcome=str(winner),
            pnl=pnl,
            resolved_timestamp=now,
        )
        resolved_count += 1
        if config.bet_sizing == "bankroll":
            if bankroll is None:
                bankroll = _get_bankroll(conn, config)
            bankroll += pnl

    if config.bet_sizing == "bankroll" and bankroll is not None:
        _set_bankroll(conn, bankroll)

    # Open new trades if within the window.
    opened_count = 0
    if not _window_allows_new_trades(conn, config, now):
        return {"opened": 0, "resolved": resolved_count}

    if config.bet_sizing == "bankroll":
        bankroll = _get_bankroll(conn, config)

    for market in market_rows:
        condition_id = str(market.get("condition_id") or "")
        if not condition_id or condition_id in open_conditions:
            continue
        if not _eligible_market(market, config):
            continue
        predicted = market.get("leading_outcome")
        if predicted is None:
            continue
        entry = _entry_price(market)
        if entry is None:
            continue
        prob = float(market.get("weighted_consensus_percent") or 0.0)
        if config.bet_sizing == "bankroll":
            bet_size = bankroll * config.bet_fraction
            if bet_size > bankroll:
                bet_size = bankroll
        else:
            bet_size = _calculate_bet_size(prob, config)
        if bet_size <= 0:
            continue
        db.insert_paper_trade(
            conn,
            condition_id=condition_id,
            title=market.get("title"),
            predicted_outcome=str(predicted),
            confidence_score=float(market.get("confidence_score") or 0.0),
            consensus_percent=_safe_float(market.get("consensus_percent")),
            weighted_consensus_percent=_safe_float(market.get("weighted_consensus_percent")),
            participants=_safe_int(market.get("participants")),
            total_participants=_safe_int(market.get("total_participants")),
            weighted_participants=_safe_float(market.get("weighted_participants")),
            mean_entry=_safe_float(market.get("mean_entry")),
            midpoint=_safe_float(market.get("midpoint")),
            entry_price=entry,
            bet_size=float(bet_size),
            entry_timestamp=now,
        )
        opened_count += 1

    return {"opened": opened_count, "resolved": resolved_count}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
