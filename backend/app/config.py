from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: str
    n_wallets: int
    trades_limit: int
    refresh_interval_sec: int
    outbound_concurrency: int
    enable_pricing: bool
    paper_enabled: bool
    paper_window_days: int
    paper_min_confidence: float
    paper_min_participants: int
    paper_min_total_participants: int
    paper_weighted_consensus_min: float
    paper_min_weighted_participants: float
    paper_entry_min: float
    paper_entry_max: float
    paper_require_tight_band: bool
    paper_ev_min: float
    paper_bet_sizing: str
    paper_base_bet: float
    paper_max_bet: float
    paper_starting_bankroll: float
    paper_bet_fraction: float

    data_api_base: str = "https://data-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    user_agent: str = "mimic-mvp/0.1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[1]
    default_db_path = str(backend_root / "mimic.sqlite3")
    return Settings(
        db_path=os.getenv("MIMIC_DB_PATH", default_db_path),
        n_wallets=_env_int("MIMIC_N_WALLETS", 500),
        trades_limit=_env_int("MIMIC_TRADES_LIMIT", 150),
        refresh_interval_sec=_env_int("MIMIC_REFRESH_INTERVAL_SEC", 120),
        outbound_concurrency=_env_int("MIMIC_OUTBOUND_CONCURRENCY", 15),
        enable_pricing=_env_bool("MIMIC_ENABLE_PRICING", True),
        paper_enabled=_env_bool("MIMIC_PAPER_ENABLED", False),
        paper_window_days=_env_int("MIMIC_PAPER_WINDOW_DAYS", 14),
        paper_min_confidence=_env_float("MIMIC_PAPER_MIN_CONFIDENCE", 0.6),
        paper_min_participants=_env_int("MIMIC_PAPER_MIN_PARTICIPANTS", 2),
        paper_min_total_participants=_env_int("MIMIC_PAPER_MIN_TOTAL_PARTICIPANTS", 2),
        paper_weighted_consensus_min=_env_float("MIMIC_PAPER_WEIGHTED_CONSENSUS_MIN", 0.0),
        paper_min_weighted_participants=_env_float("MIMIC_PAPER_MIN_WEIGHTED_PARTICIPANTS", 0.0),
        paper_entry_min=_env_float("MIMIC_PAPER_ENTRY_MIN", 0.0),
        paper_entry_max=_env_float("MIMIC_PAPER_ENTRY_MAX", 1.0),
        paper_require_tight_band=_env_bool("MIMIC_PAPER_REQUIRE_TIGHT_BAND", False),
        paper_ev_min=_env_float("MIMIC_PAPER_EV_MIN", -1.0),
        paper_bet_sizing=os.getenv("MIMIC_PAPER_BET_SIZING", "bankroll"),
        paper_base_bet=_env_float("MIMIC_PAPER_BASE_BET", 100.0),
        paper_max_bet=_env_float("MIMIC_PAPER_MAX_BET", 500.0),
        paper_starting_bankroll=_env_float("MIMIC_PAPER_STARTING_BANKROLL", 200.0),
        paper_bet_fraction=_env_float("MIMIC_PAPER_BET_FRACTION", 0.02),
    )
