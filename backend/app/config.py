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
    )
