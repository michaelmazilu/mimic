from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any


def _utc_ts() -> int:
    return int(time.time())


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextmanager
def db_conn(db_path: str) -> Iterable[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with db_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS wallets (
              wallet TEXT PRIMARY KEY,
              rank INTEGER,
              user_name TEXT,
              x_username TEXT,
              verified_badge INTEGER,
              vol REAL,
              pnl REAL,
              profile_image TEXT,
              updated_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              wallet TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome TEXT NOT NULL,
              side TEXT NOT NULL,
              size REAL,
              price REAL,
              timestamp INTEGER,
              asset_id TEXT,
              tx_hash TEXT,
              fingerprint TEXT NOT NULL,
              title TEXT,
              slug TEXT,
              event_slug TEXT,
              raw_json TEXT NOT NULL,
              inserted_at INTEGER NOT NULL,
              UNIQUE(wallet, fingerprint),
              UNIQUE(tx_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_trades_condition_id ON trades(condition_id);
            CREATE INDEX IF NOT EXISTS idx_trades_wallet_ts ON trades(wallet, timestamp DESC);

            CREATE TABLE IF NOT EXISTS computed_market_state (
              condition_id TEXT PRIMARY KEY,
              title TEXT,
              leading_outcome TEXT,
              consensus_percent REAL,
              total_participants INTEGER,
              participants INTEGER,
              band_min REAL,
              band_max REAL,
              mean_entry REAL,
              stddev REAL,
              tight_band INTEGER,
              midpoint REAL,
              cooked INTEGER,
              price_unavailable INTEGER,
              ready INTEGER,
              updated_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_market_state_consensus ON computed_market_state(consensus_percent DESC);

            CREATE TABLE IF NOT EXISTS computed_clusters (
              cluster_id TEXT PRIMARY KEY,
              wallets_json TEXT NOT NULL,
              size INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )


def meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def upsert_wallets(conn: sqlite3.Connection, leaderboard_rows: list[dict[str, Any]]) -> int:
    now = _utc_ts()
    rows = []
    for entry in leaderboard_rows:
        wallet = entry.get("proxyWallet") or entry.get("wallet") or entry.get("address")
        if not wallet:
            continue
        rank = entry.get("rank")
        try:
            rank_int = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_int = None
        rows.append(
            (
                wallet.lower(),
                rank_int,
                entry.get("userName"),
                entry.get("xUsername"),
                1 if entry.get("verifiedBadge") else 0,
                entry.get("vol"),
                entry.get("pnl"),
                entry.get("profileImage"),
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO wallets(wallet, rank, user_name, x_username, verified_badge, vol, pnl, profile_image, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wallet) DO UPDATE SET
          rank = excluded.rank,
          user_name = excluded.user_name,
          x_username = excluded.x_username,
          verified_badge = excluded.verified_badge,
          vol = excluded.vol,
          pnl = excluded.pnl,
          profile_image = excluded.profile_image,
          updated_at = excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def insert_trades(conn: sqlite3.Connection, wallet: str, trades: list[dict[str, Any]]) -> int:
    now = _utc_ts()
    wallet_lc = wallet.lower()
    inserted = 0
    rows = []
    for t in trades:
        condition_id = t.get("conditionId") or t.get("condition_id")
        outcome = t.get("outcome")
        side = t.get("side")
        if not condition_id or not outcome or not side:
            continue

        timestamp = t.get("timestamp")
        try:
            ts_int = int(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            ts_int = None

        tx_hash = t.get("tx_hash") or t.get("transactionHash") or t.get("transaction_hash")
        if isinstance(tx_hash, str):
            tx_hash = tx_hash.strip()
            if tx_hash == "":
                tx_hash = None

        asset_id = t.get("asset") or t.get("asset_id") or t.get("token_id") or t.get("tokenId")
        if asset_id is not None:
            asset_id = str(asset_id)

        fingerprint = json.dumps(
            {
                "conditionId": condition_id,
                "outcome": outcome,
                "side": side,
                "timestamp": ts_int,
                "price": t.get("price"),
                "size": t.get("size"),
                "asset": asset_id,
                "tx": tx_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        rows.append(
            (
                wallet_lc,
                condition_id,
                str(outcome),
                str(side),
                t.get("size"),
                t.get("price"),
                ts_int,
                asset_id,
                tx_hash,
                fingerprint,
                t.get("title"),
                t.get("slug"),
                t.get("eventSlug") or t.get("event_slug"),
                json.dumps(t, separators=(",", ":"), ensure_ascii=False),
                now,
            )
        )

    if not rows:
        return 0

    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO trades(
          wallet, condition_id, outcome, side, size, price, timestamp, asset_id, tx_hash, fingerprint,
          title, slug, event_slug, raw_json, inserted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if cur.rowcount is not None:
        inserted = int(cur.rowcount)
    return inserted


def count_wallets(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(1) AS c FROM wallets").fetchone()
    return int(row["c"]) if row else 0


def count_trades(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(1) AS c FROM trades").fetchone()
    return int(row["c"]) if row else 0


def get_last_refresh_ts(conn: sqlite3.Connection) -> int | None:
    raw = meta_get(conn, "last_refresh_ts")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def set_last_refresh_ts(conn: sqlite3.Connection, ts: int) -> None:
    meta_set(conn, "last_refresh_ts", str(int(ts)))


def get_recent_trades_by_wallet(
    conn: sqlite3.Connection, *, wallet: str, limit: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, tx_hash, title, raw_json
        FROM trades
        WHERE wallet = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (wallet.lower(), limit),
    ).fetchall()


def get_all_wallets(conn: sqlite3.Connection, *, limit: int | None = None) -> list[str]:
    if limit is None:
        rows = conn.execute("SELECT wallet FROM wallets ORDER BY rank ASC NULLS LAST").fetchall()
    else:
        rows = conn.execute(
            "SELECT wallet FROM wallets ORDER BY rank ASC NULLS LAST LIMIT ?", (limit,)
        ).fetchall()
    return [str(r["wallet"]) for r in rows]


def get_all_buy_trades(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, title
        FROM trades
        WHERE side = 'BUY'
        ORDER BY timestamp DESC
        """
    ).fetchall()


def replace_market_state(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    now = _utc_ts()
    conn.execute("DELETE FROM computed_market_state")
    conn.executemany(
        """
        INSERT INTO computed_market_state(
          condition_id, title, leading_outcome, consensus_percent, total_participants, participants,
          band_min, band_max, mean_entry, stddev, tight_band,
          midpoint, cooked, price_unavailable, ready, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["condition_id"],
                r.get("title"),
                r.get("leading_outcome"),
                r.get("consensus_percent"),
                r.get("total_participants"),
                r.get("participants"),
                r.get("band_min"),
                r.get("band_max"),
                r.get("mean_entry"),
                r.get("stddev"),
                1 if r.get("tight_band") else 0,
                r.get("midpoint"),
                1 if r.get("cooked") else 0,
                1 if r.get("price_unavailable") else 0,
                1 if r.get("ready") else 0,
                now,
            )
            for r in rows
        ],
    )


def replace_clusters(conn: sqlite3.Connection, clusters: list[dict[str, Any]]) -> None:
    now = _utc_ts()
    conn.execute("DELETE FROM computed_clusters")
    conn.executemany(
        """
        INSERT INTO computed_clusters(cluster_id, wallets_json, size, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                c["cluster_id"],
                json.dumps(c.get("wallets", []), separators=(",", ":")),
                int(c.get("size", 0)),
                now,
            )
            for c in clusters
        ],
    )


def read_market_state(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          condition_id, title, leading_outcome, consensus_percent, total_participants, participants,
          band_min, band_max, mean_entry, stddev, tight_band, midpoint, cooked, price_unavailable, ready, updated_at
        FROM computed_market_state
        ORDER BY consensus_percent DESC, participants DESC
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "condition_id": r["condition_id"],
                "title": r["title"],
                "leading_outcome": r["leading_outcome"],
                "consensus_percent": r["consensus_percent"],
                "total_participants": r["total_participants"],
                "participants": r["participants"],
                "band_min": r["band_min"],
                "band_max": r["band_max"],
                "mean_entry": r["mean_entry"],
                "stddev": r["stddev"],
                "tight_band": bool(r["tight_band"]),
                "midpoint": r["midpoint"],
                "cooked": bool(r["cooked"]),
                "price_unavailable": bool(r["price_unavailable"]),
                "ready": bool(r["ready"]),
                "updated_at": r["updated_at"],
            }
        )
    return out


def read_clusters(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT cluster_id, wallets_json, size, updated_at FROM computed_clusters ORDER BY size DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            wallets = json.loads(r["wallets_json"])
        except Exception:
            wallets = []
        out.append(
            {
                "cluster_id": r["cluster_id"],
                "wallets": wallets,
                "size": int(r["size"]),
                "updated_at": int(r["updated_at"]),
            }
        )
    return out


def get_trades_for_condition(conn: sqlite3.Connection, condition_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, tx_hash, title, raw_json
        FROM trades
        WHERE condition_id = ?
        ORDER BY timestamp DESC
        """,
        (condition_id,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "wallet": r["wallet"],
                "condition_id": r["condition_id"],
                "outcome": r["outcome"],
                "side": r["side"],
                "price": r["price"],
                "size": r["size"],
                "timestamp": r["timestamp"],
                "asset_id": r["asset_id"],
                "tx_hash": r["tx_hash"],
                "title": r["title"],
                "raw_json": r["raw_json"],
            }
        )
    return out

