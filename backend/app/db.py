from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any


def _utc_ts() -> int:
    return int(time.time())


def connect(db_path: str, *, timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")  # 30 second timeout for locks
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
              weighted_consensus_percent REAL,
              total_participants INTEGER,
              participants INTEGER,
              weighted_participants REAL,
              band_min REAL,
              band_max REAL,
              mean_entry REAL,
              stddev REAL,
              tight_band INTEGER,
              midpoint REAL,
              cooked INTEGER,
              price_unavailable INTEGER,
              ready INTEGER,
              confidence_score REAL,
              end_date TEXT,
              is_closed INTEGER DEFAULT 0,
              is_active INTEGER DEFAULT 1,
              updated_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_market_state_consensus ON computed_market_state(consensus_percent DESC);
            CREATE INDEX IF NOT EXISTS idx_market_state_weighted ON computed_market_state(weighted_consensus_percent DESC);
            CREATE INDEX IF NOT EXISTS idx_market_state_confidence ON computed_market_state(confidence_score DESC);
            CREATE INDEX IF NOT EXISTS idx_market_state_active ON computed_market_state(is_active, is_closed);

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

            -- Track resolved market outcomes per wallet (win/loss/pending)
            CREATE TABLE IF NOT EXISTS wallet_outcomes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              wallet TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              outcome TEXT NOT NULL,
              side TEXT NOT NULL,
              entry_price REAL,
              exit_price REAL,
              size REAL,
              pnl REAL,
              status TEXT NOT NULL DEFAULT 'pending',  -- pending, won, lost
              entry_timestamp INTEGER,
              resolved_timestamp INTEGER,
              inserted_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              UNIQUE(wallet, condition_id, outcome, entry_timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_wallet_outcomes_wallet ON wallet_outcomes(wallet);
            CREATE INDEX IF NOT EXISTS idx_wallet_outcomes_status ON wallet_outcomes(status);
            CREATE INDEX IF NOT EXISTS idx_wallet_outcomes_resolved ON wallet_outcomes(resolved_timestamp DESC);

            -- Aggregated wallet performance stats
            CREATE TABLE IF NOT EXISTS wallet_stats (
              wallet TEXT PRIMARY KEY,
              total_trades INTEGER DEFAULT 0,
              won_trades INTEGER DEFAULT 0,
              lost_trades INTEGER DEFAULT 0,
              pending_trades INTEGER DEFAULT 0,
              win_rate REAL DEFAULT 0.0,
              total_pnl REAL DEFAULT 0.0,
              avg_roi REAL DEFAULT 0.0,
              recent_trades_7d INTEGER DEFAULT 0,
              recent_won_7d INTEGER DEFAULT 0,
              recent_accuracy_7d REAL DEFAULT 0.0,
              recent_trades_30d INTEGER DEFAULT 0,
              recent_won_30d INTEGER DEFAULT 0,
              recent_accuracy_30d REAL DEFAULT 0.0,
              streak INTEGER DEFAULT 0,  -- positive = win streak, negative = loss streak
              last_trade_timestamp INTEGER,
              updated_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_wallet_stats_win_rate ON wallet_stats(win_rate DESC);
            CREATE INDEX IF NOT EXISTS idx_wallet_stats_recent_accuracy ON wallet_stats(recent_accuracy_7d DESC);

            -- Store resolved market outcomes from CLOB API
            CREATE TABLE IF NOT EXISTS market_resolutions (
              condition_id TEXT PRIMARY KEY,
              winning_outcome TEXT NOT NULL,
              resolved_at INTEGER NOT NULL,
              fetched_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_market_resolutions_resolved ON market_resolutions(resolved_at DESC);

            -- Store backtest runs and results
            CREATE TABLE IF NOT EXISTS backtest_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT UNIQUE NOT NULL,
              config_json TEXT NOT NULL,
              started_at INTEGER NOT NULL,
              completed_at INTEGER,
              status TEXT DEFAULT 'running',
              total_trades INTEGER DEFAULT 0,
              winning_trades INTEGER DEFAULT 0,
              losing_trades INTEGER DEFAULT 0,
              win_rate REAL DEFAULT 0.0,
              total_pnl REAL DEFAULT 0.0,
              total_invested REAL DEFAULT 0.0,
              roi REAL DEFAULT 0.0,
              max_drawdown REAL DEFAULT 0.0,
              sharpe_ratio REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_backtest_runs_started ON backtest_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS backtest_trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              condition_id TEXT NOT NULL,
              title TEXT,
              signal_timestamp INTEGER NOT NULL,
              predicted_outcome TEXT NOT NULL,
              confidence_score REAL NOT NULL,
              bet_size REAL NOT NULL,
              entry_price REAL,
              actual_outcome TEXT,
              pnl REAL,
              won INTEGER,
              FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id);
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_timestamp ON backtest_trades(signal_timestamp DESC);
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
        SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, title, slug, event_slug
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
          condition_id, title, leading_outcome, consensus_percent, weighted_consensus_percent,
          total_participants, participants, weighted_participants,
          band_min, band_max, mean_entry, stddev, tight_band,
          midpoint, cooked, price_unavailable, ready, confidence_score,
          end_date, is_closed, is_active, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["condition_id"],
                r.get("title"),
                r.get("leading_outcome"),
                r.get("consensus_percent"),
                r.get("weighted_consensus_percent", 0.0),
                r.get("total_participants"),
                r.get("participants"),
                r.get("weighted_participants", 0.0),
                r.get("band_min"),
                r.get("band_max"),
                r.get("mean_entry"),
                r.get("stddev"),
                1 if r.get("tight_band") else 0,
                r.get("midpoint"),
                1 if r.get("cooked") else 0,
                1 if r.get("price_unavailable") else 0,
                1 if r.get("ready") else 0,
                r.get("confidence_score", 0.0),
                r.get("end_date"),
                1 if r.get("is_closed") else 0,
                1 if r.get("is_active", True) else 0,
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


def read_market_state(conn: sqlite3.Connection, *, active_only: bool = True) -> list[dict[str, Any]]:
    """Read market state, optionally filtering to only active (non-closed) markets."""
    if active_only:
        rows = conn.execute(
            """
            SELECT
              condition_id, title, leading_outcome, consensus_percent, weighted_consensus_percent,
              total_participants, participants, weighted_participants,
              band_min, band_max, mean_entry, stddev, tight_band, midpoint, cooked, 
              price_unavailable, ready, confidence_score, end_date, is_closed, is_active, updated_at
            FROM computed_market_state
            WHERE is_closed = 0 AND is_active = 1
            ORDER BY confidence_score DESC, weighted_consensus_percent DESC, participants DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
              condition_id, title, leading_outcome, consensus_percent, weighted_consensus_percent,
              total_participants, participants, weighted_participants,
              band_min, band_max, mean_entry, stddev, tight_band, midpoint, cooked, 
              price_unavailable, ready, confidence_score, end_date, is_closed, is_active, updated_at
            FROM computed_market_state
            ORDER BY confidence_score DESC, weighted_consensus_percent DESC, participants DESC
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
                "weighted_consensus_percent": r["weighted_consensus_percent"] or 0.0,
                "total_participants": r["total_participants"],
                "participants": r["participants"],
                "weighted_participants": r["weighted_participants"] or 0.0,
                "band_min": r["band_min"],
                "band_max": r["band_max"],
                "mean_entry": r["mean_entry"],
                "stddev": r["stddev"],
                "tight_band": bool(r["tight_band"]),
                "midpoint": r["midpoint"],
                "cooked": bool(r["cooked"]),
                "price_unavailable": bool(r["price_unavailable"]),
                "ready": bool(r["ready"]),
                "confidence_score": r["confidence_score"] or 0.0,
                "end_date": r["end_date"],
                "is_closed": bool(r["is_closed"]),
                "is_active": bool(r["is_active"]),
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


# ============================================================================
# Wallet Outcomes & Stats Functions
# ============================================================================


def upsert_wallet_outcome(
    conn: sqlite3.Connection,
    wallet: str,
    condition_id: str,
    outcome: str,
    side: str,
    entry_price: float | None,
    size: float | None,
    entry_timestamp: int | None,
    status: str = "pending",
    exit_price: float | None = None,
    pnl: float | None = None,
    resolved_timestamp: int | None = None,
) -> None:
    """Insert or update a wallet outcome record."""
    now = _utc_ts()
    conn.execute(
        """
        INSERT INTO wallet_outcomes(
            wallet, condition_id, outcome, side, entry_price, exit_price, size, pnl,
            status, entry_timestamp, resolved_timestamp, inserted_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wallet, condition_id, outcome, entry_timestamp) DO UPDATE SET
            exit_price = COALESCE(excluded.exit_price, wallet_outcomes.exit_price),
            pnl = COALESCE(excluded.pnl, wallet_outcomes.pnl),
            status = excluded.status,
            resolved_timestamp = COALESCE(excluded.resolved_timestamp, wallet_outcomes.resolved_timestamp),
            updated_at = excluded.updated_at
        """,
        (
            wallet.lower(),
            condition_id,
            outcome,
            side,
            entry_price,
            exit_price,
            size,
            pnl,
            status,
            entry_timestamp,
            resolved_timestamp,
            now,
            now,
        ),
    )


def resolve_wallet_outcome(
    conn: sqlite3.Connection,
    wallet: str,
    condition_id: str,
    winning_outcome: str,
    resolved_timestamp: int | None = None,
) -> int:
    """Mark outcomes for a market as won/lost based on the winning outcome."""
    now = _utc_ts()
    resolved_ts = resolved_timestamp or now
    
    # Mark winning outcomes
    conn.execute(
        """
        UPDATE wallet_outcomes
        SET status = 'won', resolved_timestamp = ?, updated_at = ?
        WHERE wallet = ? AND condition_id = ? AND outcome = ? AND status = 'pending'
        """,
        (resolved_ts, now, wallet.lower(), condition_id, winning_outcome),
    )
    
    # Mark losing outcomes
    cur = conn.execute(
        """
        UPDATE wallet_outcomes
        SET status = 'lost', resolved_timestamp = ?, updated_at = ?
        WHERE wallet = ? AND condition_id = ? AND outcome != ? AND status = 'pending'
        """,
        (resolved_ts, now, wallet.lower(), condition_id, winning_outcome),
    )
    
    return cur.rowcount or 0


def get_wallet_outcomes(
    conn: sqlite3.Connection,
    wallet: str,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get outcome records for a wallet."""
    if status:
        rows = conn.execute(
            """
            SELECT wallet, condition_id, outcome, side, entry_price, exit_price, size, pnl,
                   status, entry_timestamp, resolved_timestamp, inserted_at, updated_at
            FROM wallet_outcomes
            WHERE wallet = ? AND status = ?
            ORDER BY entry_timestamp DESC
            LIMIT ?
            """,
            (wallet.lower(), status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT wallet, condition_id, outcome, side, entry_price, exit_price, size, pnl,
                   status, entry_timestamp, resolved_timestamp, inserted_at, updated_at
            FROM wallet_outcomes
            WHERE wallet = ?
            ORDER BY entry_timestamp DESC
            LIMIT ?
            """,
            (wallet.lower(), limit),
        ).fetchall()
    
    return [
        {
            "wallet": r["wallet"],
            "condition_id": r["condition_id"],
            "outcome": r["outcome"],
            "side": r["side"],
            "entry_price": r["entry_price"],
            "exit_price": r["exit_price"],
            "size": r["size"],
            "pnl": r["pnl"],
            "status": r["status"],
            "entry_timestamp": r["entry_timestamp"],
            "resolved_timestamp": r["resolved_timestamp"],
            "inserted_at": r["inserted_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def compute_and_update_wallet_stats(conn: sqlite3.Connection, wallet: str) -> dict[str, Any]:
    """Compute and store aggregated stats for a wallet."""
    now = _utc_ts()
    wallet_lc = wallet.lower()
    
    # Time boundaries for recent stats
    seven_days_ago = now - (7 * 24 * 60 * 60)
    thirty_days_ago = now - (30 * 24 * 60 * 60)
    
    # Total stats
    total_row = conn.execute(
        """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as won,
            SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as lost,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
            SUM(COALESCE(pnl, 0)) as total_pnl,
            MAX(entry_timestamp) as last_trade
        FROM wallet_outcomes
        WHERE wallet = ?
        """,
        (wallet_lc,),
    ).fetchone()
    
    total_trades = total_row["total"] or 0
    won_trades = total_row["won"] or 0
    lost_trades = total_row["lost"] or 0
    pending_trades = total_row["pending"] or 0
    total_pnl = total_row["total_pnl"] or 0.0
    last_trade_ts = total_row["last_trade"]
    
    resolved_trades = won_trades + lost_trades
    win_rate = (won_trades / resolved_trades) if resolved_trades > 0 else 0.0
    avg_roi = (total_pnl / total_trades) if total_trades > 0 else 0.0
    
    # 7-day stats
    recent_7d_row = conn.execute(
        """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as won
        FROM wallet_outcomes
        WHERE wallet = ? AND entry_timestamp >= ? AND status IN ('won', 'lost')
        """,
        (wallet_lc, seven_days_ago),
    ).fetchone()
    
    recent_trades_7d = recent_7d_row["total"] or 0
    recent_won_7d = recent_7d_row["won"] or 0
    recent_accuracy_7d = (recent_won_7d / recent_trades_7d) if recent_trades_7d > 0 else 0.0
    
    # 30-day stats
    recent_30d_row = conn.execute(
        """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as won
        FROM wallet_outcomes
        WHERE wallet = ? AND entry_timestamp >= ? AND status IN ('won', 'lost')
        """,
        (wallet_lc, thirty_days_ago),
    ).fetchone()
    
    recent_trades_30d = recent_30d_row["total"] or 0
    recent_won_30d = recent_30d_row["won"] or 0
    recent_accuracy_30d = (recent_won_30d / recent_trades_30d) if recent_trades_30d > 0 else 0.0
    
    # Calculate streak (consecutive wins or losses)
    streak_rows = conn.execute(
        """
        SELECT status FROM wallet_outcomes
        WHERE wallet = ? AND status IN ('won', 'lost')
        ORDER BY resolved_timestamp DESC
        LIMIT 20
        """,
        (wallet_lc,),
    ).fetchall()
    
    streak = 0
    if streak_rows:
        first_status = streak_rows[0]["status"]
        for r in streak_rows:
            if r["status"] == first_status:
                streak += 1 if first_status == "won" else -1
            else:
                break
    
    # Upsert stats
    conn.execute(
        """
        INSERT INTO wallet_stats(
            wallet, total_trades, won_trades, lost_trades, pending_trades,
            win_rate, total_pnl, avg_roi,
            recent_trades_7d, recent_won_7d, recent_accuracy_7d,
            recent_trades_30d, recent_won_30d, recent_accuracy_30d,
            streak, last_trade_timestamp, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wallet) DO UPDATE SET
            total_trades = excluded.total_trades,
            won_trades = excluded.won_trades,
            lost_trades = excluded.lost_trades,
            pending_trades = excluded.pending_trades,
            win_rate = excluded.win_rate,
            total_pnl = excluded.total_pnl,
            avg_roi = excluded.avg_roi,
            recent_trades_7d = excluded.recent_trades_7d,
            recent_won_7d = excluded.recent_won_7d,
            recent_accuracy_7d = excluded.recent_accuracy_7d,
            recent_trades_30d = excluded.recent_trades_30d,
            recent_won_30d = excluded.recent_won_30d,
            recent_accuracy_30d = excluded.recent_accuracy_30d,
            streak = excluded.streak,
            last_trade_timestamp = excluded.last_trade_timestamp,
            updated_at = excluded.updated_at
        """,
        (
            wallet_lc,
            total_trades,
            won_trades,
            lost_trades,
            pending_trades,
            win_rate,
            total_pnl,
            avg_roi,
            recent_trades_7d,
            recent_won_7d,
            recent_accuracy_7d,
            recent_trades_30d,
            recent_won_30d,
            recent_accuracy_30d,
            streak,
            last_trade_ts,
            now,
        ),
    )
    
    return {
        "wallet": wallet_lc,
        "total_trades": total_trades,
        "won_trades": won_trades,
        "lost_trades": lost_trades,
        "pending_trades": pending_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_roi": avg_roi,
        "recent_trades_7d": recent_trades_7d,
        "recent_won_7d": recent_won_7d,
        "recent_accuracy_7d": recent_accuracy_7d,
        "recent_trades_30d": recent_trades_30d,
        "recent_won_30d": recent_won_30d,
        "recent_accuracy_30d": recent_accuracy_30d,
        "streak": streak,
        "last_trade_timestamp": last_trade_ts,
        "updated_at": now,
    }


def get_wallet_stats(conn: sqlite3.Connection, wallet: str) -> dict[str, Any] | None:
    """Get stats for a single wallet."""
    row = conn.execute(
        """
        SELECT wallet, total_trades, won_trades, lost_trades, pending_trades,
               win_rate, total_pnl, avg_roi,
               recent_trades_7d, recent_won_7d, recent_accuracy_7d,
               recent_trades_30d, recent_won_30d, recent_accuracy_30d,
               streak, last_trade_timestamp, updated_at
        FROM wallet_stats
        WHERE wallet = ?
        """,
        (wallet.lower(),),
    ).fetchone()
    
    if not row:
        return None
    
    return {
        "wallet": row["wallet"],
        "total_trades": row["total_trades"],
        "won_trades": row["won_trades"],
        "lost_trades": row["lost_trades"],
        "pending_trades": row["pending_trades"],
        "win_rate": row["win_rate"],
        "total_pnl": row["total_pnl"],
        "avg_roi": row["avg_roi"],
        "recent_trades_7d": row["recent_trades_7d"],
        "recent_won_7d": row["recent_won_7d"],
        "recent_accuracy_7d": row["recent_accuracy_7d"],
        "recent_trades_30d": row["recent_trades_30d"],
        "recent_won_30d": row["recent_won_30d"],
        "recent_accuracy_30d": row["recent_accuracy_30d"],
        "streak": row["streak"],
        "last_trade_timestamp": row["last_trade_timestamp"],
        "updated_at": row["updated_at"],
    }


def get_all_wallet_stats(
    conn: sqlite3.Connection,
    *,
    order_by: str = "recent_accuracy_7d",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get stats for all wallets, ordered by specified field."""
    valid_order_fields = {
        "win_rate", "recent_accuracy_7d", "recent_accuracy_30d",
        "total_pnl", "total_trades", "streak"
    }
    if order_by not in valid_order_fields:
        order_by = "recent_accuracy_7d"
    
    rows = conn.execute(
        f"""
        SELECT w.wallet, w.rank, w.user_name, w.pnl as leaderboard_pnl,
               s.total_trades, s.won_trades, s.lost_trades, s.pending_trades,
               s.win_rate, s.total_pnl, s.avg_roi,
               s.recent_trades_7d, s.recent_won_7d, s.recent_accuracy_7d,
               s.recent_trades_30d, s.recent_won_30d, s.recent_accuracy_30d,
               s.streak, s.last_trade_timestamp, s.updated_at
        FROM wallets w
        LEFT JOIN wallet_stats s ON w.wallet = s.wallet
        ORDER BY COALESCE(s.{order_by}, 0) DESC, w.rank ASC NULLS LAST
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    
    return [
        {
            "wallet": r["wallet"],
            "rank": r["rank"],
            "user_name": r["user_name"],
            "leaderboard_pnl": r["leaderboard_pnl"],
            "total_trades": r["total_trades"] or 0,
            "won_trades": r["won_trades"] or 0,
            "lost_trades": r["lost_trades"] or 0,
            "pending_trades": r["pending_trades"] or 0,
            "win_rate": r["win_rate"] or 0.0,
            "total_pnl": r["total_pnl"] or 0.0,
            "avg_roi": r["avg_roi"] or 0.0,
            "recent_trades_7d": r["recent_trades_7d"] or 0,
            "recent_won_7d": r["recent_won_7d"] or 0,
            "recent_accuracy_7d": r["recent_accuracy_7d"] or 0.0,
            "recent_trades_30d": r["recent_trades_30d"] or 0,
            "recent_won_30d": r["recent_won_30d"] or 0,
            "recent_accuracy_30d": r["recent_accuracy_30d"] or 0.0,
            "streak": r["streak"] or 0,
            "last_trade_timestamp": r["last_trade_timestamp"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def bulk_upsert_wallet_stats(conn: sqlite3.Connection, stats: list[dict[str, Any]]) -> int:
    """Bulk insert/update wallet_stats from precomputed metrics."""
    if not stats:
        return 0
    now = _utc_ts()
    rows = [
        (
            s["wallet"],
            s.get("total_trades", 0),
            s.get("won_trades", 0),
            s.get("lost_trades", 0),
            s.get("pending_trades", 0),
            s.get("win_rate", 0.0),
            s.get("total_pnl", 0.0),
            s.get("avg_roi", 0.0),
            s.get("recent_trades_7d", 0),
            s.get("recent_won_7d", 0),
            s.get("recent_accuracy_7d", 0.0),
            s.get("recent_trades_30d", 0),
            s.get("recent_won_30d", 0),
            s.get("recent_accuracy_30d", 0.0),
            s.get("streak", 0),
            s.get("last_trade_timestamp"),
            now,
        )
        for s in stats
    ]
    conn.executemany(
        """
        INSERT INTO wallet_stats(
          wallet, total_trades, won_trades, lost_trades, pending_trades,
          win_rate, total_pnl, avg_roi,
          recent_trades_7d, recent_won_7d, recent_accuracy_7d,
          recent_trades_30d, recent_won_30d, recent_accuracy_30d,
          streak, last_trade_timestamp, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(wallet) DO UPDATE SET
          total_trades = excluded.total_trades,
          won_trades = excluded.won_trades,
          lost_trades = excluded.lost_trades,
          pending_trades = excluded.pending_trades,
          win_rate = excluded.win_rate,
          total_pnl = excluded.total_pnl,
          avg_roi = excluded.avg_roi,
          recent_trades_7d = excluded.recent_trades_7d,
          recent_won_7d = excluded.recent_won_7d,
          recent_accuracy_7d = excluded.recent_accuracy_7d,
          recent_trades_30d = excluded.recent_trades_30d,
          recent_won_30d = excluded.recent_won_30d,
          recent_accuracy_30d = excluded.recent_accuracy_30d,
          streak = excluded.streak,
          last_trade_timestamp = excluded.last_trade_timestamp,
          updated_at = excluded.updated_at
        """,
        rows,
    )
    return len(rows)


def get_wallet_accuracy_map(conn: sqlite3.Connection) -> dict[str, float]:
    """Get a mapping of wallet -> reliability-adjusted accuracy for weighting consensus."""
    def _wilson_lower_bound(wins: int, total: int, *, z: float = 1.96) -> float:
        if total <= 0:
            return 0.0
        phat = wins / total
        z2 = z * z
        denom = 1 + z2 / total
        center = phat + z2 / (2 * total)
        margin = z * ((phat * (1 - phat) + z2 / (4 * total)) / total) ** 0.5
        return max(0.0, (center - margin) / denom)

    def _stability_penalty(
        win_rate: float,
        recent_accuracy: float,
        recent_trades: int,
        *,
        min_trades: int,
    ) -> float:
        if recent_trades < min_trades or win_rate <= 0:
            return 1.0
        ratio = recent_accuracy / win_rate
        ratio = min(1.0, max(0.0, ratio))
        return 0.5 + 0.5 * ratio

    rows = conn.execute(
        """
        SELECT wallet,
               won_trades,
               lost_trades,
               recent_accuracy_7d,
               recent_trades_7d,
               recent_accuracy_30d,
               recent_trades_30d
        FROM wallet_stats
        WHERE (won_trades + lost_trades) >= 10
        """
    ).fetchall()

    weights: dict[str, float] = {}
    for r in rows:
        wins = int(r["won_trades"] or 0)
        losses = int(r["lost_trades"] or 0)
        total = wins + losses
        if total <= 0:
            continue
        win_rate = wins / total
        base = _wilson_lower_bound(wins, total)

        penalties: list[float] = []
        p7 = _stability_penalty(
            win_rate,
            float(r["recent_accuracy_7d"] or 0.0),
            int(r["recent_trades_7d"] or 0),
            min_trades=5,
        )
        if int(r["recent_trades_7d"] or 0) >= 5:
            penalties.append(p7)
        p30 = _stability_penalty(
            win_rate,
            float(r["recent_accuracy_30d"] or 0.0),
            int(r["recent_trades_30d"] or 0),
            min_trades=10,
        )
        if int(r["recent_trades_30d"] or 0) >= 10:
            penalties.append(p30)

        penalty = min(penalties) if penalties else 1.0
        weights[r["wallet"]] = base * penalty

    return weights


def get_wallet_performance_map(conn: sqlite3.Connection) -> dict[str, dict[str, float | int]]:
    """Get a mapping of wallet -> performance stats used for eligibility checks."""
    rows = conn.execute(
        """
        SELECT wallet, win_rate, won_trades, lost_trades
        FROM wallet_stats
        """
    ).fetchall()
    return {
        r["wallet"]: {
            "win_rate": float(r["win_rate"] or 0.0),
            "resolved_trades": int((r["won_trades"] or 0) + (r["lost_trades"] or 0)),
        }
        for r in rows
    }


# ============================================================================
# Market Resolutions Functions
# ============================================================================


def upsert_market_resolution(
    conn: sqlite3.Connection,
    condition_id: str,
    winning_outcome: str,
    resolved_at: int,
) -> None:
    """Insert or update a market resolution."""
    now = _utc_ts()
    conn.execute(
        """
        INSERT INTO market_resolutions(condition_id, winning_outcome, resolved_at, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(condition_id) DO UPDATE SET
            winning_outcome = excluded.winning_outcome,
            resolved_at = excluded.resolved_at,
            fetched_at = excluded.fetched_at
        """,
        (condition_id, winning_outcome, resolved_at, now),
    )


def get_market_resolution(conn: sqlite3.Connection, condition_id: str) -> dict[str, Any] | None:
    """Get resolution for a specific market."""
    row = conn.execute(
        "SELECT condition_id, winning_outcome, resolved_at, fetched_at FROM market_resolutions WHERE condition_id = ?",
        (condition_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "condition_id": row["condition_id"],
        "winning_outcome": row["winning_outcome"],
        "resolved_at": row["resolved_at"],
        "fetched_at": row["fetched_at"],
    }


def get_all_resolutions(conn: sqlite3.Connection) -> dict[str, str]:
    """Get all market resolutions as condition_id -> winning_outcome mapping."""
    rows = conn.execute("SELECT condition_id, winning_outcome FROM market_resolutions").fetchall()
    return {r["condition_id"]: r["winning_outcome"] for r in rows}


def bulk_upsert_resolutions(conn: sqlite3.Connection, resolutions: list[dict[str, Any]]) -> int:
    """Bulk insert/update market resolutions."""
    now = _utc_ts()
    rows = [
        (
            r["condition_id"],
            r["winning_outcome"],
            r.get("resolved_at", now),
            now,
        )
        for r in resolutions
    ]
    conn.executemany(
        """
        INSERT INTO market_resolutions(condition_id, winning_outcome, resolved_at, fetched_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(condition_id) DO UPDATE SET
            winning_outcome = excluded.winning_outcome,
            resolved_at = excluded.resolved_at,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )
    return len(rows)


# ============================================================================
# Backtest Functions
# ============================================================================


def create_backtest_run(
    conn: sqlite3.Connection,
    run_id: str,
    config: dict[str, Any],
) -> None:
    """Create a new backtest run record."""
    now = _utc_ts()
    conn.execute(
        """
        INSERT INTO backtest_runs(run_id, config_json, started_at, status)
        VALUES (?, ?, ?, 'running')
        """,
        (run_id, json.dumps(config), now),
    )


def update_backtest_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str | None = None,
    total_trades: int | None = None,
    winning_trades: int | None = None,
    losing_trades: int | None = None,
    win_rate: float | None = None,
    total_pnl: float | None = None,
    total_invested: float | None = None,
    roi: float | None = None,
    max_drawdown: float | None = None,
    sharpe_ratio: float | None = None,
) -> None:
    """Update backtest run with results."""
    now = _utc_ts()
    updates = ["completed_at = ?"]
    values: list[Any] = [now]
    
    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if total_trades is not None:
        updates.append("total_trades = ?")
        values.append(total_trades)
    if winning_trades is not None:
        updates.append("winning_trades = ?")
        values.append(winning_trades)
    if losing_trades is not None:
        updates.append("losing_trades = ?")
        values.append(losing_trades)
    if win_rate is not None:
        updates.append("win_rate = ?")
        values.append(win_rate)
    if total_pnl is not None:
        updates.append("total_pnl = ?")
        values.append(total_pnl)
    if total_invested is not None:
        updates.append("total_invested = ?")
        values.append(total_invested)
    if roi is not None:
        updates.append("roi = ?")
        values.append(roi)
    if max_drawdown is not None:
        updates.append("max_drawdown = ?")
        values.append(max_drawdown)
    if sharpe_ratio is not None:
        updates.append("sharpe_ratio = ?")
        values.append(sharpe_ratio)
    
    values.append(run_id)
    conn.execute(
        f"UPDATE backtest_runs SET {', '.join(updates)} WHERE run_id = ?",
        values,
    )


def insert_backtest_trade(
    conn: sqlite3.Connection,
    run_id: str,
    condition_id: str,
    title: str | None,
    signal_timestamp: int,
    predicted_outcome: str,
    confidence_score: float,
    bet_size: float,
    entry_price: float | None,
    actual_outcome: str | None,
    pnl: float | None,
    won: bool | None,
) -> None:
    """Insert a backtest trade record."""
    conn.execute(
        """
        INSERT INTO backtest_trades(
            run_id, condition_id, title, signal_timestamp, predicted_outcome,
            confidence_score, bet_size, entry_price, actual_outcome, pnl, won
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            condition_id,
            title,
            signal_timestamp,
            predicted_outcome,
            confidence_score,
            bet_size,
            entry_price,
            actual_outcome,
            pnl,
            1 if won else 0 if won is not None else None,
        ),
    )


def bulk_insert_backtest_trades(conn: sqlite3.Connection, trades: list[dict[str, Any]]) -> int:
    """Bulk insert backtest trades."""
    rows = [
        (
            t["run_id"],
            t["condition_id"],
            t.get("title"),
            t["signal_timestamp"],
            t["predicted_outcome"],
            t["confidence_score"],
            t["bet_size"],
            t.get("entry_price"),
            t.get("actual_outcome"),
            t.get("pnl"),
            1 if t.get("won") else 0 if t.get("won") is not None else None,
        )
        for t in trades
    ]
    conn.executemany(
        """
        INSERT INTO backtest_trades(
            run_id, condition_id, title, signal_timestamp, predicted_outcome,
            confidence_score, bet_size, entry_price, actual_outcome, pnl, won
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def get_backtest_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Get a backtest run by ID."""
    row = conn.execute(
        """
        SELECT run_id, config_json, started_at, completed_at, status,
               total_trades, winning_trades, losing_trades, win_rate,
               total_pnl, total_invested, roi, max_drawdown, sharpe_ratio
        FROM backtest_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "run_id": row["run_id"],
        "config": json.loads(row["config_json"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "status": row["status"],
        "total_trades": row["total_trades"],
        "winning_trades": row["winning_trades"],
        "losing_trades": row["losing_trades"],
        "win_rate": row["win_rate"],
        "total_pnl": row["total_pnl"],
        "total_invested": row["total_invested"],
        "roi": row["roi"],
        "max_drawdown": row["max_drawdown"],
        "sharpe_ratio": row["sharpe_ratio"],
    }


def get_latest_backtest_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Get the most recent completed backtest run."""
    row = conn.execute(
        """
        SELECT run_id, config_json, started_at, completed_at, status,
               total_trades, winning_trades, losing_trades, win_rate,
               total_pnl, total_invested, roi, max_drawdown, sharpe_ratio
        FROM backtest_runs
        WHERE status = 'completed'
        ORDER BY completed_at DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return {
        "run_id": row["run_id"],
        "config": json.loads(row["config_json"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "status": row["status"],
        "total_trades": row["total_trades"],
        "winning_trades": row["winning_trades"],
        "losing_trades": row["losing_trades"],
        "win_rate": row["win_rate"],
        "total_pnl": row["total_pnl"],
        "total_invested": row["total_invested"],
        "roi": row["roi"],
        "max_drawdown": row["max_drawdown"],
        "sharpe_ratio": row["sharpe_ratio"],
    }


def get_backtest_trades(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Get trades for a backtest run."""
    rows = conn.execute(
        """
        SELECT run_id, condition_id, title, signal_timestamp, predicted_outcome,
               confidence_score, bet_size, entry_price, actual_outcome, pnl, won
        FROM backtest_trades
        WHERE run_id = ?
        ORDER BY signal_timestamp ASC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [
        {
            "run_id": r["run_id"],
            "condition_id": r["condition_id"],
            "title": r["title"],
            "signal_timestamp": r["signal_timestamp"],
            "predicted_outcome": r["predicted_outcome"],
            "confidence_score": r["confidence_score"],
            "bet_size": r["bet_size"],
            "entry_price": r["entry_price"],
            "actual_outcome": r["actual_outcome"],
            "pnl": r["pnl"],
            "won": bool(r["won"]) if r["won"] is not None else None,
        }
        for r in rows
    ]


def get_trades_in_timerange(
    conn: sqlite3.Connection,
    *,
    start_ts: int,
    end_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Get all trades within a time range for backtesting."""
    if end_ts:
        rows = conn.execute(
            """
            SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, title, slug, event_slug
            FROM trades
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (start_ts, end_ts),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT wallet, condition_id, outcome, side, price, size, timestamp, asset_id, title, slug, event_slug
            FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (start_ts,),
        ).fetchall()
    
    return [
        {
            "wallet": r["wallet"],
            "condition_id": r["condition_id"],
            "outcome": r["outcome"],
            "side": r["side"],
            "price": r["price"],
            "size": r["size"],
            "timestamp": r["timestamp"],
            "asset_id": r["asset_id"],
            "title": r["title"],
            "slug": r["slug"],
            "event_slug": r["event_slug"],
        }
        for r in rows
    ]


def get_unique_condition_ids_in_trades(conn: sqlite3.Connection, start_ts: int) -> list[str]:
    """Get unique condition IDs from trades after a timestamp."""
    rows = conn.execute(
        """
        SELECT DISTINCT condition_id
        FROM trades
        WHERE timestamp >= ? AND side = 'BUY'
        """,
        (start_ts,),
    ).fetchall()
    return [r["condition_id"] for r in rows]
