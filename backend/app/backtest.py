"""
Backtest Engine for Mimic Trading Strategy

This module replays historical trades against resolved markets to measure
strategy accuracy using the same signal criteria as the live system.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from . import db
from .config import Settings
# Resolution fetching is now done inline per-market


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    min_confidence: float = 0.80
    bet_sizing: str = "scaled"  # flat, kelly, scaled
    base_bet: float = 100.0
    max_bet: float = 500.0
    lookback_days: int = 180
    min_participants: int = 2  # Minimum traders for a valid signal

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestTrade:
    """A single simulated trade in the backtest."""
    condition_id: str
    title: str | None
    signal_timestamp: int
    predicted_outcome: str
    confidence_score: float
    bet_size: float
    entry_price: float | None
    actual_outcome: str | None = None
    pnl: float | None = None
    won: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    run_id: str
    config: BacktestConfig
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    pending_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_invested: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "pending_trades": self.pending_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_invested": self.total_invested,
            "roi": self.roi,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
        }


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def calculate_bet_size(confidence: float, config: BacktestConfig) -> float:
    """
    Calculate bet size based on confidence and config.
    
    For scaled sizing:
    - Scale from base_bet at 50% confidence to max_bet at 100%
    - At 80% confidence: $100 + ($500 - $100) * 0.6 = $340
    """
    if config.bet_sizing == "flat":
        return config.base_bet
    elif config.bet_sizing == "scaled":
        # Scale from 0 at 50% to 1 at 100%
        edge = max(0, (confidence - 0.5) * 2)
        return config.base_bet + (config.max_bet - config.base_bet) * edge
    elif config.bet_sizing == "kelly":
        # Kelly criterion: f* = (bp - q) / b
        # Where b = odds, p = win probability, q = 1 - p
        # Simplified: use confidence as win probability estimate
        # Cap at 25% of max bet for safety
        kelly_fraction = confidence - (1 - confidence)  # Expected edge
        kelly_bet = config.max_bet * max(0, min(0.25, kelly_fraction))
        return max(config.base_bet, kelly_bet)
    else:
        return config.base_bet


def _compute_band(prices: list[float]) -> tuple[float, float, float, float]:
    """Compute price band statistics."""
    arr = np.asarray(prices, dtype=np.float64)
    band_min = float(np.min(arr))
    band_max = float(np.max(arr))
    mean = float(np.mean(arr))
    stddev = float(np.std(arr, ddof=0))
    return band_min, band_max, mean, stddev


def compute_signal_at_time(
    trades: list[dict[str, Any]],
    cutoff_timestamp: int,
    config: BacktestConfig,
) -> dict[str, Any] | None:
    """
    Compute signal for a market using only trades up to cutoff_timestamp.
    This prevents look-ahead bias in backtesting.
    
    Returns signal dict if confidence meets threshold, None otherwise.
    """
    # Filter to only BUY trades before cutoff
    relevant_trades = [
        t for t in trades
        if t.get("side") == "BUY" and (t.get("timestamp") or 0) <= cutoff_timestamp
    ]
    
    if len(relevant_trades) < config.min_participants:
        return None
    
    # Group by outcome
    outcomes: dict[str, list[dict]] = defaultdict(list)
    title = None
    for t in relevant_trades:
        outcome = str(t.get("outcome", ""))
        outcomes[outcome].append(t)
        if title is None and t.get("title"):
            title = str(t.get("title"))
    
    if not outcomes:
        return None
    
    # Find leading outcome (most traders betting on it)
    leading_outcome, leading_trades = max(outcomes.items(), key=lambda kv: len(kv[1]))
    
    total_participants = len(relevant_trades)
    participants = len(leading_trades)
    consensus_percent = participants / total_participants if total_participants else 0.0
    
    # Compute price band
    prices = [p for p in (_safe_float(t.get("price")) for t in leading_trades) if p is not None]
    mean_entry = None
    tight_band = False
    
    if prices:
        band_min, band_max, mean_entry, stddev = _compute_band(prices)
        tight_band = ((band_max - band_min) <= 0.03) or (stddev <= 0.01)
    
    # Compute confidence score with participant-based cap
    # Cap max score based on participant count (same logic as compute.py)
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
    
    # Base score components
    weighted_score = consensus_percent * 35
    raw_score = consensus_percent * 20
    tight_score = 15 if tight_band else 0
    cooked_score = 15  # Assume not cooked in backtest
    participant_score = min(15, np.log1p(participants) * 3)
    
    raw_total = weighted_score + raw_score + tight_score + cooked_score + participant_score
    confidence_score = min(raw_total / 100.0, max_score)
    
    if confidence_score < config.min_confidence:
        return None
    
    return {
        "condition_id": trades[0].get("condition_id"),
        "title": title,
        "leading_outcome": leading_outcome,
        "confidence_score": confidence_score,
        "consensus_percent": consensus_percent,
        "participants": participants,
        "total_participants": total_participants,
        "mean_entry": mean_entry,
        "tight_band": tight_band,
        "signal_timestamp": cutoff_timestamp,
    }


def calculate_pnl(
    bet_size: float,
    entry_price: float | None,
    won: bool,
) -> float:
    """
    Calculate P&L for a trade.
    
    For prediction markets:
    - If won: payout is bet_size / entry_price (buying shares that resolve to $1)
    - If lost: lose the bet
    """
    if won:
        if entry_price and entry_price > 0:
            # Profit = (1 / entry_price - 1) * bet_size
            # Example: entry at 0.7, bet $100 -> win $100 * (1/0.7 - 1) = $42.86
            return bet_size * (1 / entry_price - 1)
        else:
            # No entry price, assume 50% odds
            return bet_size
    else:
        return -bet_size


def compute_metrics(trades: list[BacktestTrade]) -> dict[str, float]:
    """Compute aggregate metrics from trade results."""
    resolved_trades = [t for t in trades if t.won is not None]
    
    if not resolved_trades:
        return {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "pending_trades": len(trades),
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_invested": 0.0,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
        }
    
    winning_trades = [t for t in resolved_trades if t.won]
    losing_trades = [t for t in resolved_trades if not t.won]
    
    total_pnl = sum(t.pnl or 0 for t in resolved_trades)
    total_invested = sum(t.bet_size for t in resolved_trades)
    
    gross_wins = sum(t.pnl or 0 for t in winning_trades if (t.pnl or 0) > 0)
    gross_losses = abs(sum(t.pnl or 0 for t in losing_trades if (t.pnl or 0) < 0))
    
    # Compute equity curve and max drawdown
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    returns = []
    
    for t in sorted(resolved_trades, key=lambda x: x.signal_timestamp):
        equity += t.pnl or 0
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        if t.bet_size > 0:
            returns.append((t.pnl or 0) / t.bet_size)
    
    # Sharpe ratio (assuming daily returns, annualized)
    sharpe_ratio = 0.0
    if returns and len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
    
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "pending_trades": len(trades) - len(resolved_trades),
        "win_rate": len(winning_trades) / len(resolved_trades) if resolved_trades else 0.0,
        "total_pnl": total_pnl,
        "total_invested": total_invested,
        "roi": total_pnl / total_invested if total_invested > 0 else 0.0,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": float(sharpe_ratio),
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else float('inf') if gross_wins > 0 else 0.0,
    }


def build_equity_curve(trades: list[BacktestTrade]) -> list[dict[str, Any]]:
    """Build equity curve data for charting."""
    resolved_trades = [t for t in trades if t.won is not None]
    sorted_trades = sorted(resolved_trades, key=lambda x: x.signal_timestamp)
    
    equity = 0.0
    curve = []
    
    for t in sorted_trades:
        equity += t.pnl or 0
        curve.append({
            "timestamp": t.signal_timestamp,
            "equity": equity,
            "pnl": t.pnl or 0,
            "trade_count": len(curve) + 1,
        })
    
    return curve


async def run_backtest(
    conn: sqlite3.Connection,
    settings: Settings,
    config: BacktestConfig,
    *,
    client: Any,
    sem: asyncio.Semaphore,
) -> BacktestResult:
    """
    Run a backtest using historical trade data and resolved market outcomes.
    
    Algorithm:
    1. Fetch resolved markets from CLOB API
    2. Get historical trades from the lookback period
    3. For each resolved market with trader activity:
       - Compute signal at the earliest trade time (point-in-time)
       - If signal meets confidence threshold, simulate a trade
       - Compare prediction to actual outcome
       - Calculate P&L
    4. Aggregate results
    """
    run_id = str(uuid.uuid4())[:8]
    result = BacktestResult(run_id=run_id, config=config)
    
    try:
        # Step 1: Get historical trades (READ ONLY - releases lock immediately)
        now = int(time.time())
        lookback_start = now - (config.lookback_days * 24 * 60 * 60)
        
        # Read trades and immediately release the connection
        all_trades = db.get_trades_in_timerange(conn, start_ts=lookback_start)
        
        if not all_trades:
            # No trades in period - do a quick write
            for attempt in range(5):
                try:
                    db.create_backtest_run(conn, run_id, config.to_dict())
                    db.update_backtest_run(conn, run_id, status="completed")
                    conn.commit()
                    break
                except Exception:
                    await asyncio.sleep(1)
            return result
        
        # Group trades by condition_id (in memory)
        trades_by_condition: dict[str, list[dict]] = defaultdict(list)
        for t in all_trades:
            cid = t.get("condition_id")
            if cid:
                trades_by_condition[cid].append(t)
        
        # Step 2: Fetch resolutions for the specific markets we have trades for
        # This is network-only, no DB involved
        resolutions: dict[str, str] = {}
        resolution_data: list[dict] = []
        
        async def fetch_market_resolution(condition_id: str) -> None:
            """Fetch resolution for a single market."""
            url = f"{settings.clob_base}/markets/{condition_id}"
            try:
                async with sem:
                    resp = await client.get(url, headers={"User-Agent": settings.user_agent})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("closed"):
                        tokens = data.get("tokens", [])
                        for t in tokens:
                            if t.get("winner"):
                                winner = t.get("outcome")
                                resolutions[condition_id] = winner
                                resolution_data.append({
                                    "condition_id": condition_id,
                                    "winning_outcome": winner,
                                    "resolved_at": now,
                                })
                                break
            except Exception:
                pass  # Skip markets we can't fetch
        
        # Fetch resolutions in parallel (limit concurrency)
        condition_ids = list(trades_by_condition.keys())
        batch_size = 50
        for i in range(0, len(condition_ids), batch_size):
            batch = condition_ids[i:i + batch_size]
            await asyncio.gather(*[fetch_market_resolution(cid) for cid in batch])
        
        # Step 3: Generate signals for each market
        backtest_trades: list[BacktestTrade] = []
        
        for condition_id, trades in trades_by_condition.items():
            # Only process markets we have resolution data for
            if condition_id not in resolutions:
                continue
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x.get("timestamp") or 0)
            
            if not sorted_trades:
                continue
            
            # Use the timestamp when we would have had enough data
            # (after min_participants trades)
            buy_trades = [t for t in sorted_trades if t.get("side") == "BUY"]
            if len(buy_trades) < config.min_participants:
                continue
            
            # Signal timestamp is when we had enough participants
            signal_ts = buy_trades[config.min_participants - 1].get("timestamp") or 0
            
            # Compute signal at that point in time
            signal = compute_signal_at_time(sorted_trades, signal_ts, config)
            
            if not signal:
                continue
            
            # Create simulated trade
            predicted_outcome = signal["leading_outcome"]
            actual_outcome = resolutions[condition_id]
            won = predicted_outcome == actual_outcome
            
            bet_size = calculate_bet_size(signal["confidence_score"], config)
            entry_price = signal.get("mean_entry")
            pnl = calculate_pnl(bet_size, entry_price, won)
            
            trade = BacktestTrade(
                condition_id=condition_id,
                title=signal.get("title"),
                signal_timestamp=signal_ts,
                predicted_outcome=predicted_outcome,
                confidence_score=signal["confidence_score"],
                bet_size=bet_size,
                entry_price=entry_price,
                actual_outcome=actual_outcome,
                pnl=pnl,
                won=won,
            )
            backtest_trades.append(trade)
        
        # Step 4: Aggregate results
        metrics = compute_metrics(backtest_trades)
        
        result.trades = backtest_trades
        result.total_trades = metrics["total_trades"]
        result.winning_trades = metrics["winning_trades"]
        result.losing_trades = metrics["losing_trades"]
        result.pending_trades = metrics["pending_trades"]
        result.win_rate = metrics["win_rate"]
        result.total_pnl = metrics["total_pnl"]
        result.total_invested = metrics["total_invested"]
        result.roi = metrics["roi"]
        result.max_drawdown = metrics["max_drawdown"]
        result.sharpe_ratio = metrics["sharpe_ratio"]
        result.profit_factor = metrics["profit_factor"]
        result.equity_curve = build_equity_curve(backtest_trades)
        
        # Step 5: Batch all DB writes with retry logic
        trade_dicts = [
            {
                "run_id": run_id,
                "condition_id": t.condition_id,
                "title": t.title,
                "signal_timestamp": t.signal_timestamp,
                "predicted_outcome": t.predicted_outcome,
                "confidence_score": t.confidence_score,
                "bet_size": t.bet_size,
                "entry_price": t.entry_price,
                "actual_outcome": t.actual_outcome,
                "pnl": t.pnl,
                "won": t.won,
            }
            for t in backtest_trades
        ]
        
        # Retry DB writes with exponential backoff
        for attempt in range(5):
            try:
                db.create_backtest_run(conn, run_id, config.to_dict())
                
                if resolution_data:
                    db.bulk_upsert_resolutions(conn, resolution_data)
                
                if trade_dicts:
                    db.bulk_insert_backtest_trades(conn, trade_dicts)
                
                db.update_backtest_run(
                    conn,
                    run_id,
                    status="completed",
                    total_trades=result.total_trades,
                    winning_trades=result.winning_trades,
                    losing_trades=result.losing_trades,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    total_invested=result.total_invested,
                    roi=result.roi,
                    max_drawdown=result.max_drawdown,
                    sharpe_ratio=result.sharpe_ratio,
                )
                conn.commit()
                break
            except Exception as db_error:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)  # 1, 2, 4, 8 seconds
                else:
                    raise db_error
        
    except Exception as e:
        # Try to record failure, but don't fail if DB is locked
        try:
            db.create_backtest_run(conn, run_id, config.to_dict())
            db.update_backtest_run(conn, run_id, status=f"failed: {str(e)[:100]}")
            conn.commit()
        except Exception:
            pass
        raise
    
    return result
