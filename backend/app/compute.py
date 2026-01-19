from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import numpy as np
import sqlite3

from . import clustering, db
from .config import Settings
from .ingest import fetch_orderbook, fetch_active_markets


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _midpoint_from_book(book: dict[str, Any]) -> float | None:
    bids = book.get("bids") or []
    asks = book.get("asks") or []

    bid_prices: list[float] = []
    for b in bids:
        try:
            size = float(b.get("size", "0"))
            price = float(b.get("price"))
        except Exception:
            continue
        if size <= 0:
            continue
        bid_prices.append(price)

    ask_prices: list[float] = []
    for a in asks:
        try:
            size = float(a.get("size", "0"))
            price = float(a.get("price"))
        except Exception:
            continue
        if size <= 0:
            continue
        ask_prices.append(price)

    if not bid_prices or not ask_prices:
        return None

    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    return (best_bid + best_ask) / 2.0


def _compute_band(prices: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(prices, dtype=np.float64)
    band_min = float(np.min(arr))
    band_max = float(np.max(arr))
    mean = float(np.mean(arr))
    stddev = float(np.std(arr, ddof=0))
    return band_min, band_max, mean, stddev


def _compute_weighted_consensus(
    recs: list[dict[str, Any]],
    leading_outcome: str,
    accuracy_map: dict[str, float],
    default_weight: float = 0.5,
) -> tuple[float, float]:
    """
    Compute weighted consensus based on trader accuracy.
    
    Returns:
        (weighted_consensus_percent, weighted_participants)
    
    Formula: weighted_signal = sum(trader_accuracy * trader_vote) / sum(trader_accuracy)
    Where trader_vote = 1 if voting for leading outcome, 0 otherwise.
    """
    total_weight = 0.0
    leading_weight = 0.0
    
    for rec in recs:
        wallet = rec["wallet"].lower()
        # Use trader's recent accuracy as weight, or default if not available
        accuracy = accuracy_map.get(wallet, default_weight)
        # Ensure minimum weight to avoid division issues
        weight = max(accuracy, 0.1)
        
        total_weight += weight
        if rec["outcome"] == leading_outcome:
            leading_weight += weight
    
    if total_weight == 0:
        return 0.0, 0.0
    
    weighted_consensus = leading_weight / total_weight
    return weighted_consensus, leading_weight


def _compute_confidence_score(
    weighted_consensus: float,
    consensus_percent: float,
    tight_band: bool,
    cooked: bool,
    participants: int,
) -> float:
    """
    Compute an overall confidence score for a market signal.
    
    Factors:
    - Weighted consensus (most important - 40%)
    - Raw consensus (20%)
    - Entry price tightness (20%)
    - Not cooked / price hasn't moved (15%)
    - Number of participants (5%)
    """
    # Weighted consensus contribution (0-40 points)
    weighted_score = weighted_consensus * 40
    
    # Raw consensus contribution (0-20 points)
    raw_score = consensus_percent * 20
    
    # Tight band contribution (0-20 points)
    tight_score = 20 if tight_band else 0
    
    # Not cooked contribution (0-15 points)
    cooked_score = 15 if not cooked else 0
    
    # Participants contribution (0-5 points, scaled logarithmically)
    # More participants = more confidence, but diminishing returns
    participant_score = min(5, np.log1p(participants) * 1.5)
    
    total = weighted_score + raw_score + tight_score + cooked_score + participant_score
    # Normalize to 0-1 range
    return total / 100.0


async def compute_and_store(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    wallets: list[str],
    client: Any,
    sem: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wallet_set = {w.lower() for w in wallets}
    buy_rows = db.get_all_buy_trades(conn)
    
    # Get accuracy map for weighting
    accuracy_map = db.get_wallet_accuracy_map(conn)

    latest_buy: dict[tuple[str, str], dict[str, Any]] = {}
    for r in buy_rows:
        w = str(r["wallet"]).lower()
        if w not in wallet_set:
            continue
        condition_id = str(r["condition_id"])
        key = (w, condition_id)
        if key in latest_buy:
            continue
        latest_buy[key] = {
            "wallet": w,
            "condition_id": condition_id,
            "outcome": r["outcome"],
            "price": _safe_float(r["price"]),
            "timestamp": r["timestamp"],
            "asset_id": r["asset_id"],
            "title": r["title"],
        }

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in latest_buy.values():
        by_condition[rec["condition_id"]].append(rec)

    market_rows: list[dict[str, Any]] = []
    # Collect pricing lookups for each condition (only for leading outcome token).
    condition_to_asset: dict[str, str] = {}
    for condition_id, recs in by_condition.items():
        outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        title = None
        for rec in recs:
            outcomes[str(rec["outcome"])].append(rec)
            if title is None and rec.get("title"):
                title = str(rec.get("title"))

        if not outcomes:
            continue

        leading_outcome, leading_recs = max(outcomes.items(), key=lambda kv: len(kv[1]))
        total_participants = len(recs)
        participants = len(leading_recs)
        consensus_percent = (participants / total_participants) if total_participants else 0.0
        
        # Compute weighted consensus based on trader accuracy
        weighted_consensus, weighted_participants = _compute_weighted_consensus(
            recs, leading_outcome, accuracy_map
        )

        prices = [p for p in (_safe_float(x.get("price")) for x in leading_recs) if p is not None]
        band_min = band_max = mean_entry = stddev = None
        tight_band = False
        if prices:
            band_min, band_max, mean_entry, stddev = _compute_band(prices)
            tight_band = ((band_max - band_min) <= 0.03) or (stddev <= 0.01)

        asset_id = next((str(x.get("asset_id")) for x in leading_recs if x.get("asset_id")), None)
        if settings.enable_pricing and asset_id:
            condition_to_asset[condition_id] = asset_id

        market_rows.append(
            {
                "condition_id": condition_id,
                "title": title,
                "leading_outcome": leading_outcome,
                "consensus_percent": float(consensus_percent),
                "weighted_consensus_percent": float(weighted_consensus),
                "total_participants": int(total_participants),
                "participants": int(participants),
                "weighted_participants": float(weighted_participants),
                "band_min": band_min,
                "band_max": band_max,
                "mean_entry": mean_entry,
                "stddev": stddev,
                "tight_band": bool(tight_band),
                "midpoint": None,
                "cooked": False,
                "price_unavailable": True,
                "ready": False,
                "confidence_score": 0.0,
                "end_date": None,
                "is_closed": False,
                "is_active": True,
            }
        )

    # Fetch active markets metadata to filter out closed/expired markets
    active_markets = await fetch_active_markets(settings, client=client, sem=sem, limit=1000)
    active_condition_ids = {m.get("conditionId") for m in active_markets if m.get("conditionId")}
    market_metadata = {
        m.get("conditionId"): {
            "end_date": m.get("endDate"),
            "is_closed": m.get("closed", False),
            "is_active": m.get("active", True),
        }
        for m in active_markets
        if m.get("conditionId")
    }
    
    # Update market rows with metadata
    for r in market_rows:
        cid = r["condition_id"]
        if cid in market_metadata:
            meta = market_metadata[cid]
            r["end_date"] = meta.get("end_date")
            r["is_closed"] = meta.get("is_closed", False)
            r["is_active"] = meta.get("is_active", True)
        else:
            # Market not in active list - mark as closed/inactive
            r["is_closed"] = True
            r["is_active"] = False

    # Pricing in parallel, bounded by sem; dedupe by asset_id.
    asset_to_midpoint: dict[str, float | None] = {}

    async def _fetch(asset_id: str) -> None:
        book = await fetch_orderbook(asset_id, settings, client=client, sem=sem)
        asset_to_midpoint[asset_id] = None if not book else _midpoint_from_book(book)

    unique_assets = sorted(set(condition_to_asset.values()))
    await asyncio.gather(*[_fetch(a) for a in unique_assets])

    for r in market_rows:
        asset_id = condition_to_asset.get(r["condition_id"])
        midpoint = asset_to_midpoint.get(asset_id) if asset_id else None
        r["midpoint"] = midpoint
        if midpoint is None or r.get("mean_entry") is None:
            r["cooked"] = False
            r["price_unavailable"] = True
        else:
            r["price_unavailable"] = False
            r["cooked"] = abs(float(midpoint) - float(r["mean_entry"])) > 0.05
        r["ready"] = (
            float(r["weighted_consensus_percent"]) >= 0.75 
            and bool(r["tight_band"]) 
            and not bool(r["cooked"])
        )
        
        # Compute confidence score
        r["confidence_score"] = _compute_confidence_score(
            r["weighted_consensus_percent"],
            r["consensus_percent"],
            r["tight_band"],
            r["cooked"],
            r["participants"],
        )

    db.replace_market_state(conn, market_rows)

    # Clusters
    trades_by_wallet: dict[str, list[dict[str, Any]]] = {}
    for w in wallets:
        rows = db.get_recent_trades_by_wallet(conn, wallet=w, limit=50)
        trades_by_wallet[w.lower()] = [
            {
                "condition_id": row["condition_id"],
                "outcome": row["outcome"],
                "side": row["side"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    signatures = clustering.build_signatures(trades_by_wallet, max_trades=50)
    clusters = clustering.find_copycat_clusters(signatures, threshold=0.80, min_size=3)
    cluster_rows: list[dict[str, Any]] = [
        {"cluster_id": c.cluster_id, "wallets": c.wallets, "size": len(c.wallets)} for c in clusters
    ]
    db.replace_clusters(conn, cluster_rows)

    return market_rows, cluster_rows

