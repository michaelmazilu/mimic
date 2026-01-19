from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


def round_timestamp_5min(ts: int | None) -> int | None:
    if ts is None:
        return None
    return (int(ts) // 300) * 300


def trade_signature(trade: dict) -> str | None:
    condition_id = trade.get("condition_id") or trade.get("conditionId")
    outcome = trade.get("outcome")
    side = trade.get("side")
    ts = trade.get("timestamp")
    if not condition_id or not outcome or not side or ts is None:
        return None
    rounded = round_timestamp_5min(int(ts))
    return f"{condition_id}|{outcome}|{rounded}|{side}"


def build_signatures(
    trades_by_wallet: dict[str, list[dict]], *, max_trades: int = 50
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for wallet, trades in trades_by_wallet.items():
        sigs: set[str] = set()
        for t in trades[:max_trades]:
            sig = trade_signature(t)
            if sig is not None:
                sigs.add(sig)
        out[wallet] = sigs
    return out


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    wallets: list[str]


def _cluster_id(wallets: Iterable[str]) -> str:
    s = ",".join(sorted(wallets))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def find_copycat_clusters(
    signatures_by_wallet: dict[str, set[str]],
    *,
    threshold: float = 0.80,
    min_size: int = 3,
) -> list[Cluster]:
    wallets = list(signatures_by_wallet.keys())
    adjacency: dict[str, set[str]] = {w: set() for w in wallets}
    for i in range(len(wallets)):
        w1 = wallets[i]
        s1 = signatures_by_wallet[w1]
        for j in range(i + 1, len(wallets)):
            w2 = wallets[j]
            sim = jaccard_similarity(s1, signatures_by_wallet[w2])
            if sim >= threshold:
                adjacency[w1].add(w2)
                adjacency[w2].add(w1)

    seen: set[str] = set()
    clusters: list[Cluster] = []
    for w in wallets:
        if w in seen:
            continue
        if not adjacency[w]:
            seen.add(w)
            continue
        stack = [w]
        component: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            component.append(cur)
            stack.extend(adjacency[cur] - seen)
        if len(component) >= min_size:
            component_sorted = sorted(component)
            clusters.append(Cluster(cluster_id=_cluster_id(component_sorted), wallets=component_sorted))

    clusters.sort(key=lambda c: len(c.wallets), reverse=True)
    return clusters

