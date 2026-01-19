from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import Settings


class IngestError(RuntimeError):
    pass


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    sem: asyncio.Semaphore,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    async with sem:
        resp = await client.get(url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def fetch_leaderboard(
    settings: Settings,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    target_count: int = 500,
) -> list[dict]:
    """Fetch leaderboard with pagination to get up to target_count entries."""
    url = f"{settings.data_api_base}/v1/leaderboard"
    all_entries: list[dict] = []
    page_size = 50  # API max per request
    offset = 0
    
    while len(all_entries) < target_count:
        data = await _get_json(
            client,
            url,
            sem=sem,
            params={"limit": page_size, "offset": offset},
            headers={"User-Agent": settings.user_agent},
        )
        if not isinstance(data, list):
            raise IngestError("Unexpected leaderboard response shape")
        
        if not data:
            # No more entries
            break
            
        all_entries.extend(data)
        offset += page_size
        
        # If we got fewer than page_size, we've reached the end
        if len(data) < page_size:
            break
    
    return all_entries[:target_count]


async def fetch_trades_for_wallet(
    wallet: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limit: int,
) -> list[dict]:
    url = f"{settings.data_api_base}/trades"
    data = await _get_json(
        client,
        url,
        sem=sem,
        params={"user": wallet, "limit": int(limit)},
        headers={"User-Agent": settings.user_agent},
    )
    if not isinstance(data, list):
        return []
    return data


async def fetch_orderbook(
    asset_id: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    url = f"{settings.clob_base}/book"
    try:
        data = await _get_json(
            client,
            url,
            sem=sem,
            params={"token_id": asset_id},
            headers={"User-Agent": settings.user_agent},
        )
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def fetch_market_metadata(
    condition_ids: list[str],
    settings: Settings,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict[str, dict[str, Any]]:
    """
    Fetch market metadata (endDate, closed status) from Gamma API.
    Returns a dict mapping conditionId -> metadata.
    """
    gamma_base = "https://gamma-api.polymarket.com"
    result: dict[str, dict[str, Any]] = {}
    
    # Fetch in batches to avoid overwhelming the API
    batch_size = 50
    for i in range(0, len(condition_ids), batch_size):
        batch = condition_ids[i:i + batch_size]
        
        # Fetch all active markets and filter
        try:
            data = await _get_json(
                client,
                f"{gamma_base}/markets",
                sem=sem,
                params={"limit": 500, "closed": "false"},
                headers={"User-Agent": settings.user_agent},
            )
            if isinstance(data, list):
                for m in data:
                    cid = m.get("conditionId")
                    if cid:
                        result[cid] = {
                            "end_date": m.get("endDate"),
                            "closed": m.get("closed", False),
                            "active": m.get("active", True),
                            "question": m.get("question"),
                            "resolved": m.get("closed", False),
                        }
        except Exception:
            pass  # Continue without metadata if API fails
    
    return result


async def fetch_active_markets(
    settings: Settings,
    *,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Fetch list of currently active (not closed) markets."""
    gamma_base = "https://gamma-api.polymarket.com"
    try:
        data = await _get_json(
            client,
            f"{gamma_base}/markets",
            sem=sem,
            params={"limit": limit, "closed": "false", "active": "true"},
            headers={"User-Agent": settings.user_agent},
        )
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

