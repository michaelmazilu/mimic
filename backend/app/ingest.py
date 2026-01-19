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

