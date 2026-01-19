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


async def fetch_leaderboard(settings: Settings, *, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> list[dict]:
    url = f"{settings.data_api_base}/v1/leaderboard"
    data = await _get_json(
        client,
        url,
        sem=sem,
        headers={"User-Agent": settings.user_agent},
    )
    if not isinstance(data, list):
        raise IngestError("Unexpected leaderboard response shape")
    return data


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

