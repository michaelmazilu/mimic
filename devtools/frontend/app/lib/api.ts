import type { MarketDetailResponse, RefreshResponse, StateResponse, WalletsListResponse } from "@/app/lib/types";

export const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return (await res.json()) as T;
}

export async function getState(backendUrl: string = DEFAULT_BACKEND_URL): Promise<StateResponse> {
  return await fetchJson<StateResponse>(`${backendUrl}/state`);
}

export async function refreshNow(
  backendUrl: string = DEFAULT_BACKEND_URL,
  opts?: { nWallets?: number; tradesLimit?: number }
): Promise<RefreshResponse> {
  const params = new URLSearchParams();
  if (opts?.nWallets != null) params.set("n_wallets", String(opts.nWallets));
  if (opts?.tradesLimit != null) params.set("trades_limit", String(opts.tradesLimit));
  const qs = params.toString();
  return await fetchJson<RefreshResponse>(`${backendUrl}/refresh${qs ? `?${qs}` : ""}`);
}

export async function getMarketDetail(
  conditionId: string,
  backendUrl: string = DEFAULT_BACKEND_URL
): Promise<MarketDetailResponse> {
  return await fetchJson<MarketDetailResponse>(
    `${backendUrl}/market/${encodeURIComponent(conditionId)}`
  );
}

export async function getWallets(
  backendUrl: string = DEFAULT_BACKEND_URL,
  opts?: { orderBy?: string; limit?: number }
): Promise<WalletsListResponse> {
  const params = new URLSearchParams();
  if (opts?.orderBy) params.set("order_by", opts.orderBy);
  if (opts?.limit != null) params.set("limit", String(opts.limit));
  const qs = params.toString();
  return await fetchJson<WalletsListResponse>(`${backendUrl}/wallets${qs ? `?${qs}` : ""}`);
}

