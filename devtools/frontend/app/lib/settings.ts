export type ClientSettings = {
  nWallets: number;
  pollIntervalMs: number;
  refreshIntervalMs: number;
};

export const DEFAULT_SETTINGS: ClientSettings = {
  nWallets: 500,
  pollIntervalMs: 2000,
  refreshIntervalMs: 60000
};

const STORAGE_KEY = "mimic.settings.v1";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

function clampInt(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(n)));
}

export function loadSettings(): ClientSettings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) return DEFAULT_SETTINGS;
    const nWallets = clampInt(Number(parsed.nWallets), 1, 500);
    const pollIntervalMs = clampInt(Number(parsed.pollIntervalMs), 250, 60000);
    const refreshIntervalMs = clampInt(Number(parsed.refreshIntervalMs), 1000, 10 * 60 * 1000);
    return { nWallets, pollIntervalMs, refreshIntervalMs };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: ClientSettings): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // ignore
  }
}

