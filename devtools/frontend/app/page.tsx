"use client";

import React, { useEffect, useMemo, useState } from "react";

import { ClustersPanel } from "@/app/components/ClustersPanel";
import { MarketsTable } from "@/app/components/MarketsTable";
import { SettingsPanel } from "@/app/components/SettingsPanel";
import { DEFAULT_BACKEND_URL, getState, refreshNow } from "@/app/lib/api";
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type ClientSettings } from "@/app/lib/settings";
import type { RefreshResponse, StateResponse } from "@/app/lib/types";

function fmtTs(ts: number | null | undefined): string {
  if (!ts) return "—";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

export default function HomePage() {
  const backendUrl = DEFAULT_BACKEND_URL;
  const [settings, setSettings] = useState<ClientSettings>(DEFAULT_SETTINGS);
  const [state, setState] = useState<StateResponse | null>(null);
  const [refresh, setRefresh] = useState<RefreshResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const pollInterval = useMemo(() => settings.pollIntervalMs, [settings.pollIntervalMs]);
  const refreshInterval = useMemo(() => settings.refreshIntervalMs, [settings.refreshIntervalMs]);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const s = await getState(backendUrl);
        if (cancelled) return;
        setState(s);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    }
    tick();
    const id = window.setInterval(tick, pollInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [backendUrl, pollInterval]);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const r = await refreshNow(backendUrl, { nWallets: settings.nWallets });
        if (cancelled) return;
        setRefresh(r);
      } catch (e) {
        if (cancelled) return;
        setRefresh({
          status: "error",
          nowTs: Math.floor(Date.now() / 1000),
          lastRefreshTs: state?.lastRefreshTs ?? null,
          nextRefreshEarliestTs: state?.nextRefreshEarliestTs ?? null,
          refreshed: false,
          walletUpserts: 0,
          tradeInserts: 0,
          nWallets: settings.nWallets,
          tradesLimit: null,
          error: e instanceof Error ? e.message : String(e)
        });
      }
    }
    tick();
    const id = window.setInterval(tick, refreshInterval);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [backendUrl, refreshInterval, settings.nWallets, state?.lastRefreshTs, state?.nextRefreshEarliestTs]);

  const markets = state?.markets ?? [];
  const clusters = state?.clusters ?? [];

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="glass card">
        <div className="title">Mimic — Polymarket copy-trading signal dashboard</div>
        <div className="row" style={{ gap: 10 }}>
          <span className="badge">
            backend: <code>{backendUrl}</code>
          </span>
          <span className="badge">
            last refresh: <span className="muted">{fmtTs(state?.lastRefreshTs ?? null)}</span>
          </span>
          <span className="badge">
            wallets: <span className="muted">{state?.walletCount ?? 0}</span>
          </span>
          <span className="badge">
            trades: <span className="muted">{state?.tradeCount ?? 0}</span>
          </span>
          <span className="badge">
            backend interval: <span className="muted">{state?.refreshIntervalSec ?? "—"}s</span>
          </span>
          <span className="badge">
            refresh status:{" "}
            <span className={state?.refreshInProgress ? "warn" : "muted"}>
              {state?.refreshInProgress ? "in progress" : "idle"}
            </span>
          </span>
        </div>
        {error ? (
          <div style={{ marginTop: 12 }} className="bad">
            <div style={{ fontWeight: 650 }}>State poll error</div>
            <div className="muted" style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {error}
            </div>
          </div>
        ) : null}

        {refresh?.status === "error" ? (
          <div style={{ marginTop: 12 }} className="bad">
            <div style={{ fontWeight: 650 }}>Refresh error</div>
            <div className="muted" style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {refresh.error}
            </div>
          </div>
        ) : null}
      </div>

      <div className="row">
        <SettingsPanel
          settings={settings}
          onChange={(next) => setSettings(next)}
          onRefreshNow={async () => {
            const r = await refreshNow(backendUrl, { nWallets: settings.nWallets });
            setRefresh(r);
          }}
          refreshInProgress={state?.refreshInProgress ?? false}
        />
        <ClustersPanel clusters={clusters} />
      </div>

      <MarketsTable markets={markets} />
    </div>
  );
}

