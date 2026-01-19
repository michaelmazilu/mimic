"use client";

import React, { useEffect, useMemo, useState } from "react";

import { ClustersPanel } from "@/app/components/ClustersPanel";
import { MarketsTable } from "@/app/components/MarketsTable";
import { SettingsPanel } from "@/app/components/SettingsPanel";
import { WalletsPanel } from "@/app/components/WalletsPanel";
import { DEFAULT_BACKEND_URL, getState, getWallets, refreshNow } from "@/app/lib/api";
import { DEFAULT_SETTINGS, loadSettings, saveSettings, type ClientSettings } from "@/app/lib/settings";
import type { RefreshResponse, StateResponse, WalletStats } from "@/app/lib/types";

function fmtTs(ts: number | null | undefined): string {
  if (!ts) return "—";
  try {
    return new Date(ts * 1000).toLocaleTimeString([], { 
      hour: "2-digit", 
      minute: "2-digit" 
    });
  } catch {
    return String(ts);
  }
}

export default function HomePage() {
  const backendUrl = DEFAULT_BACKEND_URL;
  const [settings, setSettings] = useState<ClientSettings>(DEFAULT_SETTINGS);
  const [state, setState] = useState<StateResponse | null>(null);
  const [refresh, setRefresh] = useState<RefreshResponse | null>(null);
  const [walletStats, setWalletStats] = useState<WalletStats[]>([]);
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
    async function fetchWallets() {
      try {
        const resp = await getWallets(backendUrl, { orderBy: "recent_accuracy_7d", limit: 50 });
        if (cancelled) return;
        setWalletStats(resp.wallets);
      } catch {
        // Silently fail
      }
    }
    fetchWallets();
    const id = window.setInterval(fetchWallets, pollInterval * 5);
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
    <div className="container">
      {/* Header */}
      <div className="section">
        <h1 style={{ 
          fontSize: 32, 
          fontWeight: 600, 
          letterSpacing: "-0.03em",
          margin: "0 0 8px" 
        }}>
          Mimic
        </h1>
        <p className="secondary" style={{ margin: 0, fontSize: 15 }}>
          Polymarket copy-trading signals
        </p>
      </div>

      {/* Status Bar */}
      <div className="section" style={{ 
        display: "flex", 
        gap: 32, 
        flexWrap: "wrap",
        paddingBottom: 24,
        borderBottom: "1px solid var(--border)"
      }}>
        <div className="stat">
          Last refresh <span className="stat-value">{fmtTs(state?.lastRefreshTs)}</span>
        </div>
        <div className="stat">
          Wallets <span className="stat-value">{state?.walletCount ?? 0}</span>
        </div>
        <div className="stat">
          Trades <span className="stat-value">{state?.tradeCount ?? 0}</span>
        </div>
        <div className="stat">
          Status{" "}
          <span className="stat-value">
            {state?.refreshInProgress ? "Refreshing..." : "Idle"}
          </span>
        </div>
      </div>

      {/* Error Display */}
      {error ? (
        <div className="section" style={{ 
          padding: 16, 
          border: "1px solid var(--border)", 
          borderRadius: 8 
        }}>
          <div style={{ fontWeight: 500, marginBottom: 4 }}>Connection Error</div>
          <div className="secondary" style={{ fontSize: 13 }}>{error}</div>
        </div>
      ) : null}

      {refresh?.status === "error" ? (
        <div className="section" style={{ 
          padding: 16, 
          border: "1px solid var(--border)", 
          borderRadius: 8 
        }}>
          <div style={{ fontWeight: 500, marginBottom: 4 }}>Refresh Error</div>
          <div className="secondary" style={{ fontSize: 13 }}>{refresh.error}</div>
        </div>
      ) : null}

      {/* Markets Table */}
      <MarketsTable markets={markets} />

      {/* Settings and Clusters Row */}
      <div className="section">
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
      </div>

      {/* Wallets Table */}
      <WalletsPanel wallets={walletStats} />
    </div>
  );
}
