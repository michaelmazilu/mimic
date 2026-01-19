"use client";

import React from "react";

import type { ClientSettings } from "@/app/lib/settings";

export function SettingsPanel({
  settings,
  onChange,
  onRefreshNow,
  refreshInProgress
}: {
  settings: ClientSettings;
  onChange: (next: ClientSettings) => void;
  onRefreshNow: () => Promise<void>;
  refreshInProgress: boolean;
}) {
  return (
    <div className="glass card" style={{ flex: "1 1 320px", minWidth: 280 }}>
      <div className="title">Settings (local)</div>
      <div className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
        Stored in <code>localStorage</code>. Backend enforces its own refresh interval.
      </div>

      <div className="row" style={{ gap: 12 }}>
        <div style={{ flex: "1 1 160px" }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            N_WALLETS
          </div>
          <input
            className="input"
            type="number"
            min={1}
            max={200}
            value={settings.nWallets}
            onChange={(e) => onChange({ ...settings, nWallets: Number(e.target.value) })}
          />
        </div>
        <div style={{ flex: "1 1 160px" }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            POLL_INTERVAL_MS
          </div>
          <input
            className="input"
            type="number"
            min={250}
            max={60000}
            value={settings.pollIntervalMs}
            onChange={(e) => onChange({ ...settings, pollIntervalMs: Number(e.target.value) })}
          />
        </div>
        <div style={{ flex: "1 1 160px" }}>
          <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            REFRESH_INTERVAL_MS
          </div>
          <input
            className="input"
            type="number"
            min={1000}
            max={600000}
            value={settings.refreshIntervalMs}
            onChange={(e) => onChange({ ...settings, refreshIntervalMs: Number(e.target.value) })}
          />
        </div>
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn" onClick={onRefreshNow} disabled={refreshInProgress}>
          {refreshInProgress ? "Refreshing…" : "Refresh now"}
        </button>
        <span className="muted" style={{ fontSize: 12 }}>
          Uses your <code>N_WALLETS</code> value.
        </span>
      </div>
    </div>
  );
}

