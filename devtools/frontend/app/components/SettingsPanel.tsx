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
    <div className="card" style={{ flex: "1 1 320px", minWidth: 280 }}>
      <div className="title">Settings</div>
      <div className="subtitle">Local configuration</div>

      <div style={{ display: "grid", gap: 16 }}>
        <div>
          <div className="label">Wallets to track</div>
          <input
            className="input"
            type="number"
            min={1}
            max={200}
            value={settings.nWallets}
            onChange={(e) => onChange({ ...settings, nWallets: Number(e.target.value) })}
          />
        </div>
        <div>
          <div className="label">Poll interval (ms)</div>
          <input
            className="input"
            type="number"
            min={250}
            max={60000}
            value={settings.pollIntervalMs}
            onChange={(e) => onChange({ ...settings, pollIntervalMs: Number(e.target.value) })}
          />
        </div>
        <div>
          <div className="label">Refresh interval (ms)</div>
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

      <div style={{ marginTop: 20 }}>
        <button className="btn" onClick={onRefreshNow} disabled={refreshInProgress}>
          {refreshInProgress ? "Refreshing..." : "Refresh Now"}
        </button>
      </div>
    </div>
  );
}
