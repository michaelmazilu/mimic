"use client";

import React from "react";

import type { WalletStats } from "@/app/lib/types";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function StreakBadge({ streak }: { streak: number }) {
  if (streak === 0) return <span className="muted">—</span>;
  const isWin = streak > 0;
  const absStreak = Math.abs(streak);
  return (
    <span className={isWin ? "good" : "bad"}>
      {isWin ? "+" : ""}{streak} {isWin ? "W" : "L"}
    </span>
  );
}

function AccuracyBar({ accuracy, trades }: { accuracy: number; trades: number }) {
  if (trades < 3) {
    return <span className="muted" style={{ fontSize: 11 }}>insufficient data</span>;
  }
  const pct = Math.min(100, Math.max(0, accuracy * 100));
  const color = accuracy >= 0.6 ? "#4ade80" : accuracy >= 0.45 ? "#fbbf24" : "#f87171";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          width: 40,
          height: 5,
          background: "rgba(255,255,255,0.1)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 3,
          }}
        />
      </div>
      <span style={{ fontSize: 11, color }}>{fmtPct(accuracy)}</span>
    </div>
  );
}

export function WalletsPanel({ wallets }: { wallets: WalletStats[] }) {
  if (!wallets.length) {
    return (
      <div className="glass card" style={{ flex: "1 1 600px", minWidth: 340 }}>
        <div className="title">Top Traders (by recent accuracy)</div>
        <div className="muted" style={{ fontSize: 13 }}>
          No wallet stats available yet. Stats are computed after market outcomes are resolved.
        </div>
      </div>
    );
  }

  return (
    <div className="glass card" style={{ flex: "1 1 600px", minWidth: 340 }}>
      <div className="title">Top Traders (by recent accuracy)</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
        Traders with higher recent accuracy have more weight in consensus signals
      </div>
      <div className="tableWrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Wallet</th>
              <th>7d Accuracy</th>
              <th>30d Accuracy</th>
              <th>Win Rate</th>
              <th>W/L</th>
              <th>Streak</th>
            </tr>
          </thead>
          <tbody>
            {wallets.slice(0, 20).map((w, idx) => (
              <tr key={w.wallet}>
                <td className="muted">{w.rank ?? idx + 1}</td>
                <td style={{ maxWidth: 180 }}>
                  <code style={{ fontSize: 10 }}>{w.wallet.slice(0, 10)}...</code>
                  {w.userName ? (
                    <div className="muted" style={{ fontSize: 11 }}>{w.userName}</div>
                  ) : null}
                </td>
                <td>
                  <AccuracyBar accuracy={w.recentAccuracy7d} trades={w.recentTrades7d} />
                </td>
                <td>
                  <AccuracyBar accuracy={w.recentAccuracy30d} trades={w.recentTrades30d} />
                </td>
                <td className={w.winRate >= 0.55 ? "good" : w.winRate >= 0.45 ? "muted" : "bad"}>
                  {w.totalTrades > 0 ? fmtPct(w.winRate) : "—"}
                </td>
                <td>
                  <span className="badge" style={{ fontSize: 10 }}>
                    {w.wonTrades}/{w.lostTrades}
                  </span>
                </td>
                <td>
                  <StreakBadge streak={w.streak} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
