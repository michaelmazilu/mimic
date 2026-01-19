"use client";

import React from "react";

import type { WalletStats } from "@/app/lib/types";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(0)}%`;
}

export function WalletsPanel({ wallets }: { wallets: WalletStats[] }) {
  if (!wallets.length) {
    return (
      <div className="section">
        <div className="title">Top Traders</div>
        <div className="subtitle">By recent accuracy</div>
        <div className="secondary">No wallet stats available yet</div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="title">Top Traders</div>
      <div className="subtitle">Ranked by 7-day accuracy</div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Wallet</th>
              <th>7d Accuracy</th>
              <th>30d Accuracy</th>
              <th>Win Rate</th>
              <th>Record</th>
            </tr>
          </thead>
          <tbody>
            {wallets.slice(0, 15).map((w, idx) => (
              <tr key={w.wallet}>
                <td className="secondary">{w.rank ?? idx + 1}</td>
                <td>
                  <code style={{ fontSize: 12 }}>{w.wallet.slice(0, 12)}...</code>
                  {w.userName ? (
                    <div className="secondary" style={{ fontSize: 12, marginTop: 2 }}>
                      {w.userName}
                    </div>
                  ) : null}
                </td>
                <td>
                  {w.recentTrades7d >= 3 ? (
                    <span style={{ fontWeight: 500 }}>{fmtPct(w.recentAccuracy7d)}</span>
                  ) : (
                    <span className="secondary">—</span>
                  )}
                </td>
                <td>
                  {w.recentTrades30d >= 3 ? (
                    <span>{fmtPct(w.recentAccuracy30d)}</span>
                  ) : (
                    <span className="secondary">—</span>
                  )}
                </td>
                <td>
                  {w.totalTrades > 0 ? (
                    <span>{fmtPct(w.winRate)}</span>
                  ) : (
                    <span className="secondary">—</span>
                  )}
                </td>
                <td>
                  <span>{w.wonTrades}W</span>
                  <span className="secondary"> / </span>
                  <span>{w.lostTrades}L</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
