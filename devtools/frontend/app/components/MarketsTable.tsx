import Link from "next/link";
import React from "react";

import type { MarketSummary } from "@/app/lib/types";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(0)}%`;
}

function fmtEndDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays < 0) return "Ended";
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Tomorrow";
    if (diffDays <= 7) return `${diffDays}d`;
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "—";
  }
}

export function MarketsTable({ markets }: { markets: MarketSummary[] }) {
  // Filter to only show active markets
  const activeMarkets = markets.filter(m => m.isActive && !m.isClosed);
  
  return (
    <div className="section">
      <div className="title">Active Markets</div>
      <div className="subtitle">
        {activeMarkets.length} active markets sorted by confidence score
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Market</th>
              <th>Outcome</th>
              <th>Confidence</th>
              <th>Participants</th>
              <th>Ends</th>
              <th>Ready</th>
            </tr>
          </thead>
          <tbody>
            {activeMarkets.map((m) => (
              <tr key={m.conditionId}>
                <td style={{ maxWidth: 400 }}>
                  <Link href={`/market/${encodeURIComponent(m.conditionId)}`}>
                    {m.title ? (
                      <span style={{ fontWeight: 500 }}>
                        {m.title.length > 70 ? m.title.slice(0, 70) + "..." : m.title}
                      </span>
                    ) : (
                      <code>{m.conditionId.slice(0, 20)}...</code>
                    )}
                  </Link>
                </td>
                <td>
                  <span style={{ fontWeight: 500 }}>{m.leadingOutcome ?? "—"}</span>
                </td>
                <td>
                  <span style={{ fontWeight: 600 }}>{fmtPct(m.confidenceScore)}</span>
                  <span className="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                    ({fmtPct(m.weightedConsensusPercent)} weighted)
                  </span>
                </td>
                <td>
                  <span>{m.participants}</span>
                  <span className="secondary"> / {m.totalParticipants}</span>
                </td>
                <td className="secondary" style={{ fontSize: 13 }}>
                  {fmtEndDate(m.endDate)}
                </td>
                <td>
                  {m.ready ? (
                    <span className="ready-badge">Ready</span>
                  ) : (
                    <span className="secondary">—</span>
                  )}
                </td>
              </tr>
            ))}
            {!activeMarkets.length ? (
              <tr>
                <td colSpan={6} className="secondary" style={{ textAlign: "center", padding: 32 }}>
                  No active markets yet
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
