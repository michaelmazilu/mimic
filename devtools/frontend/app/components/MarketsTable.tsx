import Link from "next/link";
import React from "react";

import type { MarketSummary } from "@/app/lib/types";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(0)}%`;
}

export function MarketsTable({ markets }: { markets: MarketSummary[] }) {
  return (
    <div className="section">
      <div className="title">Markets</div>
      <div className="subtitle">Sorted by confidence score</div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Market</th>
              <th>Outcome</th>
              <th>Confidence</th>
              <th>Participants</th>
              <th>Ready</th>
            </tr>
          </thead>
          <tbody>
            {markets.map((m) => (
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
                <td>
                  {m.ready ? (
                    <span className="ready-badge">Ready</span>
                  ) : (
                    <span className="secondary">—</span>
                  )}
                </td>
              </tr>
            ))}
            {!markets.length ? (
              <tr>
                <td colSpan={5} className="secondary" style={{ textAlign: "center", padding: 32 }}>
                  No markets yet
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
