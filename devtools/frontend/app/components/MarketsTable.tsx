import Link from "next/link";
import React from "react";

import type { MarketSummary } from "@/app/lib/types";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x: number | null | undefined, digits: number = 3): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function getConfidenceClass(score: number): string {
  if (score >= 0.7) return "good";
  if (score >= 0.5) return "warn";
  return "muted";
}

function ConfidenceBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 100));
  const color = score >= 0.7 ? "#4ade80" : score >= 0.5 ? "#fbbf24" : "#94a3b8";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          width: 50,
          height: 6,
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
      <span style={{ fontSize: 11, color }}>{fmtPct(score)}</span>
    </div>
  );
}

export function MarketsTable({ markets }: { markets: MarketSummary[] }) {
  return (
    <div className="glass card" style={{ flex: "1 1 820px", minWidth: 340 }}>
      <div className="title">Markets (sorted by confidence score)</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
        Confidence = weighted consensus (by trader accuracy) + price tightness + freshness
      </div>
      <div className="tableWrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>Market</th>
              <th>Outcome</th>
              <th>Confidence</th>
              <th>Weighted</th>
              <th>Raw</th>
              <th>Participants</th>
              <th>Band</th>
              <th>Status</th>
              <th>Ready</th>
            </tr>
          </thead>
          <tbody>
            {markets.map((m) => (
              <tr key={m.conditionId}>
                <td style={{ maxWidth: 320 }}>
                  <Link href={`/market/${encodeURIComponent(m.conditionId)}`}>
                    <code style={{ fontSize: 11 }}>{m.conditionId.slice(0, 16)}...</code>
                  </Link>
                  {m.title ? (
                    <div className="muted" style={{ marginTop: 4, fontSize: 11, lineHeight: 1.25 }}>
                      {m.title.length > 60 ? m.title.slice(0, 60) + "..." : m.title}
                    </div>
                  ) : null}
                </td>
                <td>
                  <span className="badge">{m.leadingOutcome ?? "—"}</span>
                </td>
                <td>
                  <ConfidenceBar score={m.confidenceScore} />
                </td>
                <td className={getConfidenceClass(m.weightedConsensusPercent)}>
                  {fmtPct(m.weightedConsensusPercent)}
                </td>
                <td className="muted">{fmtPct(m.consensusPercent)}</td>
                <td>
                  <span className="badge">
                    {m.participants}/{m.totalParticipants}
                  </span>
                </td>
                <td style={{ fontSize: 11 }}>
                  {m.bandMin != null && m.bandMax != null ? (
                    <span className={m.tightBand ? "good" : "muted"}>
                      {fmtNum(m.bandMin, 2)}-{fmtNum(m.bandMax, 2)}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className={m.cooked ? "bad" : m.priceUnavailable ? "muted" : "good"}>
                  {m.priceUnavailable ? "pending" : m.cooked ? "moved" : "fresh"}
                </td>
                <td className={m.ready ? "good" : "muted"}>
                  {m.ready ? (
                    <span style={{ fontWeight: 600 }}>READY</span>
                  ) : (
                    "no"
                  )}
                </td>
              </tr>
            ))}
            {!markets.length ? (
              <tr>
                <td colSpan={9} className="muted">
                  No markets yet. Backend refresh may still be in progress.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

