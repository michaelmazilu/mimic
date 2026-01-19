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

export function MarketsTable({ markets }: { markets: MarketSummary[] }) {
  return (
    <div className="glass card" style={{ flex: "1 1 820px", minWidth: 340 }}>
      <div className="title">Markets (sorted by consensus)</div>
      <div className="tableWrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr>
              <th>conditionId</th>
              <th>leadingOutcome</th>
              <th>consensusPercent</th>
              <th>participants</th>
              <th>bandMin</th>
              <th>bandMax</th>
              <th>stddev</th>
              <th>cooked</th>
              <th>ready</th>
            </tr>
          </thead>
          <tbody>
            {markets.map((m) => (
              <tr key={m.conditionId}>
                <td style={{ maxWidth: 360 }}>
                  <Link href={`/market/${encodeURIComponent(m.conditionId)}`}>
                    <code>{m.conditionId}</code>
                  </Link>
                  {m.title ? (
                    <div className="muted" style={{ marginTop: 6, fontSize: 12, lineHeight: 1.25 }}>
                      {m.title}
                    </div>
                  ) : null}
                </td>
                <td>{m.leadingOutcome ?? "—"}</td>
                <td>{fmtPct(m.consensusPercent)}</td>
                <td>
                  <span className="badge">
                    {m.participants}/{m.totalParticipants}
                  </span>
                </td>
                <td>{fmtNum(m.bandMin)}</td>
                <td>{fmtNum(m.bandMax)}</td>
                <td>{fmtNum(m.stddev)}</td>
                <td className={m.cooked ? "bad" : m.priceUnavailable ? "muted" : "good"}>
                  {m.priceUnavailable ? "n/a" : m.cooked ? "true" : "false"}
                </td>
                <td className={m.ready ? "good" : "muted"}>{m.ready ? "true" : "false"}</td>
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

