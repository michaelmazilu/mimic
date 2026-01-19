"use client";

import Link from "next/link";
import React, { useEffect, useState } from "react";

import { DEFAULT_BACKEND_URL, getMarketDetail } from "@/app/lib/api";
import type { MarketDetailResponse, TradeItem } from "@/app/lib/types";

function fmtTs(ts: number | null | undefined): string {
  if (!ts) return "—";
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

function fmtNum(x: number | null | undefined, digits: number = 3): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

function TradesTable({ trades }: { trades: TradeItem[] }) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>time</th>
            <th>side</th>
            <th>price</th>
            <th>size</th>
            <th>tx</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, idx) => (
            <tr key={`${t.txHash ?? "no-tx"}-${t.timestamp ?? 0}-${idx}`}>
              <td className="muted">{fmtTs(t.timestamp)}</td>
              <td>{t.side}</td>
              <td>{fmtNum(t.price)}</td>
              <td>{fmtNum(t.size, 2)}</td>
              <td style={{ maxWidth: 340 }}>
                <code>{t.txHash ?? "—"}</code>
              </td>
            </tr>
          ))}
          {!trades.length ? (
            <tr>
              <td colSpan={5} className="muted">
                No trades.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

export default function MarketDetailPage({ params }: { params: { conditionId: string } }) {
  const conditionId = decodeURIComponent(params.conditionId);
  const backendUrl = DEFAULT_BACKEND_URL;
  const [detail, setDetail] = useState<MarketDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const d = await getMarketDetail(conditionId, backendUrl);
        if (cancelled) return;
        setDetail(d);
        setError(null);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [backendUrl, conditionId]);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="glass card">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div className="muted" style={{ fontSize: 12 }}>
              conditionId
            </div>
            <code>{conditionId}</code>
            {detail?.title ? (
              <div className="muted" style={{ marginTop: 8 }}>
                {detail.title}
              </div>
            ) : null}
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <Link className="btn" href="/">
              ← Back
            </Link>
            <a className="btn" href={`${backendUrl}/market/${encodeURIComponent(conditionId)}`} target="_blank">
              Raw JSON
            </a>
          </div>
        </div>

        {error ? (
          <div style={{ marginTop: 12 }} className="bad">
            <div style={{ fontWeight: 650 }}>Error</div>
            <div className="muted" style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {error}
            </div>
          </div>
        ) : null}
      </div>

      {!detail ? (
        <div className="glass card" style={{ padding: 18 }}>
          <div className="muted">Loading…</div>
        </div>
      ) : (
        <div style={{ display: "grid", gap: 14 }}>
          {detail.wallets.map((w) => (
            <div key={w.wallet} className="glass card">
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    wallet
                  </div>
                  <code>{w.wallet}</code>
                </div>
                <div className="badge">{Object.keys(w.byOutcome).length} outcomes</div>
              </div>
              <div style={{ display: "grid", gap: 14, marginTop: 12 }}>
                {Object.entries(w.byOutcome).map(([outcome, trades]) => (
                  <div key={outcome}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 650 }}>{outcome}</div>
                      <div className="badge">{trades.length} trades</div>
                    </div>
                    <div style={{ marginTop: 10 }}>
                      <TradesTable trades={trades} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {!detail.wallets.length ? (
            <div className="glass card">
              <div className="muted">No wallets returned.</div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

