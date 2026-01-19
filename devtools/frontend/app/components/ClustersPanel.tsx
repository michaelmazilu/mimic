import React from "react";

import type { ClusterSummary } from "@/app/lib/types";

export function ClustersPanel({ clusters }: { clusters: ClusterSummary[] }) {
  return (
    <div className="glass card" style={{ flex: "1 1 420px", minWidth: 280 }}>
      <div className="title">Copycat clusters</div>
      <div className="muted" style={{ fontSize: 13, marginBottom: 12 }}>
        Jaccard ≥ 0.80 on last-50-trade signatures; clusters of size ≥ 3.
      </div>

      {!clusters.length ? (
        <div className="muted">No clusters detected.</div>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {clusters.map((c) => (
            <div key={c.clusterId} className="glass card" style={{ background: "var(--glass2)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    cluster id
                  </div>
                  <code>{c.clusterId}</code>
                </div>
                <div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    wallet count
                  </div>
                  <span className="badge">{c.walletCount}</span>
                </div>
              </div>
              <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
                example wallets
              </div>
              <div style={{ display: "grid", gap: 6, marginTop: 6 }}>
                {c.exampleWallets.map((w) => (
                  <code key={w} style={{ fontSize: 12 }}>
                    {w}
                  </code>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

