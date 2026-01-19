import React from "react";

import type { ClusterSummary } from "@/app/lib/types";

export function ClustersPanel({ clusters }: { clusters: ClusterSummary[] }) {
  return (
    <div className="card" style={{ flex: "1 1 400px", minWidth: 280 }}>
      <div className="title">Clusters</div>
      <div className="subtitle">Wallets trading in sync (Jaccard similarity 80%+)</div>

      {!clusters.length ? (
        <div className="secondary">No clusters detected</div>
      ) : (
        <div style={{ display: "grid", gap: 16 }}>
          {clusters.map((c) => (
            <div
              key={c.clusterId}
              style={{
                padding: 16,
                border: "1px solid var(--border)",
                borderRadius: 8,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                <div>
                  <div className="label">Cluster</div>
                  <code>{c.clusterId}</code>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div className="label">Wallets</div>
                  <span style={{ fontWeight: 600 }}>{c.walletCount}</span>
                </div>
              </div>
              <div>
                <div className="label">Sample wallets</div>
                <div style={{ display: "grid", gap: 4 }}>
                  {c.exampleWallets.slice(0, 3).map((w) => (
                    <code key={w} className="secondary" style={{ fontSize: 11 }}>
                      {w.slice(0, 24)}...
                    </code>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
