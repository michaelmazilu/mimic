"use client";

import React, { useState } from "react";

import type { BacktestRunResponse, BacktestConfig } from "@/app/lib/types";
import { runBacktest } from "@/app/lib/api";

function fmtPct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtUsd(x: number): string {
  if (!Number.isFinite(x)) return "—";
  const sign = x >= 0 ? "+" : "";
  return `${sign}$${x.toFixed(2)}`;
}

function fmtTs(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  } catch {
    return "—";
  }
}

const PAGE_SIZE = 10;

type Props = {
  initialResult?: BacktestRunResponse | null;
};

export function BacktestPanel({ initialResult }: Props) {
  const [result, setResult] = useState<BacktestRunResponse | null>(initialResult || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visibleTrades, setVisibleTrades] = useState(PAGE_SIZE);

  // Config state
  const [config, setConfig] = useState<BacktestConfig>({
    minConfidence: 0.0,
    betSizing: "bankroll",
    baseBet: 100,
    maxBet: 500,
    startingBankroll: 200,
    betFraction: 0.02,
    lookbackDays: 180,
    minParticipants: 2,
  });

  const handleRunBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runBacktest(undefined, config);
      setResult(res);
      setVisibleTrades(PAGE_SIZE);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run backtest");
    } finally {
      setLoading(false);
    }
  };

  const trades = result?.trades || [];
  const visibleTradeList = trades.slice(0, visibleTrades);
  const hasMoreTrades = visibleTrades < trades.length;

  return (
    <div className="section">
      <div className="title">Backtest</div>
      <div className="subtitle">Simulate strategy performance on historical data</div>

      {/* Config Controls */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 16 }}>
          <div>
            <label className="label">Min Confidence</label>
            <select
              className="input"
              value={config.minConfidence}
              onChange={(e) => setConfig({ ...config, minConfidence: parseFloat(e.target.value) })}
            >
              <option value={0}>0%</option>
              <option value={0.2}>20%</option>
              <option value={0.60}>60%</option>
              <option value={0.70}>70%</option>
              <option value={0.80}>80%</option>
              <option value={0.90}>90%</option>
            </select>
          </div>
          <div>
            <label className="label">Bet Sizing</label>
            <select
              className="input"
              value={config.betSizing}
              onChange={(e) => setConfig({ ...config, betSizing: e.target.value })}
            >
              <option value="flat">Flat ($100)</option>
              <option value="scaled">Scaled by Confidence</option>
              <option value="kelly">Kelly Criterion</option>
              <option value="bankroll">Bankroll Fraction</option>
            </select>
          </div>
          <div>
            <label className="label">Starting Bankroll</label>
            <input
              className="input"
              type="number"
              min={1}
              value={config.startingBankroll}
              onChange={(e) =>
                setConfig({ ...config, startingBankroll: parseFloat(e.target.value) || 0 })
              }
            />
          </div>
          <div>
            <label className="label">Bet Fraction</label>
            <input
              className="input"
              type="number"
              min={0}
              step={0.005}
              value={config.betFraction}
              onChange={(e) =>
                setConfig({ ...config, betFraction: parseFloat(e.target.value) || 0 })
              }
            />
          </div>
          <div>
            <label className="label">Lookback Days</label>
            <select
              className="input"
              value={config.lookbackDays}
              onChange={(e) => setConfig({ ...config, lookbackDays: parseInt(e.target.value) })}
            >
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
              <option value={180}>180 days</option>
              <option value={365}>1 year</option>
            </select>
          </div>
          <div>
            <label className="label">Min Participants</label>
            <input
              className="input"
              type="number"
              min={1}
              max={10}
              value={config.minParticipants}
              onChange={(e) => setConfig({ ...config, minParticipants: parseInt(e.target.value) || 2 })}
            />
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <button className="btn" onClick={handleRunBacktest} disabled={loading}>
            {loading ? "Running..." : "Run Backtest"}
          </button>
        </div>
        {error && <div style={{ color: "#c00", marginTop: 8 }}>{error}</div>}
      </div>

      {/* Results Summary */}
      {result && (
        <>
          <div className="card" style={{ marginBottom: 24 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 24 }}>
              <div className="stat">
                <div className="statLabel">Total P&L</div>
                <div className="statValue" style={{ color: result.totalPnl >= 0 ? "#080" : "#c00" }}>
                  {fmtUsd(result.totalPnl)}
                </div>
              </div>
              <div className="stat">
                <div className="statLabel">Win Rate</div>
                <div className="statValue">{fmtPct(result.winRate)}</div>
              </div>
              <div className="stat">
                <div className="statLabel">ROI</div>
                <div className="statValue">{fmtPct(result.roi)}</div>
              </div>
              <div className="stat">
                <div className="statLabel">Total Trades</div>
                <div className="statValue">{result.totalTrades}</div>
              </div>
              <div className="stat">
                <div className="statLabel">Record</div>
                <div className="statValue">
                  {result.winningTrades}W / {result.losingTrades}L
                </div>
              </div>
              <div className="stat">
                <div className="statLabel">Max Drawdown</div>
                <div className="statValue">{fmtPct(result.maxDrawdown)}</div>
              </div>
              <div className="stat">
                <div className="statLabel">Sharpe Ratio</div>
                <div className="statValue">{result.sharpeRatio.toFixed(2)}</div>
              </div>
              <div className="stat">
                <div className="statLabel">Invested</div>
                <div className="statValue">${result.totalInvested.toFixed(0)}</div>
              </div>
            </div>
          </div>

          {/* Equity Curve - Simple Text Version */}
          {result.equityCurve.length > 0 && (
            <div className="card" style={{ marginBottom: 24 }}>
              <div className="subtitle" style={{ marginBottom: 12 }}>Equity Curve</div>
              <div style={{ fontFamily: "monospace", fontSize: 12 }}>
                {result.equityCurve.slice(-20).map((point, i) => {
                  const barLength = Math.min(50, Math.max(0, Math.round((point.equity + 500) / 20)));
                  const bar = "█".repeat(barLength);
                  return (
                    <div key={i} style={{ display: "flex", gap: 8 }}>
                      <span style={{ width: 60 }}>{fmtTs(point.timestamp)}</span>
                      <span style={{ width: 80, textAlign: "right", color: point.equity >= 0 ? "#080" : "#c00" }}>
                        {fmtUsd(point.equity)}
                      </span>
                      <span style={{ color: point.equity >= 0 ? "#080" : "#c00" }}>{bar}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Trade Log */}
          <div className="subtitle" style={{ marginBottom: 12 }}>
            Trade Log ({trades.length} trades)
          </div>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Market</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Bet</th>
                  <th>Result</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {visibleTradeList.map((t, i) => (
                  <tr key={`${t.conditionId}-${i}`}>
                    <td className="secondary" style={{ fontSize: 12 }}>
                      {fmtTs(t.signalTimestamp)}
                    </td>
                    <td style={{ maxWidth: 200 }}>
                      {t.title ? (
                        <span style={{ fontSize: 13 }}>
                          {t.title.length > 40 ? t.title.slice(0, 40) + "..." : t.title}
                        </span>
                      ) : (
                        <code style={{ fontSize: 11 }}>{t.conditionId.slice(0, 16)}...</code>
                      )}
                    </td>
                    <td>
                      <span style={{ fontWeight: 500 }}>{t.predictedOutcome}</span>
                    </td>
                    <td>{fmtPct(t.confidenceScore)}</td>
                    <td>${t.betSize.toFixed(0)}</td>
                    <td>
                      {t.won === true && <span style={{ color: "#080", fontWeight: 600 }}>WIN</span>}
                      {t.won === false && <span style={{ color: "#c00", fontWeight: 600 }}>LOSS</span>}
                      {t.won === null && <span className="secondary">—</span>}
                    </td>
                    <td style={{ color: (t.pnl || 0) >= 0 ? "#080" : "#c00", fontWeight: 500 }}>
                      {t.pnl != null ? fmtUsd(t.pnl) : "—"}
                    </td>
                  </tr>
                ))}
                {trades.length === 0 && (
                  <tr>
                    <td colSpan={7} className="secondary" style={{ textAlign: "center", padding: 32 }}>
                      No trades found in backtest period
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {hasMoreTrades && (
            <div style={{ marginTop: 16, textAlign: "center" }}>
              <button className="btn" onClick={() => setVisibleTrades((v) => v + PAGE_SIZE)}>
                Load More ({trades.length - visibleTrades} remaining)
              </button>
            </div>
          )}
        </>
      )}

      {!result && !loading && (
        <div className="secondary" style={{ textAlign: "center", padding: 32 }}>
          Configure settings and click "Run Backtest" to simulate strategy performance
        </div>
      )}
    </div>
  );
}
