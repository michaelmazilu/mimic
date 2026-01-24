from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    nowTs: int
    lastRefreshTs: int | None
    nextRefreshEarliestTs: int | None
    refreshInProgress: bool


class RefreshResponse(BaseModel):
    status: str
    nowTs: int
    lastRefreshTs: int | None
    nextRefreshEarliestTs: int | None
    refreshed: bool
    walletUpserts: int = 0
    tradeInserts: int = 0
    nWallets: int | None = None
    tradesLimit: int | None = None
    error: str | None = None


class MarketSummary(BaseModel):
    conditionId: str
    title: str | None = None
    leadingOutcome: str | None = None
    consensusPercent: float = 0.0
    weightedConsensusPercent: float = 0.0  # Weighted by trader accuracy
    totalParticipants: int = 0
    participants: int = 0
    weightedParticipants: float = 0.0  # Sum of accuracy weights
    bandMin: float | None = None
    bandMax: float | None = None
    meanEntry: float | None = None
    stddev: float | None = None
    tightBand: bool = False
    midpoint: float | None = None
    cooked: bool = False
    priceUnavailable: bool = True
    ready: bool = False
    confidenceScore: float = 0.0  # Combined score factoring in weighted consensus + tightness
    endDate: str | None = None
    isClosed: bool = False
    isActive: bool = True
    updatedAt: int | None = None


class ClusterSummary(BaseModel):
    clusterId: str
    walletCount: int
    exampleWallets: list[str] = Field(default_factory=list)
    wallets: list[str] = Field(default_factory=list)


class StateResponse(BaseModel):
    nowTs: int
    lastRefreshTs: int | None
    nextRefreshEarliestTs: int | None
    refreshIntervalSec: int
    refreshInProgress: bool
    walletCount: int
    tradeCount: int
    markets: list[MarketSummary] = Field(default_factory=list)
    clusters: list[ClusterSummary] = Field(default_factory=list)


class TradeItem(BaseModel):
    wallet: str
    side: str
    outcome: str
    price: float | None = None
    size: float | None = None
    timestamp: int | None = None
    txHash: str | None = None
    assetId: str | None = None
    raw: dict[str, Any] | None = None


class WalletTrades(BaseModel):
    wallet: str
    byOutcome: dict[str, list[TradeItem]] = Field(default_factory=dict)


class MarketDetailResponse(BaseModel):
    conditionId: str
    title: str | None = None
    wallets: list[WalletTrades] = Field(default_factory=list)


class WalletStats(BaseModel):
    wallet: str
    rank: int | None = None
    userName: str | None = None
    leaderboardPnl: float | None = None
    totalTrades: int = 0
    wonTrades: int = 0
    lostTrades: int = 0
    pendingTrades: int = 0
    winRate: float = 0.0
    totalPnl: float = 0.0
    avgRoi: float = 0.0
    recentTrades7d: int = 0
    recentWon7d: int = 0
    recentAccuracy7d: float = 0.0
    recentTrades30d: int = 0
    recentWon30d: int = 0
    recentAccuracy30d: float = 0.0
    streak: int = 0  # positive = win streak, negative = loss streak
    lastTradeTimestamp: int | None = None
    updatedAt: int | None = None


class WalletStatsResponse(BaseModel):
    wallet: str
    stats: WalletStats | None = None
    recentOutcomes: list[dict[str, Any]] = Field(default_factory=list)


class WalletsListResponse(BaseModel):
    wallets: list[WalletStats] = Field(default_factory=list)
    totalCount: int = 0


# ============================================================================
# Backtest Models
# ============================================================================


class BacktestConfigModel(BaseModel):
    """Configuration for a backtest run."""
    minConfidence: float = 0.0
    betSizing: str = "bankroll"  # flat, kelly, scaled, bankroll
    baseBet: float = 100.0
    maxBet: float = 500.0
    startingBankroll: float = 200.0
    betFraction: float = 0.02
    lookbackDays: int = 180
    minParticipants: int = 2


class BacktestTradeModel(BaseModel):
    """A single simulated trade in the backtest."""
    conditionId: str
    title: str | None = None
    signalTimestamp: int
    predictedOutcome: str
    confidenceScore: float
    betSize: float
    entryPrice: float | None = None
    actualOutcome: str | None = None
    pnl: float | None = None
    won: bool | None = None


class BacktestRunResponse(BaseModel):
    """Response for a backtest run."""
    runId: str
    status: str
    config: BacktestConfigModel
    totalTrades: int = 0
    winningTrades: int = 0
    losingTrades: int = 0
    pendingTrades: int = 0
    winRate: float = 0.0
    totalPnl: float = 0.0
    totalInvested: float = 0.0
    roi: float = 0.0
    maxDrawdown: float = 0.0
    sharpeRatio: float = 0.0
    profitFactor: float = 0.0
    trades: list[BacktestTradeModel] = Field(default_factory=list)
    equityCurve: list[dict[str, Any]] = Field(default_factory=list)
