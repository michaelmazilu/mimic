export type MarketSummary = {
  conditionId: string;
  title?: string | null;
  leadingOutcome?: string | null;
  consensusPercent: number;
  weightedConsensusPercent: number;
  totalParticipants: number;
  participants: number;
  weightedParticipants: number;
  bandMin?: number | null;
  bandMax?: number | null;
  meanEntry?: number | null;
  stddev?: number | null;
  tightBand: boolean;
  midpoint?: number | null;
  cooked: boolean;
  priceUnavailable: boolean;
  ready: boolean;
  confidenceScore: number;
  endDate?: string | null;
  isClosed: boolean;
  isActive: boolean;
  updatedAt?: number | null;
};

export type ClusterSummary = {
  clusterId: string;
  walletCount: number;
  exampleWallets: string[];
  wallets: string[];
};

export type StateResponse = {
  nowTs: number;
  lastRefreshTs: number | null;
  nextRefreshEarliestTs: number | null;
  refreshIntervalSec: number;
  refreshInProgress: boolean;
  walletCount: number;
  tradeCount: number;
  markets: MarketSummary[];
  clusters: ClusterSummary[];
};

export type RefreshResponse = {
  status: string;
  nowTs: number;
  lastRefreshTs: number | null;
  nextRefreshEarliestTs: number | null;
  refreshed: boolean;
  walletUpserts: number;
  tradeInserts: number;
  nWallets: number | null;
  tradesLimit: number | null;
  error?: string | null;
};

export type TradeItem = {
  wallet: string;
  side: string;
  outcome: string;
  price?: number | null;
  size?: number | null;
  timestamp?: number | null;
  txHash?: string | null;
  assetId?: string | null;
};

export type WalletTrades = {
  wallet: string;
  byOutcome: Record<string, TradeItem[]>;
};

export type MarketDetailResponse = {
  conditionId: string;
  title?: string | null;
  wallets: WalletTrades[];
};

export type WalletStats = {
  wallet: string;
  rank?: number | null;
  userName?: string | null;
  leaderboardPnl?: number | null;
  totalTrades: number;
  wonTrades: number;
  lostTrades: number;
  pendingTrades: number;
  winRate: number;
  totalPnl: number;
  avgRoi: number;
  recentTrades7d: number;
  recentWon7d: number;
  recentAccuracy7d: number;
  recentTrades30d: number;
  recentWon30d: number;
  recentAccuracy30d: number;
  streak: number;
  lastTradeTimestamp?: number | null;
  updatedAt?: number | null;
};

export type WalletsListResponse = {
  wallets: WalletStats[];
  totalCount: number;
};

// Backtest Types
export type BacktestConfig = {
  minConfidence: number;
  betSizing: string;
  baseBet: number;
  maxBet: number;
  lookbackDays: number;
  minParticipants: number;
};

export type BacktestTrade = {
  conditionId: string;
  title?: string | null;
  signalTimestamp: number;
  predictedOutcome: string;
  confidenceScore: number;
  betSize: number;
  entryPrice?: number | null;
  actualOutcome?: string | null;
  pnl?: number | null;
  won?: boolean | null;
};

export type EquityCurvePoint = {
  timestamp: number;
  equity: number;
  pnl: number;
  trade_count?: number;
};

export type BacktestRunResponse = {
  runId: string;
  status: string;
  config: BacktestConfig;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  pendingTrades: number;
  winRate: number;
  totalPnl: number;
  totalInvested: number;
  roi: number;
  maxDrawdown: number;
  sharpeRatio: number;
  profitFactor: number;
  trades: BacktestTrade[];
  equityCurve: EquityCurvePoint[];
};

