export type MarketSummary = {
  conditionId: string;
  title?: string | null;
  leadingOutcome?: string | null;
  consensusPercent: number;
  totalParticipants: number;
  participants: number;
  bandMin?: number | null;
  bandMax?: number | null;
  meanEntry?: number | null;
  stddev?: number | null;
  tightBand: boolean;
  midpoint?: number | null;
  cooked: boolean;
  priceUnavailable: boolean;
  ready: boolean;
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

