from typing import Dict, Union, List, Optional
from pydantic import BaseModel


class BacktestingConfig(BaseModel):
    start_time: int = 1735689600  # 2025-01-01 00:00:00
    end_time: int = 1738368000  # 2025-02-01 00:00:00
    backtesting_resolution: str = "1m"
    trade_cost: float = 0.0006
    config: Union[Dict, str]


# New async backtesting models
class BacktestStartRequest(BaseModel):
    """Request model for starting a new backtest run."""
    run_name: str
    start_time: int
    end_time: int
    backtesting_resolution: str = "5m"
    trade_cost: float = 0.0006
    config: Union[Dict, str]


class BacktestStatusResponse(BaseModel):
    """Response model for backtest status."""
    run_id: str
    run_name: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    controller_name: str
    start_time: int
    end_time: int
    backtesting_resolution: str
    trade_cost: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    
    # Results (only present if COMPLETED)
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    net_pnl_quote: Optional[float] = None
    net_pnl_pct: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None


class BacktestLogEntry(BaseModel):
    """Model for a single backtest log entry."""
    timestamp: str
    log_level: str
    log_message: str
    log_category: Optional[str] = None


class BacktestTradeEntry(BaseModel):
    """Model for a single backtest trade."""
    executor_id: str
    trading_pair: str
    side: str
    entry_timestamp: int
    exit_timestamp: Optional[int] = None
    entry_price: float
    exit_price: Optional[float] = None
    amount: float
    net_pnl_quote: float
    net_pnl_pct: float
    cum_fees_quote: float
    close_type: Optional[str] = None
    status: str


class BacktestResultsResponse(BaseModel):
    """Complete results of a backtest run."""
    run_info: BacktestStatusResponse
    trades: List[BacktestTradeEntry]
    logs: List[BacktestLogEntry]
    processed_data: Optional[Dict] = None  # Only if completed


class BacktestListResponse(BaseModel):
    """List of backtest runs."""
    runs: List[BacktestStatusResponse]
    total: int
