from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import BacktestRun, BacktestLog, BacktestTrade


class BacktestRunRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_backtest_run(
        self,
        run_id: str,
        run_name: str,
        controller_name: str,
        config_data: str,
        start_time: int,
        end_time: int,
        backtesting_resolution: str,
        trade_cost: float,
        status: str = "PENDING"
    ) -> BacktestRun:
        """Create a new backtest run record."""
        backtest_run = BacktestRun(
            run_id=run_id,
            run_name=run_name,
            controller_name=controller_name,
            config_data=config_data,
            start_time=start_time,
            end_time=end_time,
            backtesting_resolution=backtesting_resolution,
            trade_cost=trade_cost,
            status=status
        )
        
        self.session.add(backtest_run)
        await self.session.flush()
        return backtest_run

    async def get_backtest_run(self, run_id: str) -> Optional[BacktestRun]:
        """Get backtest run by run_id."""
        result = await self.session.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        run_id: str,
        status: str,
        error_message: str = None,
        started_at: datetime = None,
        completed_at: datetime = None
    ):
        """Update backtest run status."""
        result = await self.session.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        )
        backtest_run = result.scalar_one_or_none()
        
        if backtest_run:
            backtest_run.status = status
            if error_message:
                backtest_run.error_message = error_message
            if started_at:
                backtest_run.started_at = started_at
            if completed_at:
                backtest_run.completed_at = completed_at
            
            await self.session.flush()

    async def update_results(
        self,
        run_id: str,
        results: dict,
        total_trades: int,
        status: str = "COMPLETED",
        completed_at: datetime = None
    ):
        """Update backtest run with final results."""
        result = await self.session.execute(
            select(BacktestRun).where(BacktestRun.run_id == run_id)
        )
        backtest_run = result.scalar_one_or_none()
        
        if backtest_run:
            backtest_run.status = status
            backtest_run.total_trades = total_trades
            backtest_run.net_pnl_quote = Decimal(str(results.get("net_pnl_quote", 0)))
            backtest_run.net_pnl_pct = Decimal(str(results.get("net_pnl", 0)))
            backtest_run.win_rate = Decimal(str(results.get("accuracy", 0)))
            backtest_run.max_drawdown = Decimal(str(results.get("max_drawdown_usd", 0)))
            backtest_run.sharpe_ratio = Decimal(str(results.get("sharpe_ratio", 0)))
            backtest_run.completed_at = completed_at or datetime.now()
            
            await self.session.flush()

    async def bulk_create_logs(self, run_id: str, logs: List[dict]):
        """Bulk create log entries for a backtest run."""
        # Get backtest_run database ID
        result = await self.session.execute(
            select(BacktestRun.id).where(BacktestRun.run_id == run_id)
        )
        backtest_run_id = result.scalar_one_or_none()
        
        if not backtest_run_id:
            return
        
        # Bulk insert logs
        log_objects = [
            BacktestLog(
                backtest_run_id=backtest_run_id,
                log_level=log["log_level"],
                log_message=log["log_message"],
                log_category=log.get("log_category", "GENERAL")
            )
            for log in logs
        ]
        
        self.session.add_all(log_objects)
        await self.session.flush()

    async def get_logs(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
        level: Optional[str] = None
    ) -> Tuple[List[BacktestLog], int]:
        """Get logs for a backtest run with pagination and filtering."""
        # Get backtest_run database ID
        result = await self.session.execute(
            select(BacktestRun.id).where(BacktestRun.run_id == run_id)
        )
        backtest_run_id = result.scalar_one_or_none()
        
        if not backtest_run_id:
            return [], 0
        
        # Build query
        query = select(BacktestLog).where(BacktestLog.backtest_run_id == backtest_run_id)
        
        if level:
            query = query.where(BacktestLog.log_level == level)
        
        # Get total count
        count_result = await self.session.execute(
            select(func.count()).select_from(query.subquery())
        )
        total = count_result.scalar()
        
        # Get paginated results
        query = query.order_by(desc(BacktestLog.timestamp)).limit(limit).offset(offset)
        result = await self.session.execute(query)
        logs = result.scalars().all()
        
        return logs, total

    async def create_trade(self, run_id: str, executor_info: dict):
        """Create a trade record from executor info."""
        # Get backtest_run database ID
        result = await self.session.execute(
            select(BacktestRun.id).where(BacktestRun.run_id == run_id)
        )
        backtest_run_id = result.scalar_one_or_none()
        
        if not backtest_run_id:
            return
        
        # Extract trade information from executor_info
        config = executor_info.get("config", {})
        
        trade = BacktestTrade(
            backtest_run_id=backtest_run_id,
            executor_id=executor_info["id"],
            trading_pair=config["trading_pair"],
            side=config["side"],
            entry_timestamp=int(executor_info["timestamp"]),
            exit_timestamp=int(executor_info["close_timestamp"]) if executor_info.get("close_timestamp") else None,
            entry_price=Decimal(str(config["entry_price"])),
            exit_price=None,  # Can be calculated from net_pnl if needed
            amount=Decimal(str(config["amount"])),
            net_pnl_quote=Decimal(str(executor_info["net_pnl_quote"])),
            net_pnl_pct=Decimal(str(executor_info["net_pnl_pct"])),
            cum_fees_quote=Decimal(str(executor_info["cum_fees_quote"])),
            close_type=executor_info.get("close_type") if isinstance(executor_info.get("close_type"), str) else None,
            status="CLOSED" if executor_info.get("close_timestamp") else "ACTIVE"
        )
        
        self.session.add(trade)
        await self.session.flush()

    async def get_trades(self, run_id: str) -> List[BacktestTrade]:
        """Get all trades for a backtest run."""
        # Get backtest_run database ID
        result = await self.session.execute(
            select(BacktestRun.id).where(BacktestRun.run_id == run_id)
        )
        backtest_run_id = result.scalar_one_or_none()
        
        if not backtest_run_id:
            return []
        
        result = await self.session.execute(
            select(BacktestTrade)
            .where(BacktestTrade.backtest_run_id == backtest_run_id)
            .order_by(BacktestTrade.entry_timestamp)
        )
        
        return result.scalars().all()

    async def list_backtest_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[BacktestRun]:
        """List backtest runs with pagination and filtering."""
        query = select(BacktestRun)
        
        if status:
            query = query.where(BacktestRun.status == status)
        
        query = query.order_by(desc(BacktestRun.created_at)).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return result.scalars().all()

