"""
Backtest runner that executes backtesting in a separate process.
Writes status and logs to PostgreSQL for monitoring.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase

from config import settings
from database import AsyncDatabaseManager, BacktestRunRepository


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """Handles the execution of a backtest run and persists results to database."""
    
    def __init__(self, run_id: str, config: Dict):
        self.run_id = run_id
        self.config = config
        self.should_stop = False
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, stopping backtest...")
        self.should_stop = True
    
    async def run(self):
        """Main execution function for the backtest."""
        db_manager = None
        
        try:
            # Initialize database connection
            db_url = os.getenv("DATABASE_URL", settings.database_url)
            db_manager = AsyncDatabaseManager(db_url)
            
            # Update status to RUNNING
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                await repo.update_status(
                    run_id=self.run_id,
                    status="RUNNING",
                    started_at=datetime.now()
                )
                await self._log(repo, "INFO", "Backtest started", "INITIALIZATION")
            
            # Initialize backtesting engine
            backtesting_engine = BacktestingEngineBase()
            
            # Get controller config
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                await self._log(repo, "INFO", f"Loading controller config: {self.config['controller_name']}", "INITIALIZATION")
            
            if isinstance(self.config['config_data'], str):
                # It's a YAML path
                controller_config = backtesting_engine.get_controller_config_instance_from_yml(
                    config_path=self.config['config_data'],
                    controllers_conf_dir_path=settings.app.controllers_path,
                    controllers_module=settings.app.controllers_module
                )
            else:
                # It's a dictionary
                controller_config = backtesting_engine.get_controller_config_instance_from_dict(
                    config_data=json.loads(self.config['config_data']) if isinstance(self.config['config_data'], str) else self.config['config_data'],
                    controllers_module=settings.app.controllers_module
                )
            
            # Run backtesting
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                await self._log(repo, "INFO", "Running backtesting simulation...", "INITIALIZATION")
            
            backtesting_results = await backtesting_engine.run_backtesting(
                controller_config=controller_config,
                trade_cost=self.config['trade_cost'],
                start=int(self.config['start_time']),
                end=int(self.config['end_time']),
                backtesting_resolution=self.config['backtesting_resolution']
            )
            
            # Check if stopped during execution
            if self.should_stop:
                async with db_manager.get_session_context() as session:
                    repo = BacktestRunRepository(session)
                    await repo.update_status(
                        run_id=self.run_id,
                        status="CANCELLED",
                        completed_at=datetime.now()
                    )
                    await self._log(repo, "INFO", "Backtest cancelled by user", "COMPLETED")
                return
            
            # Process results
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                await self._log(repo, "INFO", "Processing backtest results...", "COMPLETED")
                
                # Store trades
                executors = backtesting_results["executors"]
                for executor in executors:
                    await repo.create_trade(self.run_id, executor.to_dict())
                
                await self._log(repo, "INFO", f"Stored {len(executors)} trades", "COMPLETED")
                
                # Update final results
                results = backtesting_results["results"]
                results["sharpe_ratio"] = results["sharpe_ratio"] if results["sharpe_ratio"] is not None else 0
                
                await repo.update_results(
                    run_id=self.run_id,
                    results=results,
                    total_trades=len(executors),
                    status="COMPLETED",
                    completed_at=datetime.now()
                )
                
                await self._log(repo, "INFO", f"Backtest completed successfully. Total trades: {len(executors)}, Net PnL: {results.get('net_pnl', 0):.4%}", "COMPLETED")
            
        except Exception as e:
            logger.exception(f"Error during backtest execution: {e}")
            
            # Update status to FAILED
            if db_manager:
                try:
                    async with db_manager.get_session_context() as session:
                        repo = BacktestRunRepository(session)
                        await repo.update_status(
                            run_id=self.run_id,
                            status="FAILED",
                            error_message=str(e),
                            completed_at=datetime.now()
                        )
                        await self._log(repo, "ERROR", f"Backtest failed: {str(e)}", "ERROR")
                except Exception as db_error:
                    logger.error(f"Failed to update database with error status: {db_error}")
        
        finally:
            if db_manager:
                await db_manager.close()
    
    async def _log(self, repo: BacktestRunRepository, level: str, message: str, category: str = "GENERAL"):
        """Helper to log messages to database."""
        try:
            await repo.bulk_create_logs(
                run_id=self.run_id,
                logs=[{
                    "log_level": level,
                    "log_message": message,
                    "log_category": category
                }]
            )
        except Exception as e:
            logger.error(f"Failed to write log to database: {e}")


def main():
    """Entry point for the backtest runner process."""
    if len(sys.argv) < 3:
        logger.error("Usage: python backtest_runner.py <run_id> <config_json>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    config_json = sys.argv[2]
    
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid config JSON: {e}")
        sys.exit(1)
    
    runner = BacktestRunner(run_id, config)
    
    # Run the backtest
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()

