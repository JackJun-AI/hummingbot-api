import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Optional

import psutil
from fastapi import APIRouter, HTTPException, Query

from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase

from config import settings
from database import AsyncDatabaseManager, BacktestRunRepository
from models.backtesting import (
    BacktestingConfig,
    BacktestStartRequest,
    BacktestStatusResponse,
    BacktestResultsResponse,
    BacktestListResponse,
    BacktestLogEntry,
    BacktestTradeEntry
)

router = APIRouter(tags=["Backtesting"], prefix="/backtesting")
candles_factory = CandlesFactory()
backtesting_engine = BacktestingEngineBase()

# Store process PIDs for running backtests
_running_processes = {}


@router.post("/run-backtesting")
async def run_backtesting(backtesting_config: BacktestingConfig):
    """
    Run a backtesting simulation with the provided configuration.
    
    Args:
        backtesting_config: Configuration for the backtesting including start/end time,
                          resolution, trade cost, and controller config
                          
    Returns:
        Dictionary containing executors, processed data, and results from the backtest
        
    Raises:
        Returns error dictionary if backtesting fails
    """
    try:
        if isinstance(backtesting_config.config, str):
            controller_config = backtesting_engine.get_controller_config_instance_from_yml(
                config_path=backtesting_config.config,
                controllers_conf_dir_path=settings.app.controllers_path,
                controllers_module=settings.app.controllers_module
            )
        else:
            controller_config = backtesting_engine.get_controller_config_instance_from_dict(
                config_data=backtesting_config.config,
                controllers_module=settings.app.controllers_module
            )
        backtesting_results = await backtesting_engine.run_backtesting(
            controller_config=controller_config, trade_cost=backtesting_config.trade_cost,
            start=int(backtesting_config.start_time), end=int(backtesting_config.end_time),
            backtesting_resolution=backtesting_config.backtesting_resolution)
        processed_data = backtesting_results["processed_data"]["features"].fillna(0)
        executors_info = [e.to_dict() for e in backtesting_results["executors"]]
        backtesting_results["processed_data"] = processed_data.to_dict()
        results = backtesting_results["results"]
        results["sharpe_ratio"] = results["sharpe_ratio"] if results["sharpe_ratio"] is not None else 0
        return {
            "executors": executors_info,
            "processed_data": backtesting_results["processed_data"],
            "results": backtesting_results["results"],
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# NEW: Async Backtesting Endpoints
# ============================================================================

@router.post("/start", response_model=dict)
async def start_backtest(request: BacktestStartRequest):
    """
    Start a new backtest run in a separate process.
    
    Returns the run_id to track progress.
    """
    db_manager = AsyncDatabaseManager(settings.database_url)
    
    try:
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        # Extract controller name from config
        if isinstance(request.config, str):
            # It's a YAML path, extract controller name from path
            controller_name = os.path.basename(request.config).replace(".yml", "")
        else:
            # It's a dict, extract from controller_name or controller_type
            controller_name = request.config.get("controller_name") or request.config.get("controller_type", "unknown")
        
        # Serialize config
        if isinstance(request.config, str):
            config_data_str = request.config  # Keep as path
        else:
            config_data_str = json.dumps(request.config)
        
        # Create backtest run record in database
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.create_backtest_run(
                run_id=run_id,
                run_name=request.run_name,
                controller_name=controller_name,
                config_data=config_data_str,
                start_time=request.start_time,
                end_time=request.end_time,
                backtesting_resolution=request.backtesting_resolution,
                trade_cost=request.trade_cost,
                status="PENDING"
            )
        
        # Prepare config for subprocess
        process_config = {
            "controller_name": controller_name,
            "config_data": request.config if isinstance(request.config, str) else request.config,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "backtesting_resolution": request.backtesting_resolution,
            "trade_cost": request.trade_cost
        }
        
        # Start backtest in separate process
        runner_path = os.path.join(os.path.dirname(__file__), "..", "services", "backtest_runner.py")
        process = subprocess.Popen(
            [sys.executable, runner_path, run_id, json.dumps(process_config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "DATABASE_URL": settings.database_url}
        )
        
        # Store process information
        _running_processes[run_id] = {
            "pid": process.pid,
            "process": process
        }
        
        return {
            "run_id": run_id,
            "status": "PENDING",
            "message": f"Backtest started with run_id: {run_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")
    finally:
        await db_manager.close()


@router.get("/status/{run_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(run_id: str):
    """
    Get the current status of a backtest run.
    """
    db_manager = AsyncDatabaseManager(settings.database_url)
    
    try:
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            backtest_run = await repo.get_backtest_run(run_id)
            
            if not backtest_run:
                raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
            
            # Convert to response model
            return BacktestStatusResponse(
                run_id=backtest_run.run_id,
                run_name=backtest_run.run_name,
                status=backtest_run.status,
                controller_name=backtest_run.controller_name,
                start_time=backtest_run.start_time,
                end_time=backtest_run.end_time,
                backtesting_resolution=backtest_run.backtesting_resolution,
                trade_cost=float(backtest_run.trade_cost),
                created_at=backtest_run.created_at.isoformat(),
                started_at=backtest_run.started_at.isoformat() if backtest_run.started_at else None,
                completed_at=backtest_run.completed_at.isoformat() if backtest_run.completed_at else None,
                error_message=backtest_run.error_message,
                total_trades=backtest_run.total_trades,
                win_rate=float(backtest_run.win_rate) if backtest_run.win_rate else None,
                net_pnl_quote=float(backtest_run.net_pnl_quote) if backtest_run.net_pnl_quote else None,
                net_pnl_pct=float(backtest_run.net_pnl_pct) if backtest_run.net_pnl_pct else None,
                max_drawdown=float(backtest_run.max_drawdown) if backtest_run.max_drawdown else None,
                sharpe_ratio=float(backtest_run.sharpe_ratio) if backtest_run.sharpe_ratio else None
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")
    finally:
        await db_manager.close()


@router.get("/results/{run_id}", response_model=BacktestResultsResponse)
async def get_backtest_results(
    run_id: str,
    include_logs: bool = Query(True, description="Include log entries in response"),
    log_limit: int = Query(100, description="Maximum number of logs to return")
):
    """
    Get complete results of a backtest run, including trades and logs.
    """
    db_manager = AsyncDatabaseManager(settings.database_url)
    
    try:
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            
            # Get backtest run info
            backtest_run = await repo.get_backtest_run(run_id)
            if not backtest_run:
                raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
            
            # Get trades
            trades = await repo.get_trades(run_id)
            trade_entries = [
                BacktestTradeEntry(
                    executor_id=trade.executor_id,
                    trading_pair=trade.trading_pair,
                    side=trade.side,
                    entry_timestamp=trade.entry_timestamp,
                    exit_timestamp=trade.exit_timestamp,
                    entry_price=float(trade.entry_price),
                    exit_price=float(trade.exit_price) if trade.exit_price else None,
                    amount=float(trade.amount),
                    net_pnl_quote=float(trade.net_pnl_quote),
                    net_pnl_pct=float(trade.net_pnl_pct),
                    cum_fees_quote=float(trade.cum_fees_quote),
                    close_type=trade.close_type,
                    status=trade.status
                )
                for trade in trades
            ]
            
            # Get logs if requested
            log_entries = []
            if include_logs:
                logs, _ = await repo.get_logs(run_id, limit=log_limit)
                log_entries = [
                    BacktestLogEntry(
                        timestamp=log.timestamp.isoformat(),
                        log_level=log.log_level,
                        log_message=log.log_message,
                        log_category=log.log_category
                    )
                    for log in logs
                ]
            
            # Build response
            run_info = BacktestStatusResponse(
                run_id=backtest_run.run_id,
                run_name=backtest_run.run_name,
                status=backtest_run.status,
                controller_name=backtest_run.controller_name,
                start_time=backtest_run.start_time,
                end_time=backtest_run.end_time,
                backtesting_resolution=backtest_run.backtesting_resolution,
                trade_cost=float(backtest_run.trade_cost),
                created_at=backtest_run.created_at.isoformat(),
                started_at=backtest_run.started_at.isoformat() if backtest_run.started_at else None,
                completed_at=backtest_run.completed_at.isoformat() if backtest_run.completed_at else None,
                error_message=backtest_run.error_message,
                total_trades=backtest_run.total_trades,
                win_rate=float(backtest_run.win_rate) if backtest_run.win_rate else None,
                net_pnl_quote=float(backtest_run.net_pnl_quote) if backtest_run.net_pnl_quote else None,
                net_pnl_pct=float(backtest_run.net_pnl_pct) if backtest_run.net_pnl_pct else None,
                max_drawdown=float(backtest_run.max_drawdown) if backtest_run.max_drawdown else None,
                sharpe_ratio=float(backtest_run.sharpe_ratio) if backtest_run.sharpe_ratio else None
            )
            
            return BacktestResultsResponse(
                run_info=run_info,
                trades=trade_entries,
                logs=log_entries,
                processed_data=None  # Can add processed_data if needed
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest results: {str(e)}")
    finally:
        await db_manager.close()


@router.post("/stop/{run_id}")
async def stop_backtest(run_id: str):
    """
    Stop a running backtest by sending SIGTERM to the process.
    """
    if run_id not in _running_processes:
        raise HTTPException(status_code=404, detail=f"No running process found for run_id: {run_id}")
    
    try:
        process_info = _running_processes[run_id]
        pid = process_info["pid"]
        
        # Check if process is still running
        if psutil.pid_exists(pid):
            # Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            
            return {
                "run_id": run_id,
                "message": f"Stop signal sent to process {pid}",
                "status": "STOPPING"
            }
        else:
            # Process already finished
            del _running_processes[run_id]
            return {
                "run_id": run_id,
                "message": "Process already finished",
                "status": "FINISHED"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop backtest: {str(e)}")


@router.get("/list", response_model=BacktestListResponse)
async def list_backtests(
    limit: int = Query(20, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List all backtest runs with pagination and filtering.
    """
    db_manager = AsyncDatabaseManager(settings.database_url)
    
    try:
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            runs = await repo.list_backtest_runs(limit=limit, offset=offset, status=status)
            
            run_responses = [
                BacktestStatusResponse(
                    run_id=run.run_id,
                    run_name=run.run_name,
                    status=run.status,
                    controller_name=run.controller_name,
                    start_time=run.start_time,
                    end_time=run.end_time,
                    backtesting_resolution=run.backtesting_resolution,
                    trade_cost=float(run.trade_cost),
                    created_at=run.created_at.isoformat(),
                    started_at=run.started_at.isoformat() if run.started_at else None,
                    completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    error_message=run.error_message,
                    total_trades=run.total_trades,
                    win_rate=float(run.win_rate) if run.win_rate else None,
                    net_pnl_quote=float(run.net_pnl_quote) if run.net_pnl_quote else None,
                    net_pnl_pct=float(run.net_pnl_pct) if run.net_pnl_pct else None,
                    max_drawdown=float(run.max_drawdown) if run.max_drawdown else None,
                    sharpe_ratio=float(run.sharpe_ratio) if run.sharpe_ratio else None
                )
                for run in runs
            ]
            
            return BacktestListResponse(
                runs=run_responses,
                total=len(run_responses)
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")
    finally:
        await db_manager.close()

