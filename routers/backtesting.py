import asyncio
import json
import math
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Any, Dict
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query

from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
# ✅ 引入增强版回测引擎（用于异步 AI Agent）
from services.backtesting_engine_async import BacktestingEngineAsync

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

# 🔑 两个引擎实例：
# 1. 原版引擎：用于同步 endpoint /run-backtesting（兼容旧 Controller）
backtesting_engine = BacktestingEngineBase()

# 2. 增强版引擎：用于异步 endpoints /backtesting/start（支持异步 AI Agent）
backtesting_engine_async = BacktestingEngineAsync()

# Store running backtest tasks
_running_tasks = {}

# 🔑 使用 contextvars 实现异步任务隔离的上下文（支持多任务并发）
backtest_run_id_var: ContextVar[Optional[str]] = ContextVar('backtest_run_id', default=None)
backtest_db_url_var: ContextVar[Optional[str]] = ContextVar('backtest_db_url', default=None)


def sanitize_float_value(value: Any, default: float = 0.0) -> Any:
    """
    Replace inf, -inf, and nan float values with a default value.
    
    Args:
        value: The value to sanitize
        default: The default value to use for inf/-inf/nan (default: 0.0)
        
    Returns:
        Sanitized value
    """
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return default
    return value


def sanitize_dict(data: Dict[str, Any], default: float = 0.0) -> Dict[str, Any]:
    """
    Recursively sanitize all float values in a dictionary.
    
    Args:
        data: Dictionary to sanitize
        default: The default value to use for inf/-inf/nan (default: 0.0)
        
    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, default)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_dict(item, default) if isinstance(item, dict) 
                else sanitize_float_value(item, default)
                for item in value
            ]
        else:
            sanitized[key] = sanitize_float_value(value, default)
    return sanitized


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
        
        # Process data and replace NaN with 0
        processed_data = backtesting_results["processed_data"]["features"].fillna(0)
        processed_data_dict = processed_data.to_dict()
        
        # Sanitize executors to remove inf/-inf/nan values
        executors_info = [e.to_dict() for e in backtesting_results["executors"]]
        executors_info = [sanitize_dict(executor) for executor in executors_info]
        
        # Sanitize results to remove inf/-inf/nan values
        results = sanitize_dict(backtesting_results["results"])
        
        # Sanitize processed data to remove any remaining inf/-inf/nan values
        processed_data_dict = sanitize_dict(processed_data_dict)
        
        return {
            "executors": executors_info,
            "processed_data": processed_data_dict,
            "results": results,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# NEW: Async Backtesting Endpoints
# ============================================================================

async def run_backtest_task(run_id: str, request: BacktestStartRequest):
    """
    Background task to run the backtest.
    This runs in the same process as the API, but asynchronously.
    """
    # 🔑 设置当前任务的上下文变量（线程安全，支持多任务并发）
    backtest_run_id_var.set(run_id)
    backtest_db_url_var.set(settings.database.url)
    
    db_manager = AsyncDatabaseManager(settings.database.url)
    
    try:
        # Update status to RUNNING
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.update_status(
                run_id=run_id,
                status="RUNNING",
                started_at=datetime.now()
            )
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": "Backtest started",
                    "log_category": "INITIALIZATION"
                }]
            )
        
        # Get controller config (similar to sync version)
        if isinstance(request.config, str):
            controller_config = backtesting_engine_async.get_controller_config_instance_from_yml(
                config_path=request.config,
                controllers_conf_dir_path=settings.app.controllers_path,
                controllers_module=settings.app.controllers_module
            )
        else:
            controller_config = backtesting_engine_async.get_controller_config_instance_from_dict(
                config_data=request.config,
                controllers_module=settings.app.controllers_module
            )
        
        # Log configuration loaded
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": f"Controller config loaded: {controller_config.controller_name}",
                    "log_category": "INITIALIZATION"
                }]
            )
        
        # Run backtesting with ASYNC engine (支持异步 AI Agent)
        # Controller 会通过 contextvars 获取 run_id 并实时记录日志
        backtesting_results = await backtesting_engine_async.run_backtesting(
            controller_config=controller_config,
            trade_cost=request.trade_cost,
            start=int(request.start_time),
            end=int(request.end_time),
            backtesting_resolution=request.backtesting_resolution
        )
        
        # Log completion
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": "Backtesting simulation completed",
                    "log_category": "COMPLETED"
                }]
            )
        
        # Store trades
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            executors = backtesting_results["executors"]
            
            for executor in executors:
                await repo.create_trade(run_id, executor.to_dict())
            
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": f"Stored {len(executors)} trades",
                    "log_category": "COMPLETED"
                }]
            )
        
        # Update final results
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            results = backtesting_results["results"]
            results["sharpe_ratio"] = results["sharpe_ratio"] if results["sharpe_ratio"] is not None else 0
            
            await repo.update_results(
                run_id=run_id,
                results=results,
                total_trades=len(executors),
                status="COMPLETED",
                completed_at=datetime.now()
            )
            
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": f"Backtest completed. Net PnL: {results.get('net_pnl', 0):.4%}",
                    "log_category": "COMPLETED"
                }]
            )
        
    except asyncio.CancelledError:
        # Task was cancelled by user
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.update_status(
                run_id=run_id,
                status="CANCELLED",
                completed_at=datetime.now()
            )
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "INFO",
                    "log_message": "Backtest cancelled by user",
                    "log_category": "COMPLETED"
                }]
            )
        raise
        
    except Exception as e:
        # Log error
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            await repo.update_status(
                run_id=run_id,
                status="FAILED",
                error_message=str(e),
                completed_at=datetime.now()
            )
            await repo.bulk_create_logs(
                run_id=run_id,
                logs=[{
                    "log_level": "ERROR",
                    "log_message": f"Backtest failed: {str(e)}",
                    "log_category": "ERROR"
                }]
            )
        raise
        
    finally:
        await db_manager.close()
        # Remove from running tasks
        if run_id in _running_tasks:
            del _running_tasks[run_id]


@router.post("/start", response_model=dict)
async def start_backtest(request: BacktestStartRequest):
    """
    Start a new backtest run in the background.
    
    Returns the run_id to track progress.
    """
    db_manager = AsyncDatabaseManager(settings.database.url)
    
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
        
        # Create background task to run the backtest
        task = asyncio.create_task(run_backtest_task(run_id, request))
        _running_tasks[run_id] = task
        
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
    db_manager = AsyncDatabaseManager(settings.database.url)
    
    try:
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            backtest_run = await repo.get_backtest_run(run_id)
            
            if not backtest_run:
                raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
            
            # Convert to response model with sanitized float values
            return BacktestStatusResponse(
                run_id=backtest_run.run_id,
                run_name=backtest_run.run_name,
                status=backtest_run.status,
                controller_name=backtest_run.controller_name,
                start_time=backtest_run.start_time,
                end_time=backtest_run.end_time,
                backtesting_resolution=backtest_run.backtesting_resolution,
                trade_cost=sanitize_float_value(float(backtest_run.trade_cost)),
                created_at=backtest_run.created_at.isoformat(),
                started_at=backtest_run.started_at.isoformat() if backtest_run.started_at else None,
                completed_at=backtest_run.completed_at.isoformat() if backtest_run.completed_at else None,
                error_message=backtest_run.error_message,
                total_trades=backtest_run.total_trades,
                win_rate=sanitize_float_value(float(backtest_run.win_rate)) if backtest_run.win_rate is not None else None,
                net_pnl_quote=sanitize_float_value(float(backtest_run.net_pnl_quote)) if backtest_run.net_pnl_quote is not None else None,
                net_pnl_pct=sanitize_float_value(float(backtest_run.net_pnl_pct)) if backtest_run.net_pnl_pct is not None else None,
                max_drawdown=sanitize_float_value(float(backtest_run.max_drawdown)) if backtest_run.max_drawdown is not None else None,
                sharpe_ratio=sanitize_float_value(float(backtest_run.sharpe_ratio)) if backtest_run.sharpe_ratio is not None else None
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
    db_manager = AsyncDatabaseManager(settings.database.url)
    
    try:
        async with db_manager.get_session_context() as session:
            repo = BacktestRunRepository(session)
            
            # Get backtest run info
            backtest_run = await repo.get_backtest_run(run_id)
            if not backtest_run:
                raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
            
            # Get trades and sanitize float values
            trades = await repo.get_trades(run_id)
            trade_entries = [
                BacktestTradeEntry(
                    executor_id=trade.executor_id,
                    trading_pair=trade.trading_pair,
                    side=trade.side,
                    entry_timestamp=trade.entry_timestamp,
                    exit_timestamp=trade.exit_timestamp,
                    entry_price=sanitize_float_value(float(trade.entry_price)),
                    exit_price=sanitize_float_value(float(trade.exit_price)) if trade.exit_price is not None else None,
                    amount=sanitize_float_value(float(trade.amount)),
                    net_pnl_quote=sanitize_float_value(float(trade.net_pnl_quote)),
                    net_pnl_pct=sanitize_float_value(float(trade.net_pnl_pct)),
                    cum_fees_quote=sanitize_float_value(float(trade.cum_fees_quote)),
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
            
            # Build response with sanitized float values
            run_info = BacktestStatusResponse(
                run_id=backtest_run.run_id,
                run_name=backtest_run.run_name,
                status=backtest_run.status,
                controller_name=backtest_run.controller_name,
                start_time=backtest_run.start_time,
                end_time=backtest_run.end_time,
                backtesting_resolution=backtest_run.backtesting_resolution,
                trade_cost=sanitize_float_value(float(backtest_run.trade_cost)),
                created_at=backtest_run.created_at.isoformat(),
                started_at=backtest_run.started_at.isoformat() if backtest_run.started_at else None,
                completed_at=backtest_run.completed_at.isoformat() if backtest_run.completed_at else None,
                error_message=backtest_run.error_message,
                total_trades=backtest_run.total_trades,
                win_rate=sanitize_float_value(float(backtest_run.win_rate)) if backtest_run.win_rate is not None else None,
                net_pnl_quote=sanitize_float_value(float(backtest_run.net_pnl_quote)) if backtest_run.net_pnl_quote is not None else None,
                net_pnl_pct=sanitize_float_value(float(backtest_run.net_pnl_pct)) if backtest_run.net_pnl_pct is not None else None,
                max_drawdown=sanitize_float_value(float(backtest_run.max_drawdown)) if backtest_run.max_drawdown is not None else None,
                sharpe_ratio=sanitize_float_value(float(backtest_run.sharpe_ratio)) if backtest_run.sharpe_ratio is not None else None
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
    Stop a running backtest by cancelling the task.
    """
    if run_id not in _running_tasks:
        raise HTTPException(status_code=404, detail=f"No running backtest found for run_id: {run_id}")
    
    try:
        task = _running_tasks[run_id]
        
        # Check if task is still running
        if not task.done():
            # Cancel the task
            task.cancel()
            
            return {
                "run_id": run_id,
                "message": f"Stop signal sent to backtest {run_id}",
                "status": "STOPPING"
            }
        else:
            # Task already finished
            del _running_tasks[run_id]
            return {
                "run_id": run_id,
                "message": "Backtest already finished",
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
    db_manager = AsyncDatabaseManager(settings.database.url)
    
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
                    trade_cost=sanitize_float_value(float(run.trade_cost)),
                    created_at=run.created_at.isoformat(),
                    started_at=run.started_at.isoformat() if run.started_at else None,
                    completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    error_message=run.error_message,
                    total_trades=run.total_trades,
                    win_rate=sanitize_float_value(float(run.win_rate)) if run.win_rate is not None else None,
                    net_pnl_quote=sanitize_float_value(float(run.net_pnl_quote)) if run.net_pnl_quote is not None else None,
                    net_pnl_pct=sanitize_float_value(float(run.net_pnl_pct)) if run.net_pnl_pct is not None else None,
                    max_drawdown=sanitize_float_value(float(run.max_drawdown)) if run.max_drawdown is not None else None,
                    sharpe_ratio=sanitize_float_value(float(run.sharpe_ratio)) if run.sharpe_ratio is not None else None
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

