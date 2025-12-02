"""
异步增强版回测引擎

支持 Controller 的 determine_executor_actions 是异步方法
"""
import inspect
from typing import List
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType


class BacktestingEngineAsync(BacktestingEngineBase):
    """
    异步增强版回测引擎
    
    主要改进：
    1. 支持 async determine_executor_actions()
    2. 自动检测 Controller 方法是否是异步的
    3. 向后兼容同步 Controller
    """
    
    async def simulate_execution(self, trade_cost: float) -> list:
        """
        覆盖父类方法，支持异步 determine_executor_actions
        
        Args:
            trade_cost (float): The cost per trade.
        
        Returns:
            List[ExecutorInfo]: List of executor information objects detailing the simulation results.
        """
        processed_features = self.prepare_market_data()
        self.active_executor_simulations: List = []
        self.stopped_executors_info: List = []
        
        for i, row in processed_features.iterrows():
            await self.update_state(row)
            
            # 🔑 检测 determine_executor_actions 是否是异步方法
            if inspect.iscoroutinefunction(self.controller.determine_executor_actions):
                # ✅ 异步调用（新的 AI Agent）
                actions = await self.controller.determine_executor_actions()
            else:
                # ✅ 同步调用（兼容旧的 Controller）
                actions = self.controller.determine_executor_actions()
            
            # 处理 actions
            for action in actions:
                if isinstance(action, CreateExecutorAction):
                    executor_simulation = self.simulate_executor(
                        action.executor_config, 
                        processed_features.loc[i:], 
                        trade_cost
                    )
                    if executor_simulation is not None and executor_simulation.close_type != CloseType.FAILED:
                        self.manage_active_executors(executor_simulation)
                elif isinstance(action, StopExecutorAction):
                    self.handle_stop_action(action, row["timestamp"])
        
        return self.controller.executors_info
