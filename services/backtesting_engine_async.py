# 🔥 终极方案：改为异步 `determine_executor_actions`

## 🔍 为什么 Hummingbot 设计成同步？

### 调用链分析

#### 1️⃣ 实盘模式（StrategyV2）

```python
# controller_base.py:171
async def control_task(self):
    if self.market_data_provider.ready and self.executors_update_event.is_set():
        await self.update_processed_data()
        
        # 🔑 在 async 函数中同步调用
        executor_actions: List[ExecutorAction] = self.determine_executor_actions()
        
        if len(executor_actions) > 0:
            await self.send_actions(executor_actions)
```

#### 2️⃣ 回测模式（BacktestingEngine）

```python
# backtesting_engine_base.py:115
async def simulate_execution(self, trade_cost: float):
    for i, row in processed_features.iterrows():
        await self.update_state(row)
        
        # 🔑 在 async 函数中同步调用
        for action in self.controller.determine_executor_actions():
            ...
```

---

### 💡 关键发现

**两种模式都在 `async` 函数中调用！**

所以 Hummingbot 设计成同步的原因是：
1. ✅ **简单**：大多数策略逻辑是计算型的，不需要 I/O
2. ✅ **兼容**：不强制所有 Controller 实现 async
3. ✅ **历史遗留**：早期版本没有考虑 LLM 调用这种长耗时操作

**但这并不意味着不能改成异步！**

---

## ✅ 正确的解决方案：改为 `async def`

### 方案 1：在你的项目中覆盖基类（推荐）⭐

你可以在 `hummingbot-api` 项目中创建一个增强版的 `ControllerBase`：

```python
# hummingbot-api/bots/controllers/controller_base_async.py
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase as HBControllerBase
from typing import List
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction

class ControllerBaseAsync(HBControllerBase):
    """
    异步增强版 ControllerBase
    
    覆盖 control_task，支持异步 determine_executor_actions
    """
    
    async def control_task(self):
        """覆盖父类的 control_task，支持异步决策"""
        if self.market_data_provider.ready and self.executors_update_event.is_set():
            await self.update_processed_data()
            
            # 🔑 改为异步调用
            executor_actions = await self.determine_executor_actions_async()
            
            if len(executor_actions) > 0:
                self.logger().debug(f"Sending actions: {executor_actions}")
                await self.send_actions(executor_actions)
    
    async def determine_executor_actions_async(self) -> List[ExecutorAction]:
        """
        异步版本的 determine_executor_actions
        子类应该覆盖这个方法
        """
        # 默认回退到同步版本（兼容旧的 Controller）
        return self.determine_executor_actions()
    
    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        保留同步版本以兼容
        但建议子类覆盖 determine_executor_actions_async
        """
        raise NotImplementedError("Please implement determine_executor_actions_async instead")
```

---

### 方案 2：修改回测引擎支持异步（更彻底）⭐⭐⭐

在你的项目中创建增强版回测引擎：

```python
# hummingbot-api/services/backtesting_engine_async.py
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
import inspect

class BacktestingEngineAsync(BacktestingEngineBase):
    """
    异步增强版回测引擎
    
    支持 Controller 的 determine_executor_actions 是异步方法
    """
    
    async def simulate_execution(self, trade_cost: float):
        """覆盖父类，支持异步 determine_executor_actions"""
        processed_features = self.prepare_market_data()
        self.active_executor_simulations = []
        self.stopped_executors_info = []
        
        for i, row in processed_features.iterrows():
            await self.update_state(row)
            
            # 🔑 检测 determine_executor_actions 是否是异步方法
            if inspect.iscoroutinefunction(self.controller.determine_executor_actions):
                # 异步调用
                actions = await self.controller.determine_executor_actions()
            else:
                # 同步调用（兼容旧的 Controller）
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
```

---

### 方案 3：使用增强的 AI Agent Controller

修改你的 `ai_agent_v1.py`：

```python
# hummingbot-api/bots/controllers/directional_trading/ai_agent_v1.py
from typing import List
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase
)

class AIAgentV1Controller(DirectionalTradingControllerBase):
    
    # 🔑 改为异步方法
    async def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        异步版本的决策方法
        
        ⚠️ 需要配合增强版回测引擎使用
        """
        current_time = self.market_data_provider.time()
        
        # 检查决策间隔
        if self._last_decision_time == 0:
            self._last_decision_time = current_time - self.config.decision_interval
        
        time_since_last = current_time - self._last_decision_time
        
        if time_since_last < self.config.decision_interval:
            return []
        
        # 🔑 直接异步执行 AI 决策（不再延迟）
        self.logger().info("=" * 80)
        self.logger().info(f"🤖 AI Decision Cycle Triggered")
        self.logger().info("=" * 80)
        
        try:
            # ✅ 直接 await，不延迟
            ai_decisions = await self._execute_ai_decision_cycle()
            
        except Exception as e:
            self.logger().error(f"❌ AI decision cycle failed: {e}", exc_info=True)
            ai_decisions = []
        
        # 更新决策时间
        self._last_decision_time = current_time
        
        # 生成 Executor Actions
        if not ai_decisions:
            return []
        
        actions = []
        for decision in ai_decisions:
            action_type = decision.get("action")
            
            if action_type == "open_long":
                action = self._create_open_action(decision, TradeType.BUY)
                if action:
                    actions.append(action)
            elif action_type == "open_short":
                action = self._create_open_action(decision, TradeType.SELL)
                if action:
                    actions.append(action)
            elif action_type == "close_position":
                action = self._create_close_action(decision)
                if action:
                    actions.append(action)
        
        return [action for action in actions if action is not None]
```

---

## 🚀 实施步骤

### Step 1: 创建增强版回测引擎

<function_calls>
<invoke name="write">
<parameter name="contents">"""
异步增强版回测引擎

支持 Controller 的 determine_executor_actions 是异步方法
"""
import inspect
from typing import List
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.executors.data_types import CloseType


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

