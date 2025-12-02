import asyncio
import json
import os
import time
from contextvars import ContextVar
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import TradeType, PriceType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction


# 🔑 全局 contextvars（与 backtesting.py 中定义的保持一致）
# 这些变量在每个异步任务中独立，不会互相干扰
try:
    from contextvars import ContextVar
    backtest_run_id_var: ContextVar[Optional[str]] = ContextVar('backtest_run_id', default=None)
    backtest_db_url_var: ContextVar[Optional[str]] = ContextVar('backtest_db_url', default=None)
except ImportError:
    # 如果导入失败（不太可能），使用 None
    backtest_run_id_var = None
    backtest_db_url_var = None


class AIAgentV1Config(DirectionalTradingControllerConfigBase):
    """AI Agent Trading Controller Configuration"""
    controller_name: str = "ai_agent_v1"
    
    # K线配置（回测必需，与 Bollinger V1 保持一致）
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True
        }
    )
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True
        }
    )
    interval: str = Field(
        default="5m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True
        }
    )
    
    # 多币种配置
    trading_pairs: List[str] = Field(
        default=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        json_schema_extra={
            "prompt": "Enter comma-separated trading pairs (e.g., BTC-USDT,ETH-USDT): ",
            "prompt_on_new": True
        }
    )
    
    # AI 配置
    openrouter_api_key: str = Field(
        default="",
        json_schema_extra={
            "prompt": "Enter OpenRouter API key: ",
            "prompt_on_new": True,
            "is_secure": True
        }
    )
    
    llm_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        json_schema_extra={
            "prompt": "Enter LLM model (e.g., anthropic/claude-3.5-sonnet, deepseek/deepseek-chat): ",
            "prompt_on_new": True
        }
    )
    
    llm_temperature: Decimal = Field(
        default=Decimal("0.1"),
        json_schema_extra={
            "prompt": "Enter LLM temperature (0.0-1.0, lower = more conservative): ",
            "prompt_on_new": True
        }
    )
    
    llm_max_tokens: int = Field(
        default=4000,
        json_schema_extra={
            "prompt": "Enter LLM max tokens: ",
            "prompt_on_new": True
        }
    )
    
    # 决策间隔（秒）
    decision_interval: int = Field(
        default=180,  # 3分钟
        json_schema_extra={
            "prompt": "Enter decision interval in seconds (e.g., 180 for 3 minutes): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    # K线配置
    candles_interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter candles interval (e.g., 3m, 5m, 15m): ",
            "prompt_on_new": True
        }
    )
    
    candles_max_records: int = Field(
        default=100,
        json_schema_extra={
            "prompt": "Enter max candles records: ",
            "prompt_on_new": True
        }
    )
    
    # 风险控制
    max_concurrent_positions: int = Field(
        default=3,
        json_schema_extra={
            "prompt": "Enter max concurrent positions: ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    single_position_size_pct: Decimal = Field(
        default=Decimal("0.3"),
        json_schema_extra={
            "prompt": "Enter single position size as percentage of total_amount_quote (e.g., 0.3 for 30%): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    # Validators（自动设置 candles_connector 和 candles_trading_pair）
    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class AIAgentV1Controller(DirectionalTradingControllerBase):
    """
    AI Agent Trading Controller - V1 MVP
    
    Features:
    - Multi-pair monitoring
    - Funding rate tracking
    - LLM-based decision making (via OpenRouter)
    - Position and trade history tracking
    """
    
    def __init__(self, config: AIAgentV1Config, *args, **kwargs):
        self.config = config
        
        # 为每个交易对配置K线数据
        # 与 Bollinger V1 保持一致的初始化逻辑
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=pair,
                    interval=config.interval,  # 使用 interval 字段（与 candles_interval 保持同步）
                    max_records=config.candles_max_records
                ) for pair in config.trading_pairs
            ]
        
        super().__init__(config, *args, **kwargs)
        
        # 决策时间追踪
        self._last_decision_time = 0
        self._decision_in_progress = False
        
        # 历史 Funding Rate 缓存（用于回测）
        self._historical_funding_rates: Dict[str, pd.DataFrame] = {}
        self._funding_rate_initialized = False
        
        # 🔑 从 contextvars 获取回测上下文（线程安全，支持多任务并发）
        self._backtest_run_id = None
        self._backtest_db_url = None
        self._decision_cycle_count = 0  # 决策轮次计数
        
        # 尝试从 contextvars 获取回测信息
        if backtest_run_id_var is not None:
            try:
                self._backtest_run_id = backtest_run_id_var.get()
                self._backtest_db_url = backtest_db_url_var.get()
            except LookupError:
                # contextvars 未设置（实盘模式）
                pass
        
        # 初始化 LangChain LLM
        self._init_langchain_llm()
        
        self.logger().info(f"AI Agent V1 initialized - monitoring {len(config.trading_pairs)} pairs")
        
        if self._backtest_run_id:
            self.logger().info(f"🔬 Backtest mode detected - Run ID: {self._backtest_run_id}")
    
    def _init_langchain_llm(self):
        """初始化 LangChain LLM"""
        try:
            # 使用 LangChain 的 ChatOpenAI（兼容 OpenRouter）
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=float(self.config.llm_temperature),
                max_tokens=self.config.llm_max_tokens,
                timeout=30,
                max_retries=2,
            )
            
            # JSON 输出解析器
            self.json_parser = JsonOutputParser()
            
            self.logger().info(f"LangChain LLM initialized: {self.config.llm_model}")
            
        except Exception as e:
            self.logger().error(f"Failed to initialize LangChain LLM: {e}")
            self.llm = None
            self.json_parser = None
    
    async def _initialize_historical_funding_rates(self):
        """
        初始化历史 Funding Rate 数据（仅回测环境）
        
        通过 Binance API 下载回测时间范围内的历史 funding rate
        API: GET /fapi/v1/fundingRate
        """
        if self._funding_rate_initialized:
            return
        
        # 检测是否是回测环境
        is_backtest = not hasattr(self, 'connectors') or not self.connectors
        
        if not is_backtest:
            self.logger().info("Live trading mode - skipping historical funding rate download")
            self._funding_rate_initialized = True
            return
        
        # 只支持 Binance Perpetual
        if "binance_perpetual" not in self.config.connector_name:
            self.logger().warning(f"Historical funding rate download only supports binance_perpetual, got {self.config.connector_name}")
            self._funding_rate_initialized = True
            return
        
        self.logger().info("🔄 Downloading historical funding rates for backtest...")
        
        try:
            import aiohttp
            
            # 获取回测时间范围
            start_time = int(self.market_data_provider.start_time * 1000)  # 转换为 ms
            end_time = int(self.market_data_provider.end_time * 1000)
            
            base_url = "https://fapi.binance.com"
            
            for trading_pair in self.config.trading_pairs:
                # 转换交易对格式 BTC-USDT -> BTCUSDT
                symbol = trading_pair.replace("-", "")
                
                try:
                    async with aiohttp.ClientSession() as session:
                        all_funding_rates = []
                        current_start = start_time
                        
                        # 分批下载（每次最多1000条）
                        while current_start < end_time:
                            url = f"{base_url}/fapi/v1/fundingRate"
                            params = {
                                "symbol": symbol,
                                "startTime": current_start,
                                "endTime": end_time,
                                "limit": 1000
                            }
                            
                            async with session.get(url, params=params) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if not data:
                                        break
                                    
                                    all_funding_rates.extend(data)
                                    
                                    # 更新下次查询的起始时间
                                    current_start = data[-1]["fundingTime"] + 1
                                    
                                    if len(data) < 1000:
                                        break
                                else:
                                    self.logger().error(
                                        f"Failed to download funding rate for {trading_pair}: "
                                        f"HTTP {response.status}"
                                    )
                                    break
                        
                        if all_funding_rates:
                            # 转换为 DataFrame
                            df = pd.DataFrame(all_funding_rates)
                            df["fundingRate"] = df["fundingRate"].astype(float)
                            df["fundingTime"] = df["fundingTime"].astype(int) / 1000  # 转换为秒
                            df["markPrice"] = df["markPrice"].astype(float)
                            
                            self._historical_funding_rates[trading_pair] = df
                            
                            self.logger().info(
                                f"✅ Downloaded {len(df)} funding rate records for {trading_pair} "
                                f"(from {pd.to_datetime(df['fundingTime'].min(), unit='s')} to "
                                f"{pd.to_datetime(df['fundingTime'].max(), unit='s')})"
                            )
                        else:
                            self.logger().warning(f"No funding rate data found for {trading_pair}")
                            
                except Exception as e:
                    self.logger().error(f"Failed to download funding rate for {trading_pair}: {e}", exc_info=True)
                    
        except Exception as e:
            self.logger().error(f"Failed to initialize historical funding rates: {e}", exc_info=True)
        
        self._funding_rate_initialized = True
        self.logger().info(f"📊 Historical funding rates initialized for {len(self._historical_funding_rates)} pairs")
    
    async def update_processed_data(self):
        """
        更新处理后的数据
        
        ⚠️  注意：回测引擎只在开始前调用一次此方法，然后在循环中直接调用 determine_executor_actions()
        因此决策间隔逻辑已移到 determine_executor_actions() 中处理
        
        实盘模式：每个 tick 调用（但实际决策在 determine_executor_actions）
        回测模式：只调用一次（用于初始化）
        """
        # 简单标记数据已准备好
        self.processed_data["initialized"] = True
        
        # 实盘模式下，可以在这里预加载一些数据
        # 但不执行 AI 决策（决策在 determine_executor_actions 中）
        pass
    
    async def _build_trading_context(self) -> Dict:
        """
        构建 AI 决策所需的完整上下文
        """
        # 初始化历史 funding rate（仅回测环境，只在第一次调用时执行）
        if not self._funding_rate_initialized:
            await self._initialize_historical_funding_rates()
        
        context = {
            "timestamp": self.market_data_provider.time(),
            "account": self._get_account_summary(),
            "positions": self._get_positions_summary(),
            "market_data": {},
            "funding_rates": {},
            "recent_trades": self._get_recent_trades(limit=10),
        }
        
        # 收集每个币种的市场数据
        self.logger().debug(f"Collecting market data for {len(self.config.trading_pairs)} pairs...")
        
        for pair in self.config.trading_pairs:
            try:
                self.logger().debug(f"Getting market info for {pair}...")
                market_info = await self._get_market_info(pair)
                
                # 🔧 即使有错误也要记录（便于调试）
                context["market_data"][pair] = market_info
                
                if "error" in market_info:
                    self.logger().warning(f"⚠️  {pair}: {market_info['error']}")
                else:
                    self.logger().debug(f"✅ {pair}: Price ${market_info.get('current_price', 0):.2f}")
                
                # 获取资金费率（仅 Perpetual）
                if "_perpetual" in self.config.connector_name:
                    try:
                        funding_rate = await self._get_funding_rate(pair)
                        context["funding_rates"][pair] = funding_rate
                        self.logger().debug(f"✅ {pair}: Funding rate {funding_rate.get('rate', 0)*100:.4f}%")
                    except Exception as e:
                        self.logger().warning(f"⚠️  Failed to get funding rate for {pair}: {e}")
                        # 即使失败也要设置默认值，确保 context 中有这个字段
                        context["funding_rates"][pair] = {"rate": 0.0, "next_funding_time": 0}
                    
            except Exception as e:
                self.logger().error(f"❌ Failed to get market info for {pair}: {e}", exc_info=True)
                # 🔧 记录错误，而不是跳过
                context["market_data"][pair] = {"error": str(e)}
        
        self.logger().info(f"Market data collected: {len(context['market_data'])} pairs")
        
        return context
    
    def _get_account_summary(self) -> Dict:
        """获取账户摘要"""
        # 计算当前持仓总价值 = 初始资本 + 所有已结束交易的盈亏 + 当前活跃持仓的未实现盈亏
        initial_capital = float(self.config.total_amount_quote)
        current_holdings = initial_capital
        
        # 统计已结束交易的累计盈亏
        closed_pnl = 0.0
        active_pnl = 0.0
        
        for executor in self.executors_info:
            try:
                # 已结束的交易（已实现盈亏）
                if hasattr(executor, 'status') and str(executor.status) == 'RunnableStatus.TERMINATED':
                    if hasattr(executor, 'net_pnl_quote') and executor.net_pnl_quote is not None:
                        pnl = float(executor.net_pnl_quote)
                        closed_pnl += pnl
                        self.logger().debug(f"Closed executor {executor.id}: PnL ${pnl:.2f}")
                
                # 活跃持仓（未实现盈亏）
                elif executor.is_active and executor.is_trading:
                    if hasattr(executor, 'net_pnl_quote') and executor.net_pnl_quote is not None:
                        pnl = float(executor.net_pnl_quote)
                        active_pnl += pnl
                        self.logger().debug(f"Active executor {executor.id}: PnL ${pnl:.2f}")
                        
            except Exception as e:
                self.logger().warning(f"Error calculating PnL for executor {executor.id}: {e}")
        
        # 当前账户总价值 = 初始资本 + 已实现盈亏 + 未实现盈亏
        current_holdings = initial_capital + closed_pnl + active_pnl
        total_pnl = current_holdings - initial_capital
        
        self.logger().info(f"Account Summary: Initial=${initial_capital:.2f}, Closed PnL=${closed_pnl:.2f}, Active PnL=${active_pnl:.2f}, Total=${current_holdings:.2f}")
        
        return {
            "total_amount_quote": initial_capital,  # 初始资本
            "current_holdings": current_holdings,  # 当前账户总价值
            "total_pnl": total_pnl,  # 总盈亏（已实现 + 未实现）
            "closed_pnl": closed_pnl,  # 已实现盈亏
            "active_pnl": active_pnl,  # 未实现盈亏
            "max_concurrent_positions": self.config.max_concurrent_positions,
            "single_position_size_pct": float(self.config.single_position_size_pct),
        }
    
    def _get_positions_summary(self) -> List[Dict]:
        """获取当前持仓摘要"""
        positions = []
        
        for executor in self.executors_info:
            if executor.is_active and executor.is_trading:
                try:
                    pos = {
                        "symbol": executor.config.trading_pair,
                        "side": executor.config.side.name,
                        "entry_price": float(executor.config.entry_price),
                        "amount": float(executor.config.amount),
                        "net_pnl_pct": float(executor.net_pnl_pct),
                        "net_pnl_quote": float(executor.net_pnl_quote),
                        "timestamp": executor.timestamp,
                        "executor_id": executor.id,
                    }
                    positions.append(pos)
                except Exception as e:
                    self.logger().warning(f"Error processing executor {executor.id}: {e}")
        
        return positions
    
    def _get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的交易记录
        
        从 executors_info 中提取已完成的交易
        """
        closed_executors = []
        
        for e in self.executors_info:
            if not (hasattr(e, 'status') and str(e.status) == 'RunnableStatus.TERMINATED'):
                continue
            
            # 获取入场价格
            entry_price = 0.0
            if hasattr(e, 'entry_price') and e.entry_price:
                entry_price = float(e.entry_price)
            elif hasattr(e, 'config') and hasattr(e.config, 'entry_price'):
                entry_price = float(e.config.entry_price)
            
            # 计算退出价格（基于 PnL）
            exit_price = 0.0
            if entry_price > 0 and hasattr(e, 'net_pnl_pct') and e.net_pnl_pct:
                pnl_pct = float(e.net_pnl_pct)
                side_multiplier = 1 if e.config.side.name == "BUY" else -1
                # exit_price = entry_price * (1 + pnl_pct * side_multiplier)
                # 简化：直接根据 PnL 百分比计算
                exit_price = entry_price * (1 + pnl_pct * side_multiplier)
            
            trade = {
                "symbol": e.config.trading_pair,
                "side": e.config.side.name,
                "entry_price": entry_price,
                "exit_price": exit_price,  # 🔧 添加退出价格
                "pnl_pct": float(e.net_pnl_pct) if hasattr(e, 'net_pnl_pct') else 0,
                "pnl_quote": float(e.net_pnl_quote) if hasattr(e, 'net_pnl_quote') else 0,
                "close_type": e.close_type.name if hasattr(e, 'close_type') and e.close_type else "UNKNOWN",
                "timestamp": e.timestamp if hasattr(e, 'timestamp') else 0,
                "close_timestamp": e.close_timestamp if hasattr(e, 'close_timestamp') else 0,
            }
            closed_executors.append(trade)
        
        return closed_executors[-limit:] if closed_executors else []
    
    async def _get_market_info(self, trading_pair: str) -> Dict:
        """获取单个币种的市场信息"""
        try:
            self.logger().debug(f"Fetching candles for {trading_pair}...")
            self.logger().debug(f"  Connector: {self.config.connector_name}")
            self.logger().debug(f"  Interval: {self.config.interval}")
            self.logger().debug(f"  Max records: {self.config.candles_max_records}")
            
            # 🔧 调试：检查 market_data_provider 中是否有这个交易对的数据
            available_keys = list(self.market_data_provider.candles_feeds.keys()) if hasattr(self.market_data_provider, 'candles_feeds') else []
            self.logger().debug(f"  Available candles feeds: {available_keys}")
            
            # 获取K线数据
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=trading_pair,
                interval=self.config.interval,
                max_records=self.config.candles_max_records
            )
            
            # 🔑 关键修复：过滤未来数据（防止 look-ahead bias）
            # 兼容实盘和回测两种模式
            if hasattr(self.market_data_provider, 'time'):
                current_time = self.market_data_provider.time()
                
                if not candles.empty and "timestamp" in candles.columns:
                    before_filter = len(candles)
                    candles = candles[candles["timestamp"] <= current_time]
                    after_filter = len(candles)
                    
                    # 只在回测时记录过滤信息（避免实盘日志过多）
                    if before_filter != after_filter:
                        self.logger().warning(
                            f"🔒 Time filter: {before_filter} → {after_filter} candles "
                            f"(current_time: {pd.to_datetime(current_time, unit='s')})"
                        )
                    
                    # 只保留最近 max_records 条（避免计算太多历史数据）
                    if len(candles) > self.config.candles_max_records:
                        candles = candles.tail(self.config.candles_max_records)
                        self.logger().warning(f"   Keeping last {self.config.candles_max_records} candles")
            
            self.logger().warning(f"Received {len(candles) if not candles.empty else 0} candles for {trading_pair}")
            
            # 🔧 修复：如果数据不足，返回基础信息而不是错误
            # 在回测早期，数据可能不足 20 根，但仍应提供价格信息
            if candles.empty:
                self.logger().warning(
                    f"❌ No candles for {trading_pair}\n"
                    f"   Current time: {pd.to_datetime(current_time, unit='s') if 'current_time' in locals() else 'N/A'}\n"
                    f"   Available feeds: {available_keys}\n"
                    f"   Looking for: {self.config.connector_name}_{trading_pair}_{self.config.interval}"
                )
                return {"error": "no_data", "symbol": trading_pair, "candles_count": 0}
            
            # 计算技术指标（尽可能使用可用数据）
            close = candles["close"]
            high = candles["high"]
            low = candles["low"]
            
            current_price = float(close.iloc[-1])
            candles_count = len(candles)
            
            self.logger().debug(f"Calculating indicators for {trading_pair} with {candles_count} candles...")
            
            # 🔧 计算技术指标的完整序列（用于趋势分析）
            rsi_series = ta.rsi(close, length=14) if candles_count >= 14 else None
            macd_df = ta.macd(close, fast=12, slow=26, signal=9) if candles_count >= 26 else None
            ema_20_series = ta.ema(close, length=20) if candles_count >= 20 else None
            
            # 提取最近 5 个值（显示趋势变化）
            history_length = min(5, candles_count)
            
            market_info = {
                "symbol": trading_pair,
                "current_price": current_price,
                
                # 当前值（向后兼容）
                "rsi": float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.isna().iloc[-1] else None,
                "macd": float(macd_df[f"MACD_12_26_9"].iloc[-1]) if macd_df is not None and not macd_df.empty else None,
                "macd_signal": float(macd_df[f"MACDs_12_26_9"].iloc[-1]) if macd_df is not None and not macd_df.empty else None,
                "ema_20": float(ema_20_series.iloc[-1]) if ema_20_series is not None and not ema_20_series.isna().iloc[-1] else None,
                
                # 🔑 新增：历史趋势数据（最近 5 个值）
                "price_history": [float(p) for p in close.tail(history_length).tolist()],
                "rsi_history": [float(v) for v in rsi_series.tail(history_length).tolist()] if rsi_series is not None else None,
                "macd_history": [float(v) for v in macd_df[f"MACD_12_26_9"].tail(history_length).tolist()] if macd_df is not None else None,
                "macd_signal_history": [float(v) for v in macd_df[f"MACDs_12_26_9"].tail(history_length).tolist()] if macd_df is not None else None,
                "ema_20_history": [float(v) for v in ema_20_series.tail(history_length).tolist()] if ema_20_series is not None else None,
                "volume_history": [float(v) for v in candles["volume"].tail(history_length).tolist()],  # 成交量历史
                
                "price_change_24h_pct": self._calculate_price_change(candles),
                "volume_24h": float(candles["volume"].sum()),  # 24小时总成交量
                "avg_volume": float(candles["volume"].mean()),  # 平均成交量
                "candles_available": candles_count,
            }
            
            # 🔧 修复：先格式化值，再构建日志字符串
            rsi_str = f"{market_info['rsi']:.1f}" if market_info['rsi'] is not None else 'N/A'
            macd_str = f"{market_info['macd']:.2f}" if market_info['macd'] is not None else 'N/A'
            ema_str = f"${market_info['ema_20']:.2f}" if market_info['ema_20'] is not None else 'N/A'
            
            # 如果数据不足，添加警告
            warning_suffix = f" (⚠️ Limited data: {candles_count} candles)" if candles_count < 20 else ""
            
            self.logger().warning(
                f"✅ {trading_pair}: Price=${current_price:.2f}, "
                f"RSI={rsi_str}, MACD={macd_str}, EMA(20)={ema_str}{warning_suffix}"
            )
            self.logger().debug(f"   Price history (last 5): {market_info['price_history']}")
            self.logger().debug(f"   RSI history (last 5): {market_info['rsi_history']}")
            
            return market_info
            
        except Exception as e:
            self.logger().error(f"❌ Error getting market info for {trading_pair}: {e}", exc_info=True)
            return {"error": str(e), "symbol": trading_pair}
    
    def _calculate_price_change(self, candles: pd.DataFrame) -> float:
        """计算24小时价格变化百分比"""
        if len(candles) < 2:
            return 0.0
        first_price = float(candles["close"].iloc[0])
        last_price = float(candles["close"].iloc[-1])
        return ((last_price - first_price) / first_price) * 100
    
    async def _get_funding_rate(self, trading_pair: str) -> Dict:
        """
        获取资金费率（仅 Perpetual 合约）
        
        优先级：
        1. 历史数据（回测环境）
        2. market_data_provider（实盘环境）
        3. connector（实盘环境备选）
        4. 默认值（fallback）
        """
        try:
            current_time = self.market_data_provider.time()
            
            # 方法 1: 从历史数据查询（回测环境）
            if trading_pair in self._historical_funding_rates:
                df = self._historical_funding_rates[trading_pair]
                
                # 查找最接近当前时间的 funding rate
                # funding rate 每8小时更新一次，找到距离当前时间最近且不晚于当前时间的那条
                past_rates = df[df["fundingTime"] <= current_time]
                
                if not past_rates.empty:
                    # 取最新的一条
                    latest_rate = past_rates.iloc[-1]
                    rate = float(latest_rate["fundingRate"])
                    funding_time = int(latest_rate["fundingTime"])
                    
                    self.logger().debug(
                        f"✅ Using historical funding rate for {trading_pair}: "
                        f"{rate*100:.4f}% at {pd.to_datetime(funding_time, unit='s')}"
                    )
                    
                    return {
                        "rate": rate,
                        "next_funding_time": funding_time + 28800,  # 8小时后
                    }
                else:
                    self.logger().warning(
                        f"No historical funding rate found for {trading_pair} at time {current_time}, "
                        f"available range: {df['fundingTime'].min()} - {df['fundingTime'].max()}"
                    )
            
            # 方法 2: 使用 market_data_provider (实盘环境)
            if hasattr(self, 'market_data_provider') and self.market_data_provider:
                try:
                    funding_info = self.market_data_provider.get_funding_info(
                        self.config.connector_name, 
                        trading_pair
                    )
                    if funding_info:
                        self.logger().info(
                            f"✅ Got funding rate for {trading_pair}: {float(funding_info.rate)*100:.4f}%"
                        )
                        return {
                            "rate": float(funding_info.rate),
                            "next_funding_time": funding_info.next_funding_utc_timestamp,
                        }
                except Exception as e:
                    self.logger().debug(f"market_data_provider.get_funding_info failed for {trading_pair}: {e}")
            
            # 方法 3: 直接从 connector 获取 (实盘环境备选)
            if hasattr(self, 'connectors') and self.connectors:
                connector = self.connectors.get(self.config.connector_name)
                if connector and hasattr(connector, 'get_funding_info'):
                    try:
                        funding_info = connector.get_funding_info(trading_pair)
                        if funding_info:
                            self.logger().info(
                                f"✅ Got funding rate for {trading_pair}: {float(funding_info.rate)*100:.4f}%"
                            )
                            return {
                                "rate": float(funding_info.rate),
                                "next_funding_time": funding_info.next_funding_utc_timestamp,
                            }
                    except Exception as e:
                        self.logger().debug(f"connector.get_funding_info failed for {trading_pair}: {e}")
            
            # 方法 4: 回测环境或无法获取，返回默认值
            # 回测引擎使用离线数据，无法获取实时 funding rate
            self.logger().debug(
                f"Funding rate not available for {trading_pair}. Using default neutral rate (0.0%)"
            )
            return {"rate": 0.0, "next_funding_time": 0}
                
        except Exception as e:
            self.logger().error(f"Failed to get funding rate for {trading_pair}: {e}")
            return {"rate": 0.0, "next_funding_time": 0}
    
    async def _get_ai_decisions(self, context: Dict) -> List[Dict]:
        """
        调用 LLM 获取交易决策（使用 LangChain）
        """
        try:
            # Step 1: 构建 Prompt
            self.logger().debug("Building prompts...")
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context)
            

            # self.logger().warning(f"System prompt: {system_prompt}")
            self.logger().warning(f"User prompt: {user_prompt}")
        
            
            # Step 2: 使用 LangChain 调用 LLM
            self.logger().info("Calling LLM API...")
            response = await self._call_langchain_llm(system_prompt, user_prompt)
            self.logger().info("LLM response received")
            
            # 打印完整响应用于调试
            self.logger().warning(f"LLM full response:\n{response}")
            
            # Step 3: 解析决策
            self.logger().debug("Parsing LLM response...")
            decisions = self._parse_ai_response(response)
            self.logger().info(f"Parsed {len(decisions)} raw decisions from LLM")
            
            # Step 4: 验证决策
            self.logger().debug("Validating decisions...")
            validated_decisions = self._validate_decisions(decisions, context)
            self.logger().info(f"Validated {len(validated_decisions)}/{len(decisions)} decisions")
            
            return validated_decisions
            
        except Exception as e:
            self.logger().error(f"Error in AI decision process: {e}", exc_info=True)
            return []
    
    # 固定的 OUTPUT FORMAT（不可配置）
    OUTPUT_FORMAT = """
---

# OUTPUT FORMAT (MANDATORY)

**Respond ONLY with a JSON array. No extra text.**

Required fields for each decision:
- **reasoning** (string): **DETAILED multi-step analysis (4-6 sentences minimum)**
  Must include:
  1. Market context & trend identification (what's happening)
  2. Technical confluence (which indicators align/conflict)
  3. Risk/Reward calculation (entry, SL, TP levels with rationale)
  4. Thesis & invalidation point (what you expect & when to exit)
  5. Why NOW is the right time (or why NOT)
  
- **action** (string): "open_long" | "open_short" | "close_position" | "hold"
- **symbol** (string): Trading pair (e.g., "BTC-USDT") or null
- **stop_loss_pct** (float): 0.015-0.035 (1.5%-3.5%) or null
- **take_profit_pct** (float): 0.03-0.08 (3%-8%, min 2:1 R/R) or null
- **confidence** (int): 0-100 (0 for hold, 50+ to trade)

**Example:**
```json
[
  {
    "reasoning": "",
    "action": "",
    "symbol": "",
    "stop_loss_pct": 0.025,
    "take_profit_pct": 0.05,
    "confidence": 0
  }
]
```
---

Analyze and respond with the JSON array only.
"""
    
    def _build_system_prompt(self) -> str:
        """构建系统 Prompt（可配置部分）"""
        system_prompt = f"""You are an autonomous cryptocurrency trading agent with systematic, disciplined approach.

# ROLE & MISSION
Your mission: Maximize risk-adjusted returns through disciplined trading decisions based on technical analysis and risk management principles.

---

# TRADING ENVIRONMENT

## Your Trading Setup
- **Exchange**: {self.config.connector_name}
- **Available Pairs**: {', '.join(self.config.trading_pairs)}
- **Max Concurrent Positions**: {self.config.max_concurrent_positions}
- **Position Size**: {float(self.config.single_position_size_pct) * 100}% of capital per trade"""
        
        # 🔧 修复：stop_loss, take_profit, time_limit 可能为 None
        if self.config.triple_barrier_config.stop_loss is not None:
            system_prompt += f"\n- **Base Stop Loss**: {float(self.config.triple_barrier_config.stop_loss) * 100}%"
        
        if self.config.triple_barrier_config.take_profit is not None:
            system_prompt += f"\n- **Base Take Profit**: {float(self.config.triple_barrier_config.take_profit) * 100}%"
        
        if self.config.triple_barrier_config.time_limit is not None:
            system_prompt += f"\n- **Max Hold Time**: {self.config.triple_barrier_config.time_limit / 3600:.1f} hours"
        
        system_prompt += """

## Market Type
- **Perpetual Contracts**: No expiration, funding rate mechanism
- **Funding Rate Impact**: Extreme rates (>0.01%) often signal overextension and potential reversal

---

# HIGH-PROBABILITY TRADING FRAMEWORK

## Core Principles (Non-Negotiable)

1. **Trade WITH the trend, not against it**
   - Uptrend: Only LONG on pullbacks to support
   - Downtrend: Only SHORT on rallies to resistance
   - Sideways: WAIT or trade range boundaries with tight stops

2. **Require multiple confirmations (NOT single indicators)**
   - Trend alignment: Price vs EMA
   - Momentum confirmation: MACD direction + histogram expansion
   - Volume confirmation: Increasing volume supports trend validity
   - Sentiment check: RSI NOT in extreme (avoid overbought longs, oversold shorts)
   - Entry timing: Pullback to support (long) or resistance (short)

3. **Risk/Reward BEFORE entry**
   - Minimum 2:1 R/R ratio (prefer 3:1+)
   - Stop loss at invalidation point (below support for long, above resistance for short)
   - Take profit at next resistance (long) or support (short)

4. **Position management discipline**
   - Close losing trades FAST when thesis breaks
   - Let winners run until trend shows weakness
   - Never average down on losing positions

## Trading Setup Checklist (ALL must align)

### For LONG Entry:
- [ ] Price > EMA(20) (uptrend confirmed)
- [ ] MACD > Signal AND histogram expanding (bullish momentum)
- [ ] RSI 40-70 (NOT oversold, healthy pullback)
- [ ] Volume increasing or above average (confirms buying interest)
- [ ] Price pulled back to support or EMA (entry opportunity)
- [ ] Clear stop loss below recent swing low
- [ ] R/R ratio ≥ 2:1

### For SHORT Entry:
- [ ] Price < EMA(20) (downtrend confirmed)
- [ ] MACD < Signal AND histogram expanding down (bearish momentum)
- [ ] RSI 30-60 (NOT overbought, healthy bounce)
- [ ] Volume increasing or above average (confirms selling pressure)
- [ ] Price rallied to resistance or EMA (entry opportunity)
- [ ] Clear stop loss above recent swing high
- [ ] R/R ratio ≥ 2:1

### Exit Signals (Close position immediately):
- [ ] Stop loss hit (no exceptions)
- [ ] Take profit reached
- [ ] MACD crossover against position (momentum shift)
- [ ] Price breaks key support/resistance (structure broken)
- [ ] Funding rate extreme (sentiment exhaustion)

## Decision Priority (Follow in order)

1. **Position Management First**: Check existing positions for exit signals
2. **Market Context**: Identify clear trends (ignore choppy/sideways markets)
3. **Setup Quality**: Scan for multi-confirmed setups meeting ALL checklist items
4. **Risk Assessment**: Calculate R/R, validate stop loss placement
5. **Execute or Wait**: If ANY doubt exists → HOLD (quality > quantity)

## What NOT to Do (Common Mistakes)

❌ Trade without clear trend (sideways = death by 1000 cuts)
❌ Enter on single indicator (RSI alone, MACD alone = low probability)
❌ Chase breakouts without pullback (FOMO = bad entries)
❌ Short oversold or long overbought (fighting momentum)
❌ Hold losing positions hoping for reversal (hope ≠ strategy)
❌ Trade every signal (overtrading = account killer)

## Mindset

- You don't need to trade every day to be profitable
- Missing a trade is better than taking a bad trade
- The best traders are patient, selective, and disciplined
- Your edge comes from waiting for high-probability setups, not from activity

"""
        
        # 添加固定的 OUTPUT FORMAT
        return system_prompt + self.OUTPUT_FORMAT
    
    def _build_user_prompt(self, context: Dict) -> str:
        """构建用户 Prompt（包含实时数据）"""
        import json
        
        prompt_parts = []
        
        # 1. 账户信息
        prompt_parts.append(f"# ACCOUNT STATUS")
        prompt_parts.append(f"Initial Capital: ${context['account']['total_amount_quote']:.2f}")
        prompt_parts.append(f"Current Holdings: ${context['account']['current_holdings']:.2f}")
        
        # 分解盈亏显示
        total_pnl = context['account']['total_pnl']
        closed_pnl = context['account'].get('closed_pnl', 0.0)
        active_pnl = context['account'].get('active_pnl', 0.0)
        pnl_pct = (total_pnl / context['account']['total_amount_quote']) * 100
        pnl_emoji = "📈" if total_pnl > 0 else "📉" if total_pnl < 0 else "➖"
        
        prompt_parts.append(f"Total P&L: ${total_pnl:.2f} ({pnl_pct:+.2f}%) {pnl_emoji}")
        prompt_parts.append(f"  - Realized P&L (Closed): ${closed_pnl:.2f}")
        prompt_parts.append(f"  - Unrealized P&L (Active): ${active_pnl:.2f}")
        
        prompt_parts.append(f"Active Positions: {len(context['positions'])}/{self.config.max_concurrent_positions}")
        prompt_parts.append(f"Available Slots: {self.config.max_concurrent_positions - len(context['positions'])}")
        
        # 2. 当前持仓
        if context["positions"]:
            prompt_parts.append(f"\n# CURRENT POSITIONS")
            for pos in context["positions"]:
                prompt_parts.append(
                    f"\n**{pos['symbol']} {pos['side']}**"
                    f"\n- Entry Price: ${pos['entry_price']:.2f}"
                    f"\n- Current PnL: {pos['net_pnl_pct']*100:.2f}% (${pos['net_pnl_quote']:.2f})"
                    f"\n- Position ID: {pos['executor_id']}"
                )
                prompt_parts.append("\n**Action Required?** Evaluate if this position should be closed based on:")
                prompt_parts.append("- Has profit target been reached?")
                prompt_parts.append("- Is stop loss triggered?")
                prompt_parts.append("- Is the thesis still valid?")
        else:
            prompt_parts.append(f"\n# CURRENT POSITIONS")
            prompt_parts.append(f"No active positions. You may open new positions if good opportunities exist.")
        
        # 3. 市场数据
        prompt_parts.append(f"\n# MARKET DATA")
        
        # 统计有多少数据不完整
        limited_data_count = sum(1 for data in context["market_data"].values() 
                                 if not "error" in data and data.get("candles_available", 100) < 20)
        
        if limited_data_count > 0:
            prompt_parts.append(f"\n⚠️  **Data Limitation Notice**: {limited_data_count} pair(s) have limited historical data (<20 candles).")
            prompt_parts.append("Technical indicators may be less reliable. Consider waiting or using caution.\n")
        
        for symbol, data in context["market_data"].items():
            if "error" in data:
                # 🔧 显示有错误的交易对
                prompt_parts.append(f"\n## {symbol}")
                prompt_parts.append(f"⚠️  Data unavailable: {data['error']}")
                continue
            
            candles_available = data.get("candles_available", 100)
            data_warning = f" (⚠️ Limited: {candles_available} candles)" if candles_available < 20 else ""
            
            funding_info = context["funding_rates"].get(symbol, {})
            funding_rate = funding_info.get("rate", 0.0)
            
            # 🔧 修复：先格式化各个值
            ema_str = f"${data['ema_20']:.2f}" if data['ema_20'] is not None else "N/A (insufficient data)"
            rsi_str = f"{data['rsi']:.1f}" if data['rsi'] is not None else "N/A (insufficient data)"
            macd_str = f"{data['macd']:.2f}" if data['macd'] is not None else "N/A (insufficient data)"
            macd_signal_str = f"{data['macd_signal']:.2f}" if data['macd_signal'] is not None else "N/A (insufficient data)"
            
            # 🔑 格式化历史趋势数据
            price_hist = data.get('price_history', [])
            rsi_hist = data.get('rsi_history', [])
            macd_hist = data.get('macd_history', [])
            volume_hist = data.get('volume_history', [])
            
            prompt_parts.append(
                f"\n## {symbol}{data_warning}"
                f"\n**Price & Trend:**"
                f"\n- Current Price: ${data['current_price']:.2f}"
            )
            
            # 显示价格趋势（最近 5 个值）
            if price_hist and len(price_hist) >= 2:
                price_trend_str = " → ".join([f"${p:.2f}" for p in price_hist])
                price_change = ((price_hist[-1] - price_hist[0]) / price_hist[0]) * 100
                trend_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
                prompt_parts.append(f"- Recent Price Trend: {price_trend_str} {trend_emoji} ({price_change:+.2f}%)")
            
            prompt_parts.append(
                f"- 24h Change: {data['price_change_24h_pct']:.2f}%"
                f"\n- EMA(20): {ema_str}"
            )
            
            if data['ema_20']:
                if data['current_price'] > data['ema_20']:
                    prompt_parts.append(f"- Trend: UPTREND (Price > EMA)")
                else:
                    prompt_parts.append(f"- Trend: DOWNTREND (Price < EMA)")
            
            # 🔑 新增：成交量信息
            prompt_parts.append(f"\n**Volume:**")
            if volume_hist and len(volume_hist) >= 2:
                # 格式化成交量（使用 K, M, B 单位）
                def format_volume(vol):
                    if vol >= 1e9:
                        return f"{vol/1e9:.2f}B"
                    elif vol >= 1e6:
                        return f"{vol/1e6:.2f}M"
                    elif vol >= 1e3:
                        return f"{vol/1e3:.2f}K"
                    else:
                        return f"{vol:.2f}"
                
                volume_trend_str = " → ".join([format_volume(v) for v in volume_hist])
                
                # 计算成交量变化
                volume_change = ((volume_hist[-1] - volume_hist[0]) / volume_hist[0]) * 100 if volume_hist[0] > 0 else 0
                
                # 判断成交量趋势
                if volume_change > 20:
                    volume_emoji = "📈"
                    volume_trend = "Increasing (strong interest)"
                elif volume_change < -20:
                    volume_emoji = "📉"
                    volume_trend = "Decreasing (weak interest)"
                else:
                    volume_emoji = "➡️"
                    volume_trend = "Stable"
                
                prompt_parts.append(f"- Recent Volume: {volume_trend_str} {volume_emoji}")
                prompt_parts.append(f"  → {volume_trend}")
                
                # 比较当前成交量与平均值
                current_vol = volume_hist[-1]
                avg_vol = data.get('avg_volume', current_vol)
                if current_vol > avg_vol * 1.5:
                    prompt_parts.append(f"  → Above average (strong confirmation)")
                elif current_vol < avg_vol * 0.5:
                    prompt_parts.append(f"  → Below average (weak confirmation)")
                else:
                    prompt_parts.append(f"  → Near average")
            
            prompt_parts.append(f"\n**Technical Indicators:**")
            
            # RSI 及其趋势
            prompt_parts.append(f"- RSI: {rsi_str}")
            if rsi_hist and len(rsi_hist) >= 2:
                rsi_trend_str = " → ".join([f"{r:.1f}" for r in rsi_hist])
                rsi_change = rsi_hist[-1] - rsi_hist[0]
                momentum = "📈 Rising" if rsi_change > 5 else "📉 Falling" if rsi_change < -5 else "➡️ Stable"
                prompt_parts.append(f"  Trend: {rsi_trend_str} ({momentum})")
            
            if data['rsi']:
                if data['rsi'] > 70:
                    prompt_parts.append(f"  → Overbought (potential reversal or strong trend)")
                elif data['rsi'] < 30:
                    prompt_parts.append(f"  → Oversold (potential reversal)")
                else:
                    prompt_parts.append(f"  → Neutral")
            
            # MACD 及其趋势
            prompt_parts.append(f"- MACD: {macd_str}")
            prompt_parts.append(f"- MACD Signal: {macd_signal_str}")
            
            if macd_hist and len(macd_hist) >= 2:
                macd_trend_str = " → ".join([f"{m:.2f}" for m in macd_hist])
                macd_change = macd_hist[-1] - macd_hist[0]
                momentum = "📈 Strengthening" if macd_change > 0 else "📉 Weakening" if macd_change < 0 else "➡️ Stable"
                prompt_parts.append(f"  Trend: {macd_trend_str} ({momentum})")
            
            if data['macd'] and data['macd_signal']:
                if data['macd'] > data['macd_signal']:
                    prompt_parts.append(f"  → Bullish momentum (MACD > Signal)")
                else:
                    prompt_parts.append(f"  → Bearish momentum (MACD < Signal)")
            
            # 显示 funding rate（回测和实盘都支持）
            if "_perpetual" in self.config.connector_name:
                if funding_info and funding_rate != 0.0:
                    # 有真实数据（实盘或回测历史数据）
                    prompt_parts.append(
                        f"\n**Funding Rate:** {funding_rate*100:.4f}% (8h)"
                    )
                    if funding_rate > 0.0001:
                        prompt_parts.append(f"  → Bullish sentiment (longs paying shorts)")
                    elif funding_rate < -0.0001:
                        prompt_parts.append(f"  → Bearish sentiment (shorts paying longs)")
                    else:
                        prompt_parts.append(f"  → Neutral sentiment")
                elif funding_info:
                    # 有 funding_info 但 rate 为 0（可能是真实的 0 或默认值）
                    prompt_parts.append(
                        f"\n**Funding Rate:** {funding_rate*100:.4f}% (8h)"
                    )
                    prompt_parts.append(f"  → Neutral sentiment")
                else:
                    # 完全没有数据
                    prompt_parts.append(f"\n**Funding Rate:** Not available")
        
        # 4. 历史交易记录
        if context["recent_trades"]:
            prompt_parts.append(f"\n# RECENT TRADES (Last {len(context['recent_trades'])})")
            prompt_parts.append("Learn from these trades to improve your strategy:\n")
            
            for trade in context["recent_trades"]:
                # 计算持仓时长
                duration_hours = 0
                if trade['close_timestamp'] and trade['timestamp']:
                    duration_hours = (trade['close_timestamp'] - trade['timestamp']) / 3600
                
                # 格式化交易信息
                pnl_emoji = "✅ PROFIT" if trade['pnl_quote'] > 0 else "❌ LOSS"
                prompt_parts.append(
                    f"- **{trade['symbol']} {trade['side']}**: "
                    f"Entry ${trade['entry_price']:.2f} → Exit ${trade['exit_price']:.2f}, "
                    f"PnL: {trade['pnl_pct']*100:.2f}% (${trade['pnl_quote']:.2f}) {pnl_emoji}, "
                    f"Close Reason: {trade['close_type']}, "
                    f"Duration: {duration_hours:.1f}h"
                )
        
        # 5. 决策指令 - 简化版
        prompt_parts.append(f"\n\nProvide your decision in JSON format.")
        
        return "\n".join(prompt_parts)
    
    async def _call_langchain_llm(self, system_prompt: str, user_prompt: str) -> str:
        """使用 LangChain 调用 LLM"""
        if self.llm is None:
            raise RuntimeError("LangChain LLM not initialized")
        
        try:
            start_time = time.time()
            
            # 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger().debug(f"Sending request to LLM ({self.config.llm_model})...")
            
            # 使用 ainvoke 异步调用
            response = await self.llm.ainvoke(messages)
            
            elapsed = time.time() - start_time
            content = response.content
            
            self.logger().info(f"LLM response received in {elapsed:.2f}s, length: {len(content)} chars")
            self.logger().debug(f"LLM Response preview: {content[:300]}...")
            
            return content
            
        except Exception as e:
            self.logger().error(f"LangChain LLM call failed: {e}", exc_info=True)
            raise
    
    def _parse_ai_response(self, response: str) -> List[Dict]:
        """解析 AI 返回的 JSON 决策"""
        try:
            self.logger().debug("Parsing LLM response...")
            
            # 尝试提取 JSON（可能被包裹在 ```json ``` 中）
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                self.logger().debug("Found JSON in ```json``` block")
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                self.logger().debug("Found JSON in ``` block")
            else:
                json_str = response.strip()
                self.logger().debug("Using raw response as JSON")
            
            self.logger().debug(f"JSON string to parse: {json_str[:200]}...")
            
            decisions = json.loads(json_str)
            
            if not isinstance(decisions, list):
                self.logger().warning("AI response is not a list, wrapping it")
                decisions = [decisions] if decisions else []
            
            # 🔧 修复：兼容多种字段名和值的变体
            normalized_decisions = []
            for dec in decisions:
                # 1. 规范化 action 字段名
                if "decision" in dec and "action" not in dec:
                    dec["action"] = dec.pop("decision")
                    self.logger().debug(f"Normalized 'decision' field to 'action': {dec.get('action')}")
                
                # 2. 规范化 action 值（统一小写，处理各种变体）
                action = dec.get("action", "").lower().replace("/", "_").replace(" ", "_")
                
                # 映射各种变体到标准 action
                action_mapping = {
                    "wait": "hold",
                    "hold_wait": "hold",
                    "do_nothing": "hold",
                    "long": "open_long",
                    "short": "open_short",
                    "buy": "open_long",
                    "sell": "open_short",
                    "close": "close_position",
                    "exit": "close_position",
                }
                
                if action in action_mapping:
                    original_action = dec.get("action")
                    dec["action"] = action_mapping[action]
                    self.logger().debug(f"Normalized action: '{original_action}' -> '{dec['action']}'")
                else:
                    dec["action"] = action
                
                # 3. 确保 reasoning 是字符串
                reasoning = dec.get("reasoning", "")
                if isinstance(reasoning, dict):
                    # 如果 reasoning 是字典，转换为简洁的文本
                    reasoning_parts = []
                    for key, value in reasoning.items():
                        if isinstance(value, dict):
                            # 进一步展开嵌套字典
                            for k, v in value.items():
                                reasoning_parts.append(f"{k}: {v}")
                        else:
                            reasoning_parts.append(f"{key}: {value}")
                    dec["reasoning"] = ". ".join(reasoning_parts)
                    self.logger().debug(f"Converted dict reasoning to string: {dec['reasoning'][:100]}...")
                
                normalized_decisions.append(dec)
            
            self.logger().info(f"Successfully parsed {len(normalized_decisions)} decisions from LLM response")
            
            # 打印每个决策的基本信息
            for i, dec in enumerate(normalized_decisions, 1):
                action = dec.get("action", "unknown")
                symbol = dec.get("symbol", "N/A")
                reasoning_preview = str(dec.get("reasoning", ""))[:50]
                self.logger().debug(f"Decision {i}: action={action}, symbol={symbol}, reasoning={reasoning_preview}...")
            
            return normalized_decisions
            
        except json.JSONDecodeError as e:
            self.logger().error(f"Failed to parse AI response as JSON: {e}")
            self.logger().error(f"Raw response (first 500 chars): {response[:500]}")
            return []
    
    def _validate_decisions(self, decisions: List[Dict], context: Dict) -> List[Dict]:
        """验证决策的合法性"""
        self.logger().info(f"Validating {len(decisions)} decisions...")
        
        validated = []
        current_positions = len(context["positions"])
        
        self.logger().debug(f"Current positions: {current_positions}, Max allowed: {self.config.max_concurrent_positions}")
        
        for i, decision in enumerate(decisions, 1):
            # 🔧 修复：先检查 reasoning（必须字段）
            reasoning = decision.get("reasoning", "")
            if not reasoning:
                self.logger().warning(f"❌ Decision {i}: missing reasoning field - {decision}")
                continue
            
            action = decision.get("action")
            symbol = decision.get("symbol")
            
            self.logger().debug(f"Validating decision {i}: {action} {symbol}")
            
            # 🔧 修复：reasoning 可能是字符串或字典
            if isinstance(reasoning, dict):
                reasoning_preview = str(reasoning)[:100]
            elif isinstance(reasoning, str):
                reasoning_preview = reasoning[:100]
            else:
                reasoning_preview = str(reasoning)[:100]
            
            self.logger().debug(f"   Reasoning: {reasoning_preview}...")
            
            # hold 动作不需要 symbol
            if action == "hold":
                validated.append(decision)
                self.logger().info(f"✅ Decision {i}: HOLD - {reasoning_preview[:50]}...")
                continue
            
            # 其他动作需要 symbol
            if not symbol:
                self.logger().warning(f"❌ Decision {i}: missing symbol for action {action}")
                continue
            
            # 检查交易对是否在配置中
            if symbol not in self.config.trading_pairs:
                self.logger().warning(f"❌ Decision {i}: symbol {symbol} not in configured pairs {self.config.trading_pairs}")
                continue
            
            # 检查仓位数量限制
            if action in ["open_long", "open_short"]:
                if current_positions >= self.config.max_concurrent_positions:
                    self.logger().warning(f"❌ Decision {i}: max positions ({self.config.max_concurrent_positions}) reached, skipping {action} for {symbol}")
                    continue
                
                # 检查止损止盈
                stop_loss_pct = decision.get("stop_loss_pct", 0.02)
                take_profit_pct = decision.get("take_profit_pct", 0.04)
                confidence = decision.get("confidence", 50)
                
                self.logger().debug(f"   SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%, Confidence: {confidence}%")
                
                # 验证风险回报比
                if take_profit_pct < stop_loss_pct * 1.5:
                    self.logger().warning(f"⚠️  Decision {i}: R/R ratio too low for {symbol}, adjusting TP from {take_profit_pct*100:.1f}% to {stop_loss_pct*200:.1f}%")
                    decision["take_profit_pct"] = stop_loss_pct * 2
                
                current_positions += 1
                self.logger().debug(f"   ✅ Decision {i} validated, would be position #{current_positions}")
            
            validated.append(decision)
        
        self.logger().info(f"✅ Validation complete: {len(validated)}/{len(decisions)} decisions passed")
        
        if len(validated) < len(decisions):
            self.logger().warning(f"⚠️  {len(decisions) - len(validated)} decisions were rejected")
        
        return validated
    
    def _get_current_price(self, trading_pair: str) -> Optional[Decimal]:
        """
        获取当前价格 (支持回测和实盘)
        
        回测时从 K 线数据获取最新价格，因为 market_data_provider.prices 只支持单交易对
        """
        try:
            # 方法 1: 尝试从 market_data_provider 获取（实盘）
            price = self.market_data_provider.get_price_by_type(
                self.config.connector_name,
                trading_pair,
                price_type=PriceType.MidPrice
            )
            
            # 如果价格是默认值 1.0，说明回测时没有设置，从 K 线获取
            if price == Decimal("1") or price is None:
                # 方法 2: 从 K 线数据获取最新 close 价格（回测）
                candles_df = self.market_data_provider.get_candles_df(
                    connector_name=self.config.connector_name,
                    trading_pair=trading_pair,
                    interval=self.config.interval,
                    max_records=10  # 获取更多数据以确保有有效值
                )
                
                if not candles_df.empty:
                    # 获取最新 K 线的收盘价
                    latest_close = candles_df.iloc[-1]["close"]
                    price = Decimal(str(latest_close))
                    self.logger().debug(f"Got price from candles for {trading_pair}: {price}")
                else:
                    self.logger().error(f"❌ No candles data available for {trading_pair} - cannot get price!")
                    return None
            
            # 最后检查：确保价格有效
            if price is None or price <= 0:
                self.logger().error(f"❌ Invalid price for {trading_pair}: {price}")
                return None
            
            return price
            
        except Exception as e:
            self.logger().error(f"❌ Failed to get price for {trading_pair}: {e}", exc_info=True)
            return None
    
    async def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        异步版本：根据 AI 决策生成 Executor Actions
        
        ✅ 优势：
        - 不延迟决策（立即执行）
        - 不阻塞事件循环
        - 更准确的回测结果
        
        ⚠️ 要求：需要配合增强版回测引擎 (BacktestingEngineAsync)
        """
        current_time = self.market_data_provider.time()
        
        # 🔧 修复：初始化决策时间
        if self._last_decision_time == 0:
            self._last_decision_time = current_time - self.config.decision_interval
            self.logger().info(
                f"⏱️  Decision timer initialized at timestamp {current_time} "
                f"(interval: {self.config.decision_interval}s)"
            )
        
        # 🔧 修复：检查决策间隔（防止回测时每个tick都决策）
        time_since_last = current_time - self._last_decision_time
        
        if time_since_last < self.config.decision_interval:
            # 未到决策时间，直接返回空列表
            return []
        
        # 🔑 到达决策时间，开始 AI 决策流程
        self.logger().info("=" * 80)
        self.logger().info(
            f"🤖 AI Decision Cycle Triggered "
            f"(time since last: {time_since_last:.0f}s, interval: {self.config.decision_interval}s)"
        )
        self.logger().info("=" * 80)
        
        # ✅ 直接异步执行 AI 决策（不延迟）
        try:
            ai_decisions = await self._execute_ai_decision_cycle()
            
        except Exception as e:
            self.logger().error(f"❌ AI decision cycle failed: {e}", exc_info=True)
            ai_decisions = []
        
        # 更新上次决策时间
        self._last_decision_time = current_time
        
        # 生成 Executor Actions
        if not ai_decisions:
            self.logger().warning("⚠️  No AI decisions generated")
            return []
        
        actions = []
        
        for i, decision in enumerate(ai_decisions, 1):
            action_type = decision.get("action")
            symbol = decision.get("symbol")
            
            self.logger().debug(f"Processing decision {i}/{len(ai_decisions)}: {action_type} {symbol}")
            
            try:
                if action_type == "open_long":
                    action = self._create_open_action(decision, TradeType.BUY)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created LONG action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create LONG action for {symbol}")
                        
                elif action_type == "open_short":
                    action = self._create_open_action(decision, TradeType.SELL)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created SHORT action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create SHORT action for {symbol}")
                        
                elif action_type == "close_position":
                    action = self._create_close_action(decision)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created CLOSE action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create CLOSE action for {symbol}")
                        
                elif action_type == "hold":
                    # Hold 动作不需要创建任何 action，只记录日志
                    reasoning = decision.get("reasoning", "No reasoning provided")
                    self.logger().info(f"   ⏸️  HOLD decision: {reasoning}")
                    
                else:
                    self.logger().warning(f"   ⚠️  Unknown action type: {action_type}")
                        
            except Exception as e:
                self.logger().error(f"   ❌ Error creating action for {symbol}: {e}", exc_info=True)
        
        self.logger().info(f"📋 Generated {len(actions)} executor actions from {len(ai_decisions)} decisions")
        self.logger().info("=" * 80)
        
        # 过滤掉 None（安全检查）
        return [action for action in actions if action is not None]
    
    
    async def _execute_ai_decision_cycle(self) -> List[Dict]:
        """
        执行完整的 AI 决策流程（异步版本）
        """
        try:
            # 增加决策轮次计数
            self._decision_cycle_count += 1
            
            # Step 1: 构建上下文
            self.logger().info("📊 Building trading context...")
            context = await self._build_trading_context()
            self.logger().info(
                f"   ✅ Context: {len(context['market_data'])} pairs, "
                f"{len(context['positions'])} positions"
            )
            
            # Step 2: 构建 Prompts
            self.logger().debug("Building prompts...")
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context)
            
            # 🔑 Step 2.5: 如果在回测环境，记录决策开始日志
            if self._backtest_run_id:
                await self._log_decision_start(context, system_prompt, user_prompt)
            
            # Step 3: 调用 LLM
            self.logger().info("🧠 Calling LLM for decisions...")
            decisions = await self._get_ai_decisions(context)
            self.logger().info(f"   ✅ LLM returned {len(decisions)} decisions")
            
            # Step 4: 验证决策
            self.logger().debug("Validating decisions...")
            validated = self._validate_decisions(decisions, context)
            self.logger().info(f"   ✅ Validated {len(validated)}/{len(decisions)} decisions")
            
            # 🔑 Step 5: 如果在回测环境，记录决策结果日志
            if self._backtest_run_id:
                await self._log_decision_result(context, system_prompt, user_prompt, validated)
            
            return validated
            
        except Exception as e:
            self.logger().error(f"❌ Decision cycle failed: {e}", exc_info=True)
            
            # 🔑 记录错误到回测日志
            if self._backtest_run_id:
                await self._log_decision_error(str(e))
            
            return []
    
    async def _log_decision_start(self, context: Dict, system_prompt: str, user_prompt: str):
        """记录决策开始的日志（仅回测环境）"""
        try:
            # 导入必要的模块
            from database import AsyncDatabaseManager, BacktestRunRepository
            
            db_manager = AsyncDatabaseManager(self._backtest_db_url)
            
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                
                # 构建简要信息
                account = context.get('account', {})
                positions = context.get('positions', [])
                
                log_message = (
                    f"🔄 AI Decision Cycle #{self._decision_cycle_count} Started\n\n"
                    f"💰 Account: ${account.get('current_holdings', 0):.2f} "
                    f"(PnL: ${account.get('total_pnl', 0):.2f}, "
                    f"Closed: ${account.get('closed_pnl', 0):.2f}, "
                    f"Active: ${account.get('active_pnl', 0):.2f})\n"
                    f"📊 Positions: {len(positions)}/{self.config.max_concurrent_positions}\n"
                )
                
                for pos in positions:
                    log_message += (
                        f"  - {pos['symbol']} {pos['side']}: "
                        f"Entry ${pos['entry_price']:.2f}, "
                        f"PnL {pos['net_pnl_pct']*100:.2f}% (${pos['net_pnl_quote']:.2f})\n"
                    )
                
                await repo.bulk_create_logs(
                    run_id=self._backtest_run_id,
                    logs=[{
                        "log_level": "INFO",
                        "log_message": log_message,
                        "log_category": "AI_DECISION"
                    }]
                )
            
            await db_manager.close()
            
        except Exception as e:
            self.logger().warning(f"Failed to log decision start: {e}")
    
    async def _log_decision_result(self, context: Dict, system_prompt: str, user_prompt: str, decisions: List[Dict]):
        """记录决策结果到数据库（仅回测环境）"""
        try:
            from database import AsyncDatabaseManager, BacktestRunRepository
            
            db_manager = AsyncDatabaseManager(self._backtest_db_url)
            
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                
                logs_to_create = []
                
                # 1. 记录决策摘要
                log_message = f"✅ AI Decision Cycle #{self._decision_cycle_count} Completed\n\n"
                
                if decisions:
                    log_message += f"🤖 Decisions ({len(decisions)}):\n"
                    for i, dec in enumerate(decisions, 1):
                        action = dec.get('action', 'unknown')
                        symbol = dec.get('symbol', 'N/A')
                        
                        log_message += f"\n[{i}] Action: {action.upper()}"
                        if symbol != 'N/A':
                            log_message += f" on {symbol}"
                        
                        if action in ['open_long', 'open_short']:
                            sl = dec.get('stop_loss_pct', 0) * 100
                            tp = dec.get('take_profit_pct', 0) * 100
                            conf = dec.get('confidence', 0)
                            log_message += f"\n    SL: {sl:.1f}%, TP: {tp:.1f}%, Confidence: {conf}%"
                        
                        reasoning = dec.get('reasoning', 'No reasoning')
                        reasoning_preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                        log_message += f"\n    Reasoning: {reasoning_preview}\n"
                else:
                    log_message += "🤖 Decisions: HOLD (no actions taken)\n"
                
                logs_to_create.append({
                    "log_level": "INFO",
                    "log_message": log_message,
                    "log_category": "AI_DECISION"
                })
                
                # 2. 记录 System Prompt（DEBUG 级别，方便分析）
                logs_to_create.append({
                    "log_level": "DEBUG",
                    "log_message": f"📋 System Prompt (Cycle #{self._decision_cycle_count}):\n\n{system_prompt}",
                    "log_category": "AI_PROMPT"
                })
                
                # 3. 记录 User Prompt（DEBUG 级别）
                logs_to_create.append({
                    "log_level": "DEBUG",
                    "log_message": f"📊 User Prompt (Cycle #{self._decision_cycle_count}):\n\n{user_prompt}",
                    "log_category": "AI_PROMPT"
                })
                
                # 批量写入
                await repo.bulk_create_logs(
                    run_id=self._backtest_run_id,
                    logs=logs_to_create
                )
            
            await db_manager.close()
            self.logger().debug(f"✅ Logged decision cycle #{self._decision_cycle_count} to database")
            
        except Exception as e:
            self.logger().warning(f"Failed to log decision result: {e}")
    
    async def _log_decision_error(self, error_message: str):
        """记录决策错误（仅回测环境）"""
        try:
            from database import AsyncDatabaseManager, BacktestRunRepository
            
            db_manager = AsyncDatabaseManager(self._backtest_db_url)
            
            async with db_manager.get_session_context() as session:
                repo = BacktestRunRepository(session)
                
                await repo.bulk_create_logs(
                    run_id=self._backtest_run_id,
                    logs=[{
                        "log_level": "ERROR",
                        "log_message": f"❌ AI Decision Cycle #{self._decision_cycle_count} Failed: {error_message}",
                        "log_category": "ERROR"
                    }]
                )
            
            await db_manager.close()
            
        except Exception as e:
            self.logger().warning(f"Failed to log decision error: {e}")
    
    def _create_open_action(self, decision: Dict, trade_type: TradeType) -> Optional[CreateExecutorAction]:
        """创建开仓 Action"""
        symbol = decision["symbol"]
        
        self.logger().info(f"🔍 Attempting to create {trade_type.name} action for {symbol}...")
        
        # 获取当前价格（用于计算仓位大小）
        # ⚠️  Workaround: 回测引擎不支持多交易对，需要从 K 线数据获取价格
        price = self._get_current_price(symbol)
        
        if price is None or price <= 0:
            self.logger().error(f"❌ Cannot get valid price for {symbol}, got {price} - SKIPPING this trade!")
            return None
        
        self.logger().info(f"   ✅ Got price for {symbol}: ${price:.2f}")
        
        # 计算仓位大小
        position_size_quote = self.config.total_amount_quote * self.config.single_position_size_pct
        amount = position_size_quote / price
        
        self.logger().debug(f"   Position size: ${position_size_quote:.2f} = {amount:.6f} {symbol.split('-')[0]}")
        
        # 止损止盈
        stop_loss_pct = Decimal(str(decision.get("stop_loss_pct", 0.02)))
        take_profit_pct = Decimal(str(decision.get("take_profit_pct", 0.04)))
        
        # 创建 Triple Barrier Config
        triple_barrier = self.config.triple_barrier_config.copy()
        triple_barrier.stop_loss = stop_loss_pct
        triple_barrier.take_profit = take_profit_pct
        
        # 确保 price 不是 None（双重检查）
        if price is None:
            self.logger().error(f"❌ CRITICAL: price became None after validation! Symbol: {symbol}")
            return None
        
        # ⚠️  重要：在回测中必须提供 entry_price，使用当前市价
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=symbol,
            side=trade_type,
            entry_price=price,  # 使用当前市价作为入场价（回测必需）
            amount=amount,
            triple_barrier_config=triple_barrier,
            leverage=self.config.leverage,
        )
        
        self.logger().info(
            f"📈 Creating {trade_type.name} position for {symbol} @ ${price:.2f}, "
            f"Amount: {amount:.4f}, SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%"
        )
        
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=executor_config
        )
    
    def _create_close_action(self, decision: Dict) -> Optional[StopExecutorAction]:
        """创建平仓 Action"""
        executor_id = decision.get("executor_id")
        symbol = decision.get("symbol")
        
        # 查找对应的 Executor
        target_executor = None
        
        if executor_id:
            # 通过 ID 查找
            for executor in self.executors_info:
                if executor.id == executor_id and executor.is_active:
                    target_executor = executor
                    break
        else:
            # 通过 Symbol 查找
            for executor in self.executors_info:
                if executor.config.trading_pair == symbol and executor.is_active:
                    target_executor = executor
                    break
        
        if not target_executor:
            self.logger().warning(f"Cannot find active executor for {symbol}")
            return None
        
        self.logger().info(f"📉 Closing position for {symbol}, Executor ID: {target_executor.id}")
        
        return StopExecutorAction(
            controller_id=self.config.id,
            executor_id=target_executor.id
        )
    
    def to_format_status(self) -> List[str]:
        """格式化状态显示"""
        lines = []
        
        lines.append(f"🤖 AI Agent V1 Status")
        lines.append(f"=" * 50)
        
        # 监控币种
        lines.append(f"Monitoring Pairs: {', '.join(self.config.trading_pairs)}")
        
        # 持仓情况
        active_positions = [e for e in self.executors_info if e.is_active and e.is_trading]
        lines.append(f"Active Positions: {len(active_positions)}/{self.config.max_concurrent_positions}")
        
        for executor in active_positions:
            lines.append(
                f"  - {executor.config.trading_pair} {executor.config.side.name}: "
                f"PnL {executor.net_pnl_pct*100:.2f}% (${executor.net_pnl_quote:.2f})"
            )
        
        # 最近决策时间
        if self._last_decision_time > 0:
            time_since_last = self.market_data_provider.time() - self._last_decision_time
            lines.append(f"Last Decision: {int(time_since_last)}s ago")
            next_decision_in = max(0, self.config.decision_interval - time_since_last)
            lines.append(f"Next Decision: in {int(next_decision_in)}s")
        
        # 历史统计（从 executors_info 获取）
        closed_executors = [e for e in self.executors_info if hasattr(e, 'status') and str(e.status) == 'RunnableStatus.TERMINATED']
        if closed_executors:
            total_trades = len(closed_executors)
            winning_trades = sum(1 for e in closed_executors if hasattr(e, 'net_pnl_quote') and float(e.net_pnl_quote) > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            lines.append(f"Trade History: {total_trades} trades, Win Rate: {win_rate:.1f}%")
        
        return lines

