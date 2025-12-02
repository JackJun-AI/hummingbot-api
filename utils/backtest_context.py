"""
Backtest Context Variables

使用 contextvars 在异步任务间传递回测上下文信息，避免循环引用。

这个模块被以下模块导入：
- routers/backtesting.py (设置值)
- bots/controllers/*/*.py (读取值)
"""
from contextvars import ContextVar
from typing import Optional

# 🔑 全局 contextvars（线程安全，支持多任务并发）
# 每个异步任务有独立的上下文，不会互相干扰
backtest_run_id_var: ContextVar[Optional[str]] = ContextVar('backtest_run_id', default=None)
backtest_db_url_var: ContextVar[Optional[str]] = ContextVar('backtest_db_url', default=None)

