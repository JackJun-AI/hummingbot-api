#!/usr/bin/env python3
"""
Test script for async backtesting API.
"""

import requests
import time
from requests.auth import HTTPBasicAuth

# Configuration
BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin"

auth = HTTPBasicAuth(USERNAME, PASSWORD)


def test_async_backtesting():
    """Test the complete async backtesting workflow."""
    
    print("=" * 80)
    print("Testing Async Backtesting API")
    print("=" * 80)
    
    # 1. Start a backtest
    print("\n1. Starting backtest...")
    start_response = requests.post(
        f"{BASE_URL}/backtesting/start",
        json={
            "run_name": "Test AI Agent Backtest",
            "start_time": 1735689600,  # 2025-01-01
            "end_time": 1735776000,    # 2025-01-02 (1 day)
            "backtesting_resolution": "5m",
            "trade_cost": 0.0006,
            "config": "ai_agent_backtest.yml"  # Path to controller config
        },
        auth=auth
    )
    
    if start_response.status_code != 200:
        print(f"❌ Failed to start backtest: {start_response.text}")
        return
    
    start_data = start_response.json()
    run_id = start_data["run_id"]
    print(f"✅ Backtest started!")
    print(f"   Run ID: {run_id}")
    print(f"   Status: {start_data['status']}")
    
    # 2. Poll status until completion
    print("\n2. Polling status...")
    max_polls = 60  # Max 5 minutes (60 * 5 seconds)
    poll_count = 0
    
    while poll_count < max_polls:
        time.sleep(5)
        poll_count += 1
        
        status_response = requests.get(
            f"{BASE_URL}/backtesting/status/{run_id}",
            auth=auth
        )
        
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.text}")
            break
        
        status_data = status_response.json()
        current_status = status_data["status"]
        
        print(f"   [{poll_count}] Status: {current_status}")
        
        if current_status in ["COMPLETED", "FAILED", "CANCELLED"]:
            print(f"\n✅ Backtest finished with status: {current_status}")
            
            if current_status == "FAILED":
                print(f"   Error: {status_data.get('error_message', 'Unknown error')}")
            
            break
    else:
        print(f"\n⚠️  Backtest still running after {max_polls * 5} seconds")
        print("   You can check results later using the run_id")
        return
    
    # 3. Get complete results
    if status_data["status"] == "COMPLETED":
        print("\n3. Fetching complete results...")
        
        results_response = requests.get(
            f"{BASE_URL}/backtesting/results/{run_id}?include_logs=true&log_limit=10",
            auth=auth
        )
        
        if results_response.status_code != 200:
            print(f"❌ Failed to get results: {results_response.text}")
            return
        
        results_data = results_response.json()
        run_info = results_data["run_info"]
        trades = results_data["trades"]
        logs = results_data["logs"]
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Run Name: {run_info['run_name']}")
        print(f"   Controller: {run_info['controller_name']}")
        print(f"   Total Trades: {run_info['total_trades']}")
        print(f"   Win Rate: {run_info['win_rate']:.2%}" if run_info['win_rate'] else "   Win Rate: N/A")
        print(f"   Net PnL (Quote): {run_info['net_pnl_quote']:.2f}" if run_info['net_pnl_quote'] else "   Net PnL: N/A")
        print(f"   Net PnL (%): {run_info['net_pnl_pct']:.4%}" if run_info['net_pnl_pct'] else "   Net PnL %: N/A")
        print(f"   Max Drawdown: {run_info['max_drawdown']:.2f}" if run_info['max_drawdown'] else "   Max Drawdown: N/A")
        print(f"   Sharpe Ratio: {run_info['sharpe_ratio']:.2f}" if run_info['sharpe_ratio'] else "   Sharpe Ratio: N/A")
        
        print(f"\n📈 Trades: ({len(trades)} total)")
        for i, trade in enumerate(trades[:5]):  # Show first 5 trades
            print(f"   [{i+1}] {trade['side']} {trade['trading_pair']}")
            print(f"       Entry: {trade['entry_price']:.2f} @ {trade['entry_timestamp']}")
            print(f"       PnL: {trade['net_pnl_pct']:.4%} ({trade['net_pnl_quote']:.2f} USDT)")
            print(f"       Close Type: {trade['close_type']}")
        
        if len(trades) > 5:
            print(f"   ... and {len(trades) - 5} more trades")
        
        print(f"\n📝 Recent Logs: ({len(logs)} shown)")
        for log in logs[:5]:
            level_icon = {
                "INFO": "ℹ️ ",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
                "DEBUG": "🔍"
            }.get(log['log_level'], "  ")
            print(f"   {level_icon} [{log['log_level']}] {log['log_message']}")
        
        print("\n" + "=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)
    
    # 4. List all backtests
    print("\n4. Listing recent backtests...")
    list_response = requests.get(
        f"{BASE_URL}/backtesting/list?limit=5",
        auth=auth
    )
    
    if list_response.status_code != 200:
        print(f"❌ Failed to list backtests: {list_response.text}")
        return
    
    list_data = list_response.json()
    print(f"\n📋 Recent Backtests ({list_data['total']} total):")
    
    for run in list_data["runs"]:
        status_icon = {
            "COMPLETED": "✅",
            "FAILED": "❌",
            "CANCELLED": "⛔",
            "RUNNING": "🔄",
            "PENDING": "⏳"
        }.get(run["status"], "  ")
        
        print(f"\n   {status_icon} {run['run_name']}")
        print(f"      Run ID: {run['run_id']}")
        print(f"      Status: {run['status']}")
        print(f"      Created: {run['created_at']}")
        if run['total_trades']:
            print(f"      Trades: {run['total_trades']}, PnL: {run['net_pnl_pct']:.4%}" if run['net_pnl_pct'] else f"      Trades: {run['total_trades']}")


def test_stop_backtest():
    """Test stopping a running backtest."""
    print("\n" + "=" * 80)
    print("Testing Stop Backtest")
    print("=" * 80)
    
    # Start a backtest
    print("\n1. Starting backtest...")
    start_response = requests.post(
        f"{BASE_URL}/backtesting/start",
        json={
            "run_name": "Test Stop Backtest",
            "start_time": 1735689600,
            "end_time": 1738368000,  # Long period (1 month)
            "backtesting_resolution": "1m",
            "trade_cost": 0.0006,
            "config": "ai_agent_backtest.yml"
        },
        auth=auth
    )
    
    if start_response.status_code != 200:
        print(f"❌ Failed to start backtest: {start_response.text}")
        return
    
    run_id = start_response.json()["run_id"]
    print(f"✅ Backtest started with run_id: {run_id}")
    
    # Wait a bit to ensure it's running
    print("\n2. Waiting 10 seconds for backtest to start running...")
    time.sleep(10)
    
    # Stop the backtest
    print("\n3. Stopping backtest...")
    stop_response = requests.post(
        f"{BASE_URL}/backtesting/stop/{run_id}",
        auth=auth
    )
    
    if stop_response.status_code != 200:
        print(f"❌ Failed to stop backtest: {stop_response.text}")
        return
    
    stop_data = stop_response.json()
    print(f"✅ Stop signal sent!")
    print(f"   Message: {stop_data['message']}")
    
    # Check final status
    print("\n4. Checking final status...")
    time.sleep(3)
    
    status_response = requests.get(
        f"{BASE_URL}/backtesting/status/{run_id}",
        auth=auth
    )
    
    if status_response.status_code == 200:
        final_status = status_response.json()["status"]
        print(f"   Final Status: {final_status}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        test_stop_backtest()
    else:
        test_async_backtesting()

