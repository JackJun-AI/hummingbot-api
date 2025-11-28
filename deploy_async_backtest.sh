#!/bin/bash

# Deployment Script for Async Backtesting
# This script helps deploy and test the new async backtesting features

set -e

echo "=========================================="
echo "Hummingbot API - Async Backtesting Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to hummingbot-api directory
cd "$(dirname "$0")"

# 1. Check if Docker is running
echo -e "\n${YELLOW}[1/6]${NC} Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# 2. Check if PostgreSQL container is running
echo -e "\n${YELLOW}[2/6]${NC} Checking PostgreSQL container..."
if docker ps | grep -q "hummingbot-api-db"; then
    echo -e "${GREEN}✅ PostgreSQL container is running${NC}"
else
    echo -e "${YELLOW}⚠️  PostgreSQL container not found. Starting services...${NC}"
    docker-compose up -d
    echo "Waiting 10 seconds for PostgreSQL to initialize..."
    sleep 10
fi

# 3. Check if API container is running
echo -e "\n${YELLOW}[3/6]${NC} Checking API container..."
if docker ps | grep -q "hummingbot-api"; then
    echo -e "${GREEN}✅ API container is running${NC}"
else
    echo -e "${YELLOW}⚠️  API container not found. Starting services...${NC}"
    docker-compose up -d
    echo "Waiting 10 seconds for API to start..."
    sleep 10
fi

# 4. Apply database migrations
echo -e "\n${YELLOW}[4/6]${NC} Applying database migrations..."
echo "The new tables (backtest_runs, backtest_logs, backtest_trades) will be created automatically on API startup."
echo -e "${GREEN}✅ Migrations will be applied automatically${NC}"

# 5. Restart API to ensure new code is loaded
echo -e "\n${YELLOW}[5/6]${NC} Restarting API container..."
docker-compose restart hummingbot-api
echo "Waiting 5 seconds for API to restart..."
sleep 5
echo -e "${GREEN}✅ API restarted${NC}"

# 6. Test API health
echo -e "\n${YELLOW}[6/6]${NC} Testing API health..."
if curl -s -u admin:admin http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is responding${NC}"
else
    echo -e "${RED}❌ API is not responding. Please check logs with: docker-compose logs hummingbot-api${NC}"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "📚 New API Endpoints Available:"
echo "   POST   /backtesting/start          - Start async backtest"
echo "   GET    /backtesting/status/{id}    - Get backtest status"
echo "   GET    /backtesting/results/{id}   - Get complete results"
echo "   POST   /backtesting/stop/{id}      - Stop running backtest"
echo "   GET    /backtesting/list           - List all backtests"
echo ""
echo "📖 Documentation:"
echo "   - See ASYNC_BACKTESTING.md for full API documentation"
echo "   - Swagger UI: http://localhost:8000/docs"
echo ""
echo "🧪 Run Tests:"
echo "   python test_async_backtest.py          # Test full workflow"
echo "   python test_async_backtest.py stop     # Test stop functionality"
echo ""
echo "🔍 Check Logs:"
echo "   docker-compose logs -f hummingbot-api  # API logs"
echo "   docker-compose logs -f postgres        # Database logs"
echo ""
echo "🗄️  Check Database:"
echo "   docker exec -it hummingbot-api-db psql -U hbot -d hummingbot_api"
echo "   Then run: SELECT * FROM backtest_runs;"
echo ""

