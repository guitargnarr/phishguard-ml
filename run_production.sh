#!/bin/bash
# PhishGuard ML - Production Server (Dual-Mode)
# Loads both simple and ensemble models
# Usage: ./run_production.sh

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PhishGuard ML - Production Server"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo ""
    echo "Please run setup first:"
    echo "  python3.9 -m venv venv"
    echo "  ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Verify scikit-learn version
SKLEARN_VERSION=$(python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "NOT FOUND")
if [ "$SKLEARN_VERSION" != "1.6.1" ]; then
    echo -e "${YELLOW}⚠️  Warning: scikit-learn version is $SKLEARN_VERSION (expected 1.6.1)${NC}"
    echo "   Models may not load correctly"
    echo ""
fi

# Check if simple models exist
if [ ! -f "models/phishing_clf.pkl" ] || [ ! -f "models/tfidf_vec.pkl" ]; then
    echo -e "${RED}❌ Simple models not found!${NC}"
    echo ""
    echo "Please train models first:"
    echo "  python train_model.py"
    exit 1
fi

echo -e "${GREEN}✓${NC} Simple models found"

# Check if ensemble models exist
if [ -f "models/ensemble/ensemble_model.pkl" ]; then
    echo -e "${GREEN}✓${NC} Ensemble models found"
    ENSEMBLE_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Ensemble models not found (will run in simple-only mode)${NC}"
    echo "   To enable ensemble: python train_ensemble_enhanced.py"
    ENSEMBLE_AVAILABLE=false
fi

echo ""

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
    echo ""
    echo "Kill existing process:"
    echo "  lsof -ti:8000 | xargs kill"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Port 8000 is available"
echo ""

# Start the server
echo "=========================================="
echo "Starting Production Server"
echo "=========================================="
echo ""
if [ "$ENSEMBLE_AVAILABLE" = true ]; then
    echo -e "${BLUE}📦 Modes Available:${NC}"
    echo "  ⚡ Simple   (fast, default)"
    echo "  🎯 Ensemble (accurate)"
else
    echo -e "${BLUE}📦 Mode Available:${NC}"
    echo "  ⚡ Simple only"
fi
echo ""
echo -e "${BLUE}🌐 API Endpoints:${NC}"
echo "  POST /classify?mode=simple|ensemble"
echo "  POST /classify_detailed?mode=simple|ensemble"
echo "  GET  /health"
echo "  GET  /stats"
echo "  GET  /docs (Swagger UI)"
echo ""
echo -e "${GREEN}📡 Server: http://localhost:8000${NC}"
echo -e "${GREEN}📄 Docs:   http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python main_production.py
