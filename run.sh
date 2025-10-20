#!/bin/bash
# Security Phishing Detector - Run Script
# Activates venv and starts the server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Security Phishing Detector - Starting..."
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo ""
    echo "Please run setup first:"
    echo "  /usr/bin/python3 -m venv venv"
    echo "  ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Verify scikit-learn version (critical for model compatibility)
SKLEARN_VERSION=$(python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "NOT FOUND")
if [ "$SKLEARN_VERSION" != "1.6.1" ]; then
    echo -e "${YELLOW}⚠️  Warning: scikit-learn version is $SKLEARN_VERSION (expected 1.6.1)${NC}"
    echo "   Models may not load correctly"
    echo ""
fi

# Check if models exist
if [ ! -f "models/phishing_clf.pkl" ] || [ ! -f "models/tfidf_vec.pkl" ]; then
    echo -e "${RED}❌ ML models not found!${NC}"
    echo ""
    echo "Please train models first:"
    echo "  python train_model.py"
    exit 1
fi

echo -e "${GREEN}✓${NC} ML models found"
echo ""

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
    echo ""
    echo "Kill existing process:"
    echo "  lsof -ti:8000 | xargs kill"
    echo ""
    echo "Or edit main.py to use a different port"
    exit 1
fi

echo -e "${GREEN}✓${NC} Port 8000 is available"
echo ""

# Start the server
echo "=========================================="
echo "Starting server on http://localhost:8000"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop"
echo ""

python main.py
