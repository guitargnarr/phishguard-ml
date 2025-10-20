# Security Phishing Detector - Setup Documentation

**Last Updated**: 2025-10-20

---

## Python Environment Issues Discovered

### The Problem

This project was developed and the ML models were trained using **Python 3.9.6** (`/usr/bin/python3`). The system also has a **broken Python 3.13 installation** that silently fails on all commands.

**Issue**: Running `python3` on this system defaults to the broken Python 3.13, causing the project to fail.

**Solution**: Use the virtual environment (venv) which explicitly uses Python 3.9.6.

---

## System Python Versions

```bash
# Working Python (DO USE)
/usr/bin/python3 --version
# Output: Python 3.9.6

# Broken Python (DO NOT USE)
python3 --version
# May show Python 3.13 but commands fail silently
```

---

## Quick Start

### 1. Clone and Setup (First Time)

```bash
cd ~/Projects/Security-Tools/security-phishing-detector

# Virtual environment is already created and configured
# Dependencies are already installed

# Just run the server
./run.sh
```

### 2. If You Need to Reinstall

```bash
# Remove old venv
rm -rf venv

# Create new venv with Python 3.9.6
/usr/bin/python3 -m venv venv

# Install dependencies
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

### 3. Run the Server

```bash
./run.sh
```

Server will start on http://localhost:8000

---

## Critical Dependencies

### scikit-learn Version Match

**CRITICAL**: The ML models (`phishing_clf.pkl`, `tfidf_vec.pkl`) were trained with **scikit-learn 1.6.1**.

**You MUST use scikit-learn 1.6.1** to load these models, otherwise you'll get version mismatch warnings or errors.

The `requirements.txt` pins this version:
```
scikit-learn==1.6.1  # CRITICAL: Models trained with this version
```

### Other Pinned Versions

```
fastapi==0.118.0
uvicorn==0.37.0
pydantic==2.11.1
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<3.0.0
xgboost==2.1.4
lightgbm==4.6.0
```

---

## Troubleshooting

### Issue: "Address already in use"

**Symptom**: Server fails to start, port 8000 in use

**Solution**:
```bash
# Kill existing process
lsof -ti:8000 | xargs kill

# Or use different port (edit main.py line ~237):
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Issue: "Models not found"

**Symptom**: Server fails to start, can't find `.pkl` files

**Solution**:
```bash
# Check models exist
ls -lh models/
# Should show: phishing_clf.pkl (5.9K), tfidf_vec.pkl (27K)

# If missing, train models
./venv/bin/python train_model.py
```

### Issue: scikit-learn version mismatch

**Symptom**: Warning about sklearn version mismatch

**Solution**:
```bash
# Check installed version
./venv/bin/python -c "import sklearn; print(sklearn.__version__)"
# Should output: 1.6.1

# If wrong version, reinstall
./venv/bin/pip install scikit-learn==1.6.1
```

### Issue: venv not activating

**Symptom**: `run.sh` fails to activate venv

**Solution**:
```bash
# Recreate venv
rm -rf venv
/usr/bin/python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

---

## Project Portability

### Moving to Another Machine

To move this project to another machine:

1. **Copy project folder**:
   ```bash
   cp -r security-phishing-detector /path/to/destination
   ```

2. **On new machine** (first time only):
   ```bash
   cd security-phishing-detector

   # Remove old venv (it has hardcoded paths)
   rm -rf venv

   # Create new venv (use Python 3.9.6 or compatible)
   python3.9 -m venv venv

   # Install dependencies
   ./venv/bin/pip install --upgrade pip
   ./venv/bin/pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   ./run.sh
   ```

---

## Development Workflow

### Making Changes

```bash
# Activate venv
source venv/bin/activate

# Make your changes to Python files

# Test changes
python main.py

# Or run automated tests
pytest

# Deactivate when done
deactivate
```

### Adding Dependencies

```bash
# Activate venv
source venv/bin/activate

# Install new package
pip install new-package==X.Y.Z

# Update requirements.txt
pip freeze | grep new-package >> requirements.txt

# Deactivate
deactivate
```

### Retraining Models

If you need to retrain with updated data:

```bash
# Activate venv
source venv/bin/activate

# Train new models
python train_model.py

# Models will be saved to models/
# New models will be compatible with current scikit-learn version

# Deactivate
deactivate
```

---

## Version Control

### Files to Commit

✅ **DO commit**:
- `main.py`, `train_model.py`, etc. (all `.py` files)
- `requirements.txt`
- `README.md`, `SETUP.md`
- `run.sh`
- `models/` directory (with trained models)

❌ **DO NOT commit**:
- `venv/` directory (add to .gitignore)
- `__pycache__/` directories
- `*.pyc` files
- `logs/` directory (runtime logs)
- `.env` files (if using environment variables)

### Sample .gitignore

```
# Virtual environment
venv/
env/
.venv/

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Runtime logs
logs/
*.log

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

## Python Version Compatibility

### Recommended: Python 3.9.6

This project is **tested and working** with Python 3.9.6.

### Other Python Versions

**Python 3.8**: May work, not tested
**Python 3.9.x**: Should work (tested with 3.9.6)
**Python 3.10+**: Some dependencies may not be compatible

**AVOID**: Python 3.13 (many ML packages not yet compatible)

---

## Performance Notes

### Startup Time

- **Cold start**: 2-3 seconds (loading models)
- **Warm start**: 2-3 seconds (models cached in memory)

### Memory Usage

- **Base**: ~150 MB
- **With loaded models**: ~200 MB
- **Peak (processing)**: ~250 MB

### Response Time

- **Health check**: <10ms
- **Single email classification**: 20-50ms
- **Batch processing**: 50-200ms (depends on batch size)

---

## Security Notes

### API Security (Production)

Before deploying to production, add:

1. **API Key Authentication**
2. **Rate Limiting**
3. **HTTPS/TLS**
4. **CORS Configuration**
5. **Input Validation**

See FastAPI security docs: https://fastapi.tiangolo.com/tutorial/security/

### Model Security

- Models are stored as `.pkl` files (pickle format)
- **WARNING**: Never load `.pkl` files from untrusted sources (arbitrary code execution risk)
- Only use models you trained yourself or from trusted sources

---

## Support

**Issues**: Check `TROUBLESHOOTING.md`
**Questions**: Review `README.md` and this `SETUP.md`
**Updates**: Check git commit history for recent changes

---

*This documentation reflects the actual working configuration as of 2025-10-20*
