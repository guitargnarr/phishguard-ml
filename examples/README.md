# PhishGuard ML Examples

This directory contains examples demonstrating how to use the PhishGuard ML API.

## 📁 Directory Structure

```
examples/
├── README.md              # This file
├── scripts/               # Python scripts
│   └── simple_classification.py
├── notebooks/             # Jupyter tutorials (coming soon)
└── data/                  # Sample data files
    └── sample_emails.json
```

## 🚀 Quick Start

### 1. Start the API Server

```bash
cd /path/to/security-phishing-detector
./run.sh
```

The server will start on `http://localhost:8000`

### 2. Run the Simple Classification Example

```bash
# Option 1: Direct execution
./examples/scripts/simple_classification.py

# Option 2: Via python
python examples/scripts/simple_classification.py
```

## 📝 Available Examples

### Scripts

#### `simple_classification.py`
**Purpose**: Demonstrates basic API usage
**Complexity**: Beginner
**Features**:
- Connect to the API
- Send classification requests
- Parse responses
- Display results with confidence scores

**Output Example**:
```
Test 1/5:
Email: URGENT! Your PayPal account has been suspended...
Expected: phishing
Result: phishing
Confidence: 90.45%
Match: ✅ Correct
```

### Data Files

#### `data/sample_emails.json`
Sample email dataset for testing with labeled examples (phishing and legitimate).

## 🔧 Requirements

All examples require:
- PhishGuard ML API server running (`./run.sh`)
- Python 3.9+ (for scripts)
- `requests` library: `pip install requests`

For Jupyter notebooks (when available):
- `pip install jupyter matplotlib pandas`

## 💡 Tips

1. **API Connection**: Make sure the server is running before executing examples
2. **Port Configuration**: Default is `localhost:8000`, adjust `API_URL` if different
3. **Error Handling**: Examples include error handling for common issues
4. **Extending**: Use these as templates for your own scripts

## 📚 Additional Resources

- [Main README](../README.md) - Full documentation
- [API Documentation](../README.md#-api-documentation) - Endpoint details
- [CONTRIBUTING](../CONTRIBUTING.md) - How to add new examples

## 🐛 Issues

If you encounter issues with any examples, please [open an issue](https://github.com/guitargnarr/security-phishing-detector/issues) with:
- Example name
- Error message
- Your environment (OS, Python version)
