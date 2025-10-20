# Contributing to PhishGuard ML

We welcome contributions! Here's how to help:

## Development Setup

1. Fork the repository
2. Create a virtual environment: `python3.9 -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install dev tools: `pip install pytest flake8 black mypy`

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality
4. Run tests: `pytest tests/`
5. Check code style: `flake8 . && black --check .`
6. Commit: `git commit -m "feat: your feature"`
7. Push and create Pull Request

## Code Style

- Follow PEP 8
- Use type hints where applicable
- Write docstrings (Google style)
- Add tests for new features
- Keep functions focused and modular

## Testing Requirements

All contributions must include tests:
- Unit tests for new functions
- Integration tests for API endpoints
- Maintain 100% test pass rate
- Add edge cases and error handling tests

## Pull Request Process

1. Update documentation (README, docstrings)
2. Ensure all tests pass (`pytest tests/`)
3. Update CHANGELOG.md with your changes
4. Request review from maintainers
5. Address feedback promptly

## Reporting Bugs

Use the GitHub issue tracker:
- Search existing issues first
- Use the bug report template
- Include reproduction steps
- Provide system information (OS, Python version)
- Include error messages and logs

## Feature Requests

We love new ideas! When suggesting features:
- Use the feature request template
- Explain the use case clearly
- Describe the expected behavior
- Consider implementation impact

## Code of Conduct

Be respectful and professional:
- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the project

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

Thank you for contributing! 🙏
