# Contributing to ML Trading System

Thank you for your interest in contributing to the ML Trading System! This project is designed to be a comprehensive machine learning trading platform with support for equities and commodities.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ML_trading.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Project Structure

```
ML_trading/
├── src/                    # Source code
│   ├── features.py         # Feature engineering (559 technical indicators)
│   ├── ensemble_models.py  # ML ensemble models
│   ├── sentiment_analysis.py # Sentiment analysis
│   ├── backtesting.py      # Backtesting framework
│   ├── kafka_integration.py # Real-time data processing
│   ├── multi_asset_processor.py # Multi-asset support
│   └── monitoring_dashboard.py # Flask dashboard
├── test_system.py          # System testing
├── requirements.txt        # Dependencies
├── Dockerfile             # Containerization
├── docker-compose.yml     # Local development
└── deploy.sh              # Deployment script
```

## Testing

Run the test suite to ensure everything works:

```bash
python test_system.py
```

## Performance Benchmarks

The system has been tested and validated with the following results:

- **AAPL**: 1.435 Sharpe ratio, 51.9% returns
- **PDBC (Commodities)**: 4.193 Sharpe ratio, 63.0% returns
- **GSG (Commodities)**: 1.788 Sharpe ratio, 31.3% returns
- **GLD (Gold)**: 0.530 Sharpe ratio, 11.0% returns

## Contributing Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Add docstrings to new functions
3. **Testing**: Add tests for new features
4. **Performance**: Ensure new features don't degrade system performance
5. **Compatibility**: Maintain compatibility with existing features

## Feature Requests

When requesting new features, please include:
- Use case description
- Expected performance impact
- Implementation approach
- Testing strategy

## Bug Reports

When reporting bugs, please include:
- Steps to reproduce
- Expected vs actual behavior
- System information
- Error messages/logs

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Run tests: `python test_system.py`
6. Commit changes: `git commit -m "Add amazing feature"`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue or contact the maintainer.
