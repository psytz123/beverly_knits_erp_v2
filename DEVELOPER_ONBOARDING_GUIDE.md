# Beverly Knits ERP - Developer Onboarding Guide

## Welcome to Beverly Knits ERP Development! 👋

This guide will help you get up and running with the Beverly Knits ERP system quickly and efficiently.

## Table of Contents
1. [System Overview](#system-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Common Tasks](#common-tasks)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Resources](#resources)

---

## System Overview

Beverly Knits ERP is a comprehensive Manufacturing ERP system designed for textile and apparel manufacturing. It features:
- **Supply Chain Management**: Multi-level BOM, inventory management, procurement
- **Production Planning**: Machine scheduling, capacity planning, order management
- **ML Forecasting**: Demand prediction, inventory optimization, sales forecasting
- **Real-time Monitoring**: Live dashboards, alerts, KPI tracking
- **External Integration**: eFab API sync, supplier connections

---

## Development Environment Setup

### Prerequisites

1. **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
2. **Git** - [Download Git](https://git-scm.com/downloads)
3. **PostgreSQL** (optional, for production-like environment)
4. **Redis** (for caching and Celery)
5. **Docker** (optional, for containerized development)

### Initial Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd beverly_knits_erp_v2
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Key variables to configure:
# - DATABASE_URL (PostgreSQL connection string)
# - REDIS_URL (Redis connection)
# - SECRET_KEY (Flask secret key)
# - EFAB_API_KEY (External API key)
```

#### 5. Database Setup

##### Option A: SQLite (Development)
```bash
# SQLite is auto-created on first run
# No additional setup needed
```

##### Option B: PostgreSQL (Recommended)
```bash
# Run setup script
./setup_postgres.sh

# Or manually:
psql -U postgres
CREATE DATABASE beverly_knits;
\q

# Run migrations
python src/database/setup.py
```

#### 6. Start Redis (for caching)
```bash
# Docker
docker run -d -p 6379:6379 redis

# Or install locally
# Ubuntu/Debian: sudo apt-get install redis-server
# Mac: brew install redis
# Windows: Use WSL or Docker
```

---

## Project Structure

```
beverly_knits_erp_v2/
├── src/                        # Source code
│   ├── ai_agents/             # AI agent system
│   │   ├── core/              # Agent orchestration
│   │   └── industry/          # Domain-specific agents
│   ├── api/                   # API endpoints
│   │   ├── blueprints/        # Flask blueprints
│   │   └── handlers/          # Request handlers
│   ├── core/                  # Core application
│   │   └── beverly_comprehensive_erp.py
│   ├── database/              # Database layer
│   │   ├── models.py          # SQLAlchemy models
│   │   └── connection_pool.py # DB connections
│   ├── forecasting/           # ML forecasting
│   │   ├── enhanced_forecasting_engine.py
│   │   └── forecast_auto_retrain.py
│   ├── ml_models/             # Machine learning
│   ├── optimization/          # Optimization algorithms
│   ├── production/            # Production management
│   ├── services/              # Business logic
│   └── utils/                 # Utilities
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── scripts/                   # Utility scripts
├── deployment/                # Deployment configs
├── docs/                      # Documentation
└── web/                       # Frontend assets
```

---

## Getting Started

### Running the Application

#### Development Mode
```bash
# Start the Flask development server
python start_erp.py

# Application runs on http://localhost:5006
```

#### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:5006
```

#### Using Scripts
```bash
# Windows
START_CLEAN.bat           # Clean start
RESTART_SERVERS.bat       # Restart all services

# Linux/Mac
./start_erp.sh           # Start ERP server
./start_dashboard.sh     # Start dashboard
```

### Accessing the Application

1. **Main Dashboard**: http://localhost:5006
2. **Consolidated View**: http://localhost:5006/consolidated
3. **Machine Schedule**: http://localhost:5006/machine-schedule
4. **AI Factory Floor**: http://localhost:5006/ai-factory-floor

### API Testing
```bash
# Use the test HTML files
open tests/test_api.html
open tests/test_dashboard.html

# Or use curl
curl http://localhost:5006/api/comprehensive-kpis
```

---

## Development Workflow

### 1. Feature Development

#### Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

#### Code Organization
- **API Endpoints**: Add to `src/api/` or appropriate blueprint
- **Business Logic**: Add to `src/services/`
- **Database Models**: Update `src/database/models.py`
- **ML Models**: Add to `src/ml_models/` or `src/forecasting/`

#### Example: Adding a New API Endpoint
```python
# src/core/beverly_comprehensive_erp.py

@app.route("/api/your-endpoint", methods=["GET", "POST"])
def your_endpoint():
    """
    Your endpoint description.
    """
    try:
        # Your logic here
        data = process_request()
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        logger.error(f"Error in your_endpoint: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
```

### 2. Database Changes

#### Adding a New Model
```python
# src/database/models.py

class YourModel(Base):
    __tablename__ = 'your_table'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Add relationships
    parent_id = Column(Integer, ForeignKey('parent_table.id'))
    parent = relationship("ParentModel", back_populates="children")
```

#### Creating Migrations
```bash
# Generate migration
alembic revision -m "Add your_table"

# Apply migration
alembic upgrade head
```

### 3. Testing Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=src --cov-report=html
```

---

## Testing

### Test Structure
```
tests/
├── unit/              # Isolated component tests
├── integration/       # Component interaction tests
├── e2e/              # Full workflow tests
└── conftest.py       # Shared fixtures
```

### Writing Tests

#### Unit Test Example
```python
# tests/unit/test_your_feature.py
import pytest
from src.services.your_service import YourService

def test_your_function():
    """Test your function behavior."""
    service = YourService()
    result = service.process_data({"input": "test"})
    assert result["status"] == "success"
```

#### Integration Test Example
```python
# tests/integration/test_api.py
def test_api_endpoint(client):
    """Test API endpoint integration."""
    response = client.get("/api/your-endpoint")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
```

### Running Tests
```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/

# With verbose output
pytest -v

# With coverage
pytest --cov=src
```

---

## Common Tasks

### 1. Data Import
```python
# Import Excel data
python scripts/import_data.py --file data.xlsx --type yarn

# Sync with eFab API
python src/database/efab_api_sync.py
```

### 2. Cache Management
```python
# Clear cache
curl -X POST http://localhost:5006/api/cache-clear

# View cache stats
curl http://localhost:5006/api/cache-stats
```

### 3. Running ML Training
```python
# Retrain forecasting models
python src/forecasting/forecast_auto_retrain.py

# Validate model performance
python src/forecasting/forecast_validation_backtesting.py
```

### 4. Database Operations
```bash
# Backup database
pg_dump beverly_knits > backup.sql

# Restore database
psql beverly_knits < backup.sql

# Query database
python
>>> from src.database.models import CFVersion
>>> from src.database.connection_pool import get_session
>>> session = get_session()
>>> styles = session.query(CFVersion).all()
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
```bash
# Error: Address already in use
# Solution: Kill the process
lsof -i :5006  # Find PID
kill -9 <PID>  # Kill process
```

#### 2. Database Connection Error
```bash
# Check PostgreSQL is running
pg_isready

# Check connection string in .env
DATABASE_URL=postgresql://user:password@localhost/beverly_knits
```

#### 3. Redis Connection Error
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG
```

#### 4. Import Errors
```bash
# Ensure you're in virtual environment
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

#### 5. ML Model Errors
```bash
# Check ML error log
tail -f ml_errors.log

# Retrain models
python src/forecasting/forecast_auto_retrain.py
```

---

## Best Practices

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings to all functions and classes
- Keep functions small and focused

### Git Workflow
```bash
# Before starting work
git pull origin main

# Commit often with clear messages
git commit -m "feat: Add yarn substitution API"
git commit -m "fix: Resolve inventory calculation bug"
git commit -m "docs: Update API documentation"

# Push and create PR
git push origin feature/your-feature
```

### API Development
- Always validate input data
- Return consistent response formats
- Include proper error messages
- Add rate limiting to public endpoints
- Document all endpoints in API_ENDPOINT_CATALOG.md

### Database Best Practices
- Always use migrations for schema changes
- Add indexes for frequently queried columns
- Use transactions for multi-table updates
- Implement soft deletes for audit trail

### Security Considerations
- Never commit secrets to git
- Use environment variables for configuration
- Validate and sanitize all user input
- Implement proper authentication and authorization
- Keep dependencies updated

---

## Resources

### Internal Documentation
- [Comprehensive Intelligence Report](COMPREHENSIVE_CODEBASE_INTELLIGENCE_REPORT.md)
- [API Endpoint Catalog](API_ENDPOINT_CATALOG.md)
- [Database Schema](DATABASE_SCHEMA_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### External Resources
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)
- [Python Best Practices](https://docs.python-guide.org/)
- [Docker Documentation](https://docs.docker.com/)

### Key Files to Review
1. `src/core/beverly_comprehensive_erp.py` - Main application
2. `src/database/models.py` - Database schema
3. `src/forecasting/enhanced_forecasting_engine.py` - ML forecasting
4. `tests/test_comprehensive_coverage.py` - Test examples

### Getting Help
- Check existing documentation
- Review test files for examples
- Look at git history for similar changes
- Ask team members for guidance

---

## Quick Reference

### Useful Commands
```bash
# Start server
python start_erp.py

# Run tests
pytest

# Check code style
flake8 src/

# Format code
black src/

# Database shell
python
>>> from src.database.connection_pool import get_session
>>> session = get_session()

# API testing
curl http://localhost:5006/api/comprehensive-kpis | jq
```

### Environment Variables
```bash
APP_PORT=5006
APP_HOST=0.0.0.0
DATABASE_URL=postgresql://user:pass@localhost/beverly_knits
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
EFAB_API_KEY=your-api-key
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug specific module
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

---

## Welcome Aboard! 🚀

You're now ready to contribute to the Beverly Knits ERP system. Remember:
- Ask questions when needed
- Write tests for your code
- Document your changes
- Follow the established patterns
- Have fun building amazing features!

---

*Last Updated: 2025-01-18*
*Version: 1.0*