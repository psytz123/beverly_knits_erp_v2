# Beverly Knits ERP System

Comprehensive textile manufacturing ERP with real-time inventory intelligence, ML forecasting, and multi-agent collaboration.

## Features

- **Real-time Inventory Intelligence**: Track 1,200+ yarn types with automated shortage detection
- **6-Phase Supply Chain Planning**: Optimize production from raw materials to finished goods
- **ML-Powered Forecasting**: 90-95% accuracy using ensemble methods (ARIMA, Prophet, XGBoost)
- **Multi-Agent Collaboration**: AI-assisted development and automation via MCP
- **Production Pipeline Tracking**: Monitor goods through 4 stages (G00→G02→I01→F01)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboards)
- Redis (optional, for caching)
- PostgreSQL (optional, defaults to SQLite)

### Installation

```bash
# Clone and enter directory
cd BKI_comp

# Install Python dependencies
pip install -r requirements.txt

# Or using pyproject.toml
pip install -e .

# Install dashboard dependencies (optional)
cd agent_mcp/dashboard
npm install
cd ../..
```

### Running the Server

```bash
# Start main ERP server (runs on port 5005)

# Access dashboard
# Open browser to: http://localhost:5005/consolidated
```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t beverly-knits-erp .
docker run -p 5005:5005 -v ./ERP\ Data:/app/ERP\ Data beverly-knits-erp

# Or use docker-compose
docker-compose up -d
```

## Project Structure

```
BKI_comp/
├── beverly_comprehensive_erp.py   # Main Flask application (7000+ lines)
├── optimized_data_loader.py       # High-performance data loading
├── six_phase_planning_engine.py   # Supply chain planning
├── agent_mcp/                     # Multi-agent collaboration system
│   ├── agents/                    # Specialized AI agents
│   ├── core/                      # MCP orchestrator
│   └── dashboard/                 # Next.js monitoring UI
├── bkai/                          # Domain models and services
├── bki_erp/                       # Additional ERP modules
├── ERP Data/                      # Production data files
│   └── 5/                         # Active data directory
├── tests/                         # Test suite
└── consolidated_dashboard.html    # Primary user interface
```

## Key API Endpoints

### Inventory & Planning

- `GET /api/yarn-intelligence` - Comprehensive yarn analysis
- `GET /api/six-phase-planning` - Execute planning engine
- `GET /api/production-pipeline` - Real-time production flow

### ML & Forecasting

- `GET /api/ml-forecast-report` - ML forecast summary
- `POST /api/retrain-ml` - Trigger model retraining

### System Management

- `GET /api/health` - Health check
- `GET /api/reload-data` - Force cache refresh
- `GET /api/debug-data` - Debug data loading

## Data Configuration

The system expects data files in `/ERP Data/5/`:

- `yarn_inventory (4).xlsx` - Master yarn inventory
- `Style_BOM.csv` - Bill of Materials
- `Sales Activity Report.csv` - Sales transactions
- `eFab_Knit_Orders_*.xlsx` - Production orders
- `eFab_Inventory_*.xlsx` - Stage inventories

## Performance Metrics

- Data Load: 2.31s for 52,266 records
- API Response: <200ms average
- Planning Engine: <2 minutes for 1000+ materials
- Supports 50+ concurrent users

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=beverly_comprehensive_erp

# Run specific test categories
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m e2e           # End-to-end tests
```

## Deployment Options

### Railway.app

```bash
# Deploy to Railway
railway login
railway link
railway up
```

### Heroku

```bash
# Deploy to Heroku
heroku create beverly-knits-erp
git push heroku main
```

### AWS/GCP/Azure

See deployment guides in `/docs/deployment/`

## Environment Variables

```bash
# Required
DATA_PATH=/path/to/ERP Data/5
PORT=5005

# Optional
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/dbname
ML_CONFIDENCE_THRESHOLD=0.85
ENABLE_UNIFIED_MCP=True
```

## Support

For issues or questions, contact the Beverly Knits IT team.

## License

Proprietary - Beverly Knits International
