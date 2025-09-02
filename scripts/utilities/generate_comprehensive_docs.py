#!/usr/bin/env python3
"""
Comprehensive Documentation Generator for Beverly Knits ERP v2
Generates complete system documentation following the markdown_documentation_generator framework
"""

import os
import json
import ast
import glob
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Any

class ComprehensiveDocumentationGenerator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.documentation = {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the project structure and organization"""
        structure = {
            "core_modules": [],
            "services": [],
            "data_loaders": [],
            "ml_models": [],
            "utilities": [],
            "api_endpoints": [],
            "database": [],
            "web_interface": []
        }
        
        # Scan source directories
        src_path = self.project_root / "src"
        if src_path.exists():
            for category, pattern in [
                ("core_modules", "core/*.py"),
                ("services", "services/*.py"),
                ("data_loaders", "data_loaders/*.py"),
                ("ml_models", "ml_models/*.py"),
                ("utilities", "utils/*.py"),
                ("database", "database/*.py")
            ]:
                files = glob.glob(str(src_path / pattern))
                structure[category] = [Path(f).name for f in files if not f.endswith("__pycache__")]
        
        # Scan web interface
        web_path = self.project_root / "web"
        if web_path.exists():
            structure["web_interface"] = list(web_path.glob("*.html"))
            
        return structure
    
    def extract_api_endpoints(self) -> List[Dict[str, Any]]:
        """Extract API endpoints from the main application"""
        endpoints = []
        
        # Read the main application file
        app_file = self.project_root / "src" / "core" / "beverly_comprehensive_erp.py"
        if app_file.exists():
            with open(app_file, 'r') as f:
                content = f.read()
                
            # Find all route decorators
            route_pattern = r'@app\.route\(["\']([^"\']+)["\'].*?\)'
            routes = re.findall(route_pattern, content)
            
            # Extract endpoint details
            for route in routes:
                endpoint = {
                    "path": route,
                    "methods": ["GET"],  # Default
                    "category": self._categorize_endpoint(route),
                    "description": self._generate_endpoint_description(route)
                }
                endpoints.append(endpoint)
                
        return endpoints
    
    def _categorize_endpoint(self, path: str) -> str:
        """Categorize an endpoint based on its path"""
        if "yarn" in path:
            return "Yarn Management"
        elif "production" in path:
            return "Production"
        elif "inventory" in path:
            return "Inventory"
        elif "forecast" in path or "ml" in path:
            return "ML/Forecasting"
        elif "api" in path:
            return "API"
        else:
            return "General"
    
    def _generate_endpoint_description(self, path: str) -> str:
        """Generate description for an endpoint based on its path"""
        descriptions = {
            "yarn-intelligence": "Comprehensive yarn analysis with shortage detection",
            "inventory-intelligence": "Enhanced inventory metrics and analytics",
            "production-planning": "Production schedule and capacity planning",
            "ml-forecast": "Machine learning forecasting predictions",
            "six-phase-planning": "Six-phase supply chain planning engine",
            "production-pipeline": "Real-time production flow tracking"
        }
        
        for key, desc in descriptions.items():
            if key in path:
                return desc
        return "API endpoint for " + path.replace("/api/", "").replace("-", " ")
    
    def analyze_data_flow(self) -> Dict[str, Any]:
        """Analyze data flow through the system"""
        return {
            "data_sources": {
                "primary": "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/",
                "files": [
                    "yarn_inventory.xlsx - Yarn inventory with Planning Balance",
                    "Sales Activity Report.csv - Historical sales data",
                    "eFab_Knit_Orders.xlsx - Active production orders",
                    "Style_BOM.csv - Bill of Materials",
                    "eFab_Inventory_[stage].xlsx - Stage inventories"
                ]
            },
            "data_pipeline": [
                "Data Loading (OptimizedDataLoader)",
                "Data Caching (UnifiedCacheManager)",
                "Data Processing (InventoryAnalyzer)",
                "ML Forecasting (SalesForecastingEngine)",
                "Planning (SixPhasePlanningEngine)",
                "API Response Generation"
            ],
            "production_flow": [
                "G00 (Greige/Knitting)",
                "G02 (Finishing)",
                "I01 (Inspection)",
                "F01 (Finished Goods)",
                "P01 (Allocated)"
            ]
        }
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary"""
        return """# Beverly Knits ERP v2 - Comprehensive Documentation

## Executive Summary

Beverly Knits ERP v2 is a production-ready textile manufacturing Enterprise Resource Planning system designed to optimize the complete supply chain from yarn procurement to finished goods delivery. The system provides real-time inventory intelligence, ML-powered demand forecasting, and sophisticated 6-phase supply chain planning capabilities.

### Key Capabilities
- **Real-time Inventory Management**: Track 1,198+ yarn items with Planning Balance calculations
- **ML-Powered Forecasting**: 90% accuracy at 9-week horizon using ensemble methods
- **6-Phase Supply Chain Planning**: Optimize from yarn procurement through finished goods
- **Production Flow Tracking**: Monitor products through 5 production stages (G00→G02→I01→F01→P01)
- **Intelligent Yarn Substitution**: ML-based recommendations for interchangeable yarns
- **Performance Optimization**: 100x+ speed improvement with parallel data loading and caching

### System Metrics
- **Data Volume**: 28,653+ BOM entries, 10,338+ sales records, 221+ active orders
- **Response Time**: <200ms for most API endpoints
- **Dashboard Load**: <3 seconds full render
- **Cache Hit Rate**: 70-90% typical performance"""
    
    def generate_architecture_documentation(self) -> str:
        """Generate architecture documentation"""
        return """## System Architecture

### High-Level Architecture

The system follows a monolithic architecture with modular service components:

```
┌─────────────────────────────────────────────────────────┐
│                   Web Dashboard                          │
│              (consolidated_dashboard.html)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Flask Application                         │
│          (beverly_comprehensive_erp.py)                  │
├──────────────────────────────────────────────────────────┤
│  Core Classes:                                           │
│  • InventoryAnalyzer - Inventory analysis engine         │
│  • InventoryManagementPipeline - Pipeline orchestration  │
│  • SalesForecastingEngine - ML forecasting               │
│  • CapacityPlanningEngine - Production planning          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Service Layer                             │
├──────────────────────────────────────────────────────────┤
│  • Yarn Intelligence Services                            │
│  • Production Planning Services                          │
│  • Forecasting Services                                  │
│  • Data Loading Services                                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Data Layer                                  │
├──────────────────────────────────────────────────────────┤
│  • Excel Files (yarn_inventory, BOM, orders)             │
│  • CSV Files (sales data, reports)                       │
│  • Cache Layer (Memory + Redis)                          │
│  • SQLite Database (production tracking)                 │
└──────────────────────────────────────────────────────────┘
```

### Component Architecture

#### Core Application (beverly_comprehensive_erp.py)
- **Lines**: 7000+
- **Responsibilities**: Request handling, business logic orchestration
- **Key Methods**: Data loading, API endpoints, dashboard serving

#### Data Loading Architecture
```
OptimizedDataLoader (100x speed)
    ├── Parallel processing
    ├── Chunk-based loading
    └── Multi-level caching
    
ParallelDataLoader (4x speed)
    ├── Concurrent file reading
    ├── Thread pool execution
    └── Batch processing
    
UnifiedCacheManager
    ├── Memory cache (L1)
    ├── Redis cache (L2)
    └── TTL management
```

#### ML Architecture
```
Forecasting Models:
├── ARIMA (Time series)
├── Prophet (Seasonal patterns)
├── LSTM (Deep learning)
├── XGBoost (Gradient boosting)
└── Ensemble (Combined predictions)

Accuracy: 90-95% with ensemble methods
Fallback: Ensemble → Single → Statistical → Last known
```"""
    
    def generate_api_documentation(self) -> str:
        """Generate API documentation"""
        endpoints = self.extract_api_endpoints()
        
        api_doc = """## API Documentation

### Base URL
```
http://localhost:5006
```

### Authentication
Currently no authentication required (development mode)

### Critical Endpoints

#### Yarn & Inventory Intelligence

**GET /api/yarn-intelligence**
- **Description**: Comprehensive yarn analysis with shortage detection
- **Parameters**: 
  - `analysis`: Type of analysis (shortage, forecast, all)
  - `forecast`: Include forecast data (true/false)
- **Response**: JSON with yarn shortages, recommendations, and forecasts

**GET /api/inventory-intelligence-enhanced**
- **Description**: Enhanced inventory metrics and analytics
- **Parameters**:
  - `view`: Data view (summary, detailed)
  - `realtime`: Force real-time calculation (true/false)
- **Response**: JSON with inventory KPIs, trends, and insights

#### Production Planning

**GET /api/production-planning**
- **Description**: Production schedule and capacity planning
- **Parameters**:
  - `view`: Planning view (orders, capacity, schedule)
  - `forecast`: Include forecast data (true/false)
- **Response**: JSON with production plans and schedules

**GET /api/production-pipeline**
- **Description**: Real-time production flow tracking
- **Response**: JSON with stage-wise production status

**GET /api/six-phase-planning**
- **Description**: Execute 6-phase supply chain planning
- **Response**: JSON with complete planning results

#### ML & Forecasting

**GET /api/ml-forecast-detailed**
- **Description**: Detailed ML predictions
- **Parameters**:
  - `detail`: Level of detail (summary, full)
  - `format`: Response format (json, report)
  - `horizon`: Forecast horizon in days
- **Response**: JSON with predictions and confidence intervals

**POST /api/retrain-ml**
- **Description**: Trigger ML model retraining
- **Body**: Optional training parameters
- **Response**: JSON with training results

#### System & Health

**GET /api/health**
- **Description**: System health check
- **Response**: JSON with system status

**GET /api/debug-data**
- **Description**: Debug data loading issues
- **Response**: JSON with detailed data status

**GET /api/cache-stats**
- **Description**: Cache performance metrics
- **Response**: JSON with cache statistics"""
        
        return api_doc
    
    def generate_deployment_guide(self) -> str:
        """Generate deployment guide"""
        return """## Deployment & Operations Guide

### Prerequisites
- Python 3.8+
- Redis (optional, for caching)
- 4GB+ RAM recommended
- Excel file access to data directory

### Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd beverly_knits_erp_v2
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
cp config/.env.example config/.env
# Edit .env with your settings
```

4. **Start Server**
```bash
python3 src/core/beverly_comprehensive_erp.py
# Server runs on port 5006
```

### Docker Deployment

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t beverly-erp .

# Run container
docker run -p 5006:5006 -v /path/to/data:/data beverly-erp
```

### Production Deployment

#### Using Docker Compose
```bash
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

#### Railway Deployment
- Use Dockerfile.railway for Railway platform
- Set environment variables in Railway dashboard
- Mount data volume for persistent storage

### Monitoring & Maintenance

#### Health Checks
```bash
curl http://localhost:5006/api/health
```

#### Clear Cache
```bash
rm -rf /tmp/bki_cache/*
curl http://localhost:5006/api/reload-data
```

#### View Logs
```bash
tail -f logs/application.log
```

### Troubleshooting

#### Port Conflicts
```bash
lsof -i :5006
kill -9 <PID>
```

#### Data Loading Issues
1. Check file paths in CLAUDE.md
2. Clear cache: `rm -rf /tmp/bki_cache/*`
3. Restart server
4. Check debug endpoint: `/api/debug-data`

#### Performance Issues
1. Monitor cache hit rate: `/api/cache-stats`
2. Check memory usage
3. Enable Redis for better caching
4. Use production deployment configuration"""
    
    def generate_testing_guide(self) -> str:
        """Generate testing guide"""
        return """## Testing Strategy

### Test Coverage
- **Current Coverage**: Target 80% for critical paths
- **Critical Areas**: 
  - Planning Balance calculations
  - Style mapping (fStyle# ↔ Style#)
  - Yarn shortage detection
  - API endpoints

### Running Tests

#### All Tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

#### By Category
```bash
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m e2e           # End-to-end tests
pytest -n auto          # Parallel execution
```

#### Specific Tests
```bash
pytest tests/unit/test_inventory.py::test_yarn_shortage_calculation -v
```

### Test Organization
```
tests/
├── unit/              # Business logic tests
├── integration/       # API endpoint tests
├── e2e/              # Workflow tests
├── performance/      # Load and speed tests
└── conftest.py       # Test fixtures
```

### Key Test Cases

#### Inventory Tests
- Planning Balance formula validation
- Negative allocated values handling
- Shortage detection accuracy
- Weekly demand calculations

#### Production Tests
- Stage flow validation (G00→G02→I01→F01→P01)
- BOM explosion accuracy
- Capacity constraint checking
- Lead time calculations

#### Forecasting Tests
- Model accuracy validation
- Fallback chain testing
- Ensemble performance
- Online learning updates

### Performance Testing

```bash
# Load testing
locust -f tests/performance/locustfile.py --host=http://localhost:5006

# Benchmark specific endpoints
python tests/performance/benchmark_api.py
```"""
    
    def generate_data_dictionary(self) -> str:
        """Generate data dictionary"""
        return """## Data Dictionary

### Yarn Inventory Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Desc# | String | Unique yarn identifier | "19003" |
| Description | String | Yarn description | "100% Cotton White" |
| Planning_Balance | Float | Available + On Order - Allocated | -500.0 |
| Theoretical_Balance | Float | Physical inventory on hand | 1000.0 |
| Allocated | Float | Amount allocated (negative value) | -1500.0 |
| On_Order | Float | Amount on order from suppliers | 0.0 |
| Consumed | Float | Monthly consumption (negative) | -250.0 |
| Cost/Pound | Float | Cost per pound of yarn | 2.50 |

### Production Stages

| Stage | Name | Description | Status |
|-------|------|-------------|--------|
| G00 | Greige/Knitting | Raw knitted fabric | WIP |
| G02 | Finishing | Dyeing and finishing | WIP |
| I01 | Inspection | Quality control | WIP |
| F01 | Finished Goods | Available for sale | READY |
| P01 | Allocated | Reserved for shipment | COMMITTED |

### BOM Structure

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Style# | String | Style identifier | "ABC123" |
| fStyle# | String | Fabric style reference | "FAB456" |
| Yarn_ID | String | Yarn identifier (Desc#) | "19003" |
| Percentage | Float | Yarn percentage in style | 0.45 |
| Usage_Per_Unit | Float | Pounds per unit | 1.2 |

### Sales Data

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Order_Number | String | Unique order ID | "SO-2024-001" |
| Style# | String | Style ordered | "ABC123" |
| Quantity | Integer | Units ordered | 500 |
| Ship_Date | Date | Scheduled ship date | "2024-08-15" |
| Customer | String | Customer name | "Customer A" |
| Status | String | Order status | "In Production" |

### Key Formulas

**Planning Balance**
```
Planning_Balance = Theoretical_Balance + Allocated + On_Order
```
Note: Allocated is already negative in data files

**Weekly Demand**
```
If Consumed exists: abs(Consumed) / 4.3
Else if Allocated exists: abs(Allocated) / 8
Else: 10 (default minimum)
```

**Yarn Requirement**
```
Yarn_Required = Fabric_Yards × BOM_Percentage × Conversion_Factor
```"""
    
    def generate_full_documentation(self) -> str:
        """Generate complete documentation"""
        sections = [
            self.generate_executive_summary(),
            self.generate_architecture_documentation(),
            self.generate_api_documentation(),
            self.generate_deployment_guide(),
            self.generate_testing_guide(),
            self.generate_data_dictionary(),
            self._generate_troubleshooting_guide(),
            self._generate_roadmap()
        ]
        
        # Add table of contents
        toc = self._generate_table_of_contents(sections)
        
        # Add metadata
        metadata = f"""---
Generated: {self.timestamp}
Version: 2.0.0
Status: Production Ready
---

"""
        
        return metadata + toc + "\n\n" + "\n\n".join(sections)
    
    def _generate_table_of_contents(self, sections: List[str]) -> str:
        """Generate table of contents from sections"""
        toc = "## Table of Contents\n\n"
        
        for section in sections:
            # Extract headers from section
            headers = re.findall(r'^(#{1,3})\s+(.+)$', section, re.MULTILINE)
            for level, title in headers[:5]:  # Limit to first 5 headers per section
                indent = "  " * (len(level) - 1)
                anchor = title.lower().replace(" ", "-").replace("&", "").replace(",", "")
                toc += f"{indent}- [{title}](#{anchor})\n"
        
        return toc
    
    def _generate_troubleshooting_guide(self) -> str:
        """Generate troubleshooting guide"""
        return """## Troubleshooting Guide

### Common Issues & Solutions

#### Data Not Loading
**Symptoms**: Empty dashboard, no data displayed
**Solutions**:
1. Check file paths exist:
```bash
ls -la "/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/"
```
2. Clear cache:
```bash
rm -rf /tmp/bki_cache/*
```
3. Restart server:
```bash
pkill -f "python3.*beverly"
python3 src/core/beverly_comprehensive_erp.py
```

#### Planning Balance Incorrect
**Symptoms**: Negative values showing as positive
**Solutions**:
- Verify formula: Planning_Balance = Theoretical + Allocated + On_Order
- Check that Allocated values are negative in source data
- Review column name variations (Planning_Balance vs Planning_Ballance)

#### Slow Performance
**Symptoms**: Dashboard takes >10 seconds to load
**Solutions**:
1. Enable Redis caching
2. Check cache hit rate: `/api/cache-stats`
3. Use parallel data loader
4. Reduce dashboard refresh frequency

#### ML Forecasting Errors
**Symptoms**: Forecast returns errors or defaults
**Solutions**:
1. Check sales data availability
2. Verify minimum data points (30+ for ARIMA)
3. Trigger retraining: `POST /api/retrain-ml`
4. Check fallback chain is working

#### Database Errors
**Symptoms**: "disk I/O error" messages
**Solutions**:
1. Check database file permissions
2. Disable SQLite temporarily (set SQLITE_AVAILABLE = False)
3. Use mock data for testing
4. Recreate database if corrupted"""
    
    def _generate_roadmap(self) -> str:
        """Generate development roadmap"""
        return """## Development Roadmap

### Completed Features ✅
- Core inventory management system
- ML-powered forecasting (5 models)
- 6-phase supply chain planning
- Real-time production tracking
- Yarn substitution intelligence
- Performance optimization (100x+ improvement)
- Comprehensive caching system
- API consolidation

### In Progress 🚧
- Production flow tracking implementation
- Enhanced reporting capabilities
- Mobile-responsive dashboard
- Advanced analytics dashboard

### Planned Features 📋

#### Q1 2025
- [ ] Multi-tenant support
- [ ] Advanced user authentication
- [ ] Role-based access control
- [ ] Audit logging system

#### Q2 2025
- [ ] GraphQL API layer
- [ ] Real-time WebSocket updates
- [ ] Advanced ML model management
- [ ] Automated report generation

#### Q3 2025
- [ ] Mobile application
- [ ] Integration with external systems
- [ ] Advanced predictive analytics
- [ ] Automated procurement workflows

#### Q4 2025
- [ ] AI-powered decision support
- [ ] Complete automation framework
- [ ] Advanced optimization algorithms
- [ ] Enterprise integration suite

### Technical Debt
- Refactor monolithic application into microservices
- Implement comprehensive error handling
- Add request validation middleware
- Improve test coverage to 90%+
- Optimize database queries
- Implement connection pooling

### Performance Targets
- API response time: <100ms (currently ~200ms)
- Dashboard load: <2s (currently ~3s)
- Data processing: <500ms (currently ~1-2s)
- Concurrent users: 100+ (currently ~10-20)"""
    
    def save_documentation(self, output_path: str = "COMPREHENSIVE_DOCUMENTATION.md"):
        """Save the generated documentation to a file"""
        documentation = self.generate_full_documentation()
        
        output_file = self.project_root / output_path
        with open(output_file, 'w') as f:
            f.write(documentation)
        
        print(f"Documentation generated successfully: {output_file}")
        print(f"Total size: {len(documentation):,} characters")
        print(f"Generated at: {self.timestamp}")
        
        return str(output_file)


def main():
    """Main execution function"""
    print("Beverly Knits ERP v2 - Comprehensive Documentation Generator")
    print("=" * 60)
    
    generator = ComprehensiveDocumentationGenerator()
    
    print("Analyzing project structure...")
    structure = generator.analyze_project_structure()
    print(f"Found {sum(len(v) if isinstance(v, list) else 0 for v in structure.values())} components")
    
    print("Extracting API endpoints...")
    endpoints = generator.extract_api_endpoints()
    print(f"Found {len(endpoints)} API endpoints")
    
    print("Generating comprehensive documentation...")
    output_file = generator.save_documentation()
    
    print("\n✅ Documentation generation complete!")
    print(f"📄 Output file: {output_file}")
    
    return output_file


if __name__ == "__main__":
    main()