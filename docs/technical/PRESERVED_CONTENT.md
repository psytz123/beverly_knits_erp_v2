# Preserved Technical Content from Documentation Cleanup

**Date**: September 6, 2025
**Source**: Content extracted from obsolete documentation files before cleanup

---

## Planning Balance Formula (Critical Business Logic)

### Formula Definition
```
Planning Balance = Theoretical Balance + Allocated + On Order
```

### Critical Understanding - Allocated Values
The `Allocated` field represents yarn that has been **consumed or committed** to production orders. These values are **NEGATIVE** by design:

- **Negative Allocated** = Yarn consumed/committed (reduces available balance)
- **Positive Allocated** = Rare, typically corrections or returns

### Mathematical Example
```
Given:
- Theoretical Balance = 224.10 lbs
- Allocated = -185.14 lbs (consumed)
- On Order = 0.00 lbs

Calculation:
Planning Balance = 224.10 + (-185.14) + 0.00
Planning Balance = 38.96 lbs
```

### Validation Results
- **Test Date**: 2025-09-02
- **Rows Tested**: 982
- **Accuracy**: 97.56%
- **Failures**: 24 (due to NaN values in source data)

### Common Misunderstandings
1. **"Allocated should be positive"** - INCORRECT
   - Allocated is negative when yarn is consumed
   - The formula ADDS the negative value (subtracting consumption)

2. **"Formula should subtract Allocated"** - INCORRECT
   - Formula correctly adds Allocated
   - Allocated is already negative, so adding it subtracts

---

## Data Consistency Implementation Details

### Problem Solved
Multiple inconsistent methods for calculating yarn shortages and demand across different modules, leading to data that didn't match between tables.

### Root Causes Identified
1. **Inconsistent Column Name Handling**
   - Some modules checked for `'Planning Balance'` (with space)
   - Others checked for `'Planning_Balance'` (underscore) or `'planning_balance'` (lowercase)
   - Actual data files use `'Planning Balance'` with space

2. **Different Shortage Calculation Logic**
   - Main ERP: Used `Planning Balance < 0` OR `Theoretical Balance < 0`
   - Yarn Blueprint: Only checked `Planning Balance < 0`
   - Some modules used absolute values, others used negative values differently

3. **Inconsistent BOM Aggregation**
   - Multiple methods for aggregating yarn requirements per style
   - No consistent handling of missing BOM mappings
   - Different calculation approaches across modules

### Solution Implemented
**Centralized Data Consistency Manager**: `src/data_consistency/consistency_manager.py`

Key Features:
- **Single Source of Truth**: All column name mappings in one place
- **Unified Shortage Calculation**: Consistent logic using both Planning and Theoretical Balance
- **Standardized Risk Levels**: CRITICAL (-1000+ lbs), HIGH (-500+ lbs), MEDIUM (-100+ lbs), LOW (<0 lbs)
- **Consistent BOM Aggregation**: Standardized method for calculating yarn requirements
- **Data Validation**: Built-in checks for data integrity

---

## Docker Quick Start Commands (Valuable Reference)

### Container Management
```bash
# View logs
docker logs -f bki-erp-minimal

# Restart application
docker-compose -f docker-compose.minimal.yml restart

# Stop application
docker-compose -f docker-compose.minimal.yml down

# Update and rebuild
git pull
docker-compose -f docker-compose.minimal.yml up -d --build
```

### Health Monitoring
```bash
# Check status
curl http://localhost:5005/api/comprehensive-kpis

# Monitor resources
docker stats bki-erp-minimal

# View recent activity
docker logs bki-erp-minimal --tail 100
```

### Troubleshooting Commands
```bash
# Container won't start - Check logs
docker logs bki-erp-minimal

# Check ports
netstat -tuln | grep 5005

# Data not loading - Clear cache
docker exec bki-erp-minimal rm -rf /app/cache/*

# Force reload
curl http://localhost:5005/api/reload-data

# High memory usage - Limit memory
docker update --memory="2g" bki-erp-minimal
```

### Configuration Files Reference
| File | Purpose |
|------|---------|
| `.env.docker` | Environment variables |
| `docker-compose.minimal.yml` | Minimal deployment (current) |
| `docker-compose.yml` | Full deployment with database |
| `docker-compose.prod.yml` | Production with all services |
| `Dockerfile.minimal` | Lightweight image (current) |
| `Dockerfile.optimized` | Full ML features |

---

## Security Checklist (Production Ready)
- [ ] Change default passwords in `.env.docker`
- [ ] Generate new SECRET_KEY: `openssl rand -hex 32`
- [ ] Enable HTTPS (use nginx proxy or cloud provider)
- [ ] Configure firewall rules
- [ ] Set up regular backups
- [ ] Enable monitoring alerts

---

## Deployment Platform Comparison
| Platform | Difficulty | Cost | Best For |
|----------|------------|------|----------|
| **ngrok** | Easy | Free/$8 | Testing & demos |
| **Railway** | Very Easy | $5+ | Quick deployment |
| **Render** | Easy | Free/$7+ | Small teams |
| **DigitalOcean** | Medium | $24+ | Growing business |
| **AWS EC2** | Hard | $30+ | Enterprise |
| **Azure** | Hard | $40+ | Microsoft stack |

---

## Performance Optimization Steps
1. **Enable Redis caching**:
   ```bash
   docker-compose up -d  # Uses full stack with Redis
   ```

2. **Use production build**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Scale horizontally**:
   ```bash
   docker-compose up -d --scale erp-app=3
   ```

---

## Port Configuration Notes
- **Current Server Port**: 5006 (NOT 5005 or 5003 as shown in some old docs)
- **QUICK_START.md references**: Port 5005 (OUTDATED)
- **PRODUCTION_DEPLOYMENT_GUIDE.md references**: Port 5005 (OUTDATED)
- **Correct Access**: http://localhost:5006/consolidated

---

## Critical API Endpoints (Post-Consolidation)
All working at `/api/`:
- `production-planning` - Production schedule with parameter support
- `inventory-intelligence-enhanced` - Inventory analytics with views
- `ml-forecast-detailed` - ML predictions with format options
- `inventory-netting` - Multi-level netting calculations
- `comprehensive-kpis` - Complete KPI metrics
- `yarn-intelligence` - Yarn analysis with shortage detection
- `production-suggestions` - AI-powered recommendations
- `po-risk-analysis` - Risk assessment
- `production-pipeline` - Real-time production flow

---

## Historical Context Notes
- **API Consolidation completed**: August 29, 2025
- **Emergency fixes implemented**: Day 0 fixes provide 75% system health
- **Current actual completion**: 75-80% (not 100% as claimed in various reports)
- **Production readiness**: NOT READY - Critical gaps identified
- **Main application size**: 18,020 lines (not 11,724 as claimed in some docs)

---

**Note**: This content was extracted from files marked for deletion during the documentation cleanup process of September 2025. The information above represents valuable technical content that was preserved before obsolete files were removed.