# API Endpoint Reference Documentation

## Overview
This document provides the current API endpoints for the Beverly Knits ERP system, implementing the integrated eFab/QuadS API scheme.

## Current System Architecture

The system operates as a unified ERP with direct API integrations to:
- **eFab Platform**: Primary production and order management
- **QuadS System**: Style and greige fabric management
- **Yarn Intelligence**: Automated yarn management and substitution

## Active API Endpoints

### Primary ERP Wrapper Endpoints
These endpoints provide direct integration with eFab and QuadS systems:

### Sales Order Management
- `GET /api/sales-order/plan/list` - Fetch sales orders from eFab
- `GET /api/knitorder/list` - Fetch knit orders from eFab

### Yarn & Inventory Management
- `GET /api/yarn/active` - Active yarn inventory from eFab
- `GET /api/greige/g00` - Greige stage 1 inventory
- `GET /api/greige/g02` - Greige stage 2 inventory
- `GET /api/finished/i01` - QC/Inspection inventory
- `GET /api/finished/f01` - Finished goods inventory

### Style Management (QuadS Integration)
- `GET /api/styles` - Fetch styles from eFab
- `GET /api/styles/greige/active` - Greige styles from QuadS
- `GET /api/styles/finished/active` - Finished styles from QuadS

### Reporting & Analytics
- `GET /api/report/yarn_demand_ko` - Yarn demand (KO format)
- `GET /api/report/yarn_demand` - Standard yarn demand report
- `GET /api/yarn-po` - Yarn purchase orders
- `GET /api/report/yarn_expected` - Expected yarn deliveries

## System Configuration

### eFab Platform Configuration
```
ERP_BASE_URL=https://efab.bkiapps.com
ERP_LOGIN_URL=https://efab.bkiapps.com/login
ERP_API_PREFIX=/api
ERP_USERNAME=psytz
ERP_PASSWORD=big$cat
```

### QuadS Platform Configuration
```
QUADS_BASE_URL=https://quads.bkiapps.com
QUADS_LOGIN_URL=https://quads.bkiapps.com/LOGIN
Greige Styles: https://quads.bkiapps.com/knit-style/list/greige
Finished Styles: https://quads.bkiapps.com/knit-style/list/finished
```

### Session Management
```
SESSION_COOKIE_NAME=dancer.session
SESSION_STATE_PATH=/tmp/erp_session.json
```

## Data Flow Architecture

The system operates using direct API calls rather than file uploads:

```
eFab API → Real-time Data → ERP Processing → Dashboard Display
QuadS API → Style Data → Integration Layer → Reporting
```

## Legacy File Upload Migration

**Previous System**: Required manual file uploads for inventory, BOM, and sales data
**Current System**: All data is fetched automatically via APIs from eFab and QuadS platforms

### Migrated Data Sources
- **Yarn Inventory**: Now fetched from `GET /api/yarn/active`
- **Production Orders**: Now fetched from `GET /api/knitorder/list`
- **Sales Data**: Now fetched from `GET /api/sales-order/plan/list`
- **Style Information**: Now fetched from QuadS via `GET /api/styles/*` endpoints

No manual file uploads are required for normal operations.

## Authentication & Security

### API Authentication
The system uses session-based authentication with the eFab and QuadS platforms. Credentials are configured in environment variables and managed through session cookies.

### Session Management
- Cookie-based session management
- Automatic session refresh
- Secure credential storage

## Integration Benefits

### Eliminated Manual Processes
- ❌ **No more file uploads**: All data is fetched automatically
- ❌ **No data synchronization**: Real-time integration
- ❌ **No format conversion**: Direct API data consumption
- ❌ **No scheduled imports**: Continuous data flow

### Real-time Capabilities
- ✅ **Live inventory data** from eFab
- ✅ **Real-time order status** updates
- ✅ **Dynamic style information** from QuadS
- ✅ **Instant yarn demand reports**

## Implementation Notes

### Data Sources
All data is now sourced directly from live systems:
- **eFab**: Production orders, inventory, sales orders
- **QuadS**: Styles, greige fabrics, finished goods
- **Local Processing**: ML forecasting, analytics, reporting

### Migration from File-based System
This system replaces the previous file-upload architecture with direct API integration, eliminating the need for:
- Manual CSV/Excel file uploads
- Data validation and cleansing
- File format standardization
- Import scheduling and monitoring

## Technical Architecture

### Current System Design
The Beverly Knits ERP operates as a unified system with:

```
eFab Platform API ──→
                      ├── Beverly Knits ERP ──→ Dashboard & Analytics
QuadS Platform API ──→
```

### Key Components
1. **API Integration Layer**: Handles authentication and data fetching from external platforms
2. **Data Processing Engine**: Processes and analyzes data in real-time
3. **ML Forecasting System**: Provides predictive analytics
4. **Dashboard Interface**: Web-based user interface for data visualization

### Benefits of Current Architecture
- **Real-time Data**: No delays from batch imports
- **Reduced Complexity**: Eliminates file management overhead
- **Better Data Quality**: Direct from source systems
- **Automatic Updates**: Always current data
- **Scalability**: API-based integration scales better than file processing

## Future Enhancements

### Planned API Extensions
- Additional reporting endpoints
- Enhanced analytics capabilities
- Real-time notifications via webhooks
- Mobile API endpoints

### Integration Roadmap
- Extended eFab API coverage
- Additional QuadS functionality
- Third-party logistics integration
- Customer portal APIs

---

**Last Updated**: 2025-09-15
**Status**: Production Active with eFab/QuadS Integration
**Architecture**: Unified ERP with Real-time API Integration