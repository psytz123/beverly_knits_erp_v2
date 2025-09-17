# Session Settings
SESSION_COOKIE_NAME=dancer.session
SESSION_STATE_PATH=/tmp/erp_session.json

**Primary wrapper endpoints (used by main ERP):**

- `GET /api/sales-order/plan/list' - Fetch sales orders 
- `GET /api/knitorder/list` - Fetch knit orders `
- `GET /api/yarn/active` - Active yarn inventory 
- `GET /api/styles` - Fetch styles from eFab

**QuadS Integration endpoints:**

- `GET /api/styles/greige/active` - Greige styles from QuadS 
- `GET /api/styles/finished/active` - Finished styles from QuadS

**Inventory Stage endpoints:**

- `GET /api/greige/g00` - Greige stage 1 inventory
- `GET /api/greige/g02` - Greige stage 2 inventory
- `GET /api/finished/i01` - QC/Inspection inventory
- `GET /api/finished/f01` - Finished goods inventory

**Reporting endpoints:**

- `GET /api/report/yarn_demand_ko` - Yarn demand (KO format)
- `GET /api/report/yarn_demand` - Standard yarn demand
- `GET /api/yarn-po` - Yarn purchase orders
- `GET /api/report/yarn_expected` - Expected yarn deliveries

# eFab ERP Configuration
ERP_BASE_URL=https://efab.bkiapps.com
ERP_LOGIN_URL=https://efab.bkiapps.com/login
ERP_API_PREFIX=/api

# QuadS Configuration
QUADS_BASE_URL=https://quads.bkiapps.com
QUADS_LOGIN_URL=https://quads.bkiapps.com/LOGIN
https://quads.bkiapps.com/knit-style/list/greige
https://quads.bkiapps.com/knit-style/list/finished

# Uses same credentials as eFab

# Login Credentials
ERP_USERNAME=psytz
ERP_PASSWORD=big$cat


