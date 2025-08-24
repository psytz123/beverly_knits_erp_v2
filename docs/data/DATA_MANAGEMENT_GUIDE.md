# Beverly Knits ERP - Data Management Guide

## 🔒 EXCLUSIVE Data Source Configuration

The ERP system is configured to use **ONLY** the SharePoint ERP Data folder:
- **URL**: https://beverlyknits-my.sharepoint.com/:f:/r/personal/psytz_beverlyknits_com/Documents/ERP%20Data
- **Local Sync**: `C:\finalee\Agent-MCP-1-ddd\Agent-MCP-1-dd\ERP Data\sharepoint_sync`

## 📅 Daily Data Management Process

### How It Works:
1. **Daily Updates**: SharePoint folder is updated daily with new data
2. **Old Data Deleted**: Previous versions are removed from SharePoint
3. **Automatic Sync**: ERP checks for new data on startup
4. **Data Cleaning**: All files are automatically cleaned and validated
5. **Ready to Use**: Clean, standardized data for the ERP

### Automatic Process on ERP Startup:
```
1. Check if today's data is synced
2. If not, open SharePoint for download
3. Extract downloaded ZIP file
4. Clean and validate all data files
5. Start ERP with clean data
```

## 🧹 Data Cleaning & Validation

The system automatically cleans each file to ensure:

### 1. **Column Name Standardization**
- `Desc#`, `Yarn_ID`, `YarnID` → **`Desc#`**
- `Planning_Ballance` (typo) → **`Planning_Balance`**
- `Style Number`, `Style #` → **`Style#`**
- `On Order`, `On-Order` → **`On_Order`**

### 2. **Data Type Validation**
- Numeric columns: Remove commas, convert to numbers
- Date columns: Parse and validate dates
- Text columns: Trim whitespace, remove 'nan'
- Unit columns: Standardize (LB→LBS, YD→YDS)

### 3. **Critical Column Validation**
Each file type has required columns that are checked:
- **yarn_inventory**: Desc#, Planning_Balance, Unit
- **style_bom**: Style#, Desc#, Qty
- **sales_activity**: Style#, Order_Date, Qty
- **efab_inventory**: fStyle#, Qty
- **efab_styles**: fStyle#, Style#
- **knit_orders**: Order#, Style#, Qty

### 4. **File Type Detection**
Files are automatically identified by:
- Filename patterns (yarn_inventory, style_bom, etc.)
- Column content analysis
- Standard naming conventions

## 📊 Data Quality Reports

After each sync, a detailed report is generated:
- Total files processed
- Corrections made per file
- Data quality issues found
- Row counts before/after cleaning
- Timestamp and backup locations

## 🛠️ Manual Commands

### Check Sync Status:
```bash
cd Agent-MCP-1-dd/BKI_comp
python daily_data_sync.py --status
```

### Manual Sync (if you downloaded the ZIP):
```bash
python daily_data_sync.py --file "C:\Users\YourName\Downloads\ERP_Data.zip"
```

### Run Data Cleaning Only:
```bash
python data_parser_cleaner.py --report
```

### Force Download Latest Data:
```bash
python daily_data_sync.py --auto
```

## 📁 Directory Structure

```
ERP Data/
├── sharepoint_sync/              # Active data directory
│   ├── yarn_inventory.xlsx       # Cleaned files
│   ├── style_bom.csv
│   ├── sales_activity.csv
│   ├── cleaned/                  # Temporary cleaning directory
│   ├── .daily_sync_status.json   # Sync tracking
│   └── cleaning_report_*.json    # Data quality reports
├── backup_*/                     # Automatic backups
└── 5_backup_*/                   # Old data source backups
```

## ⚠️ Important Notes

1. **Never manually edit files** in the sharepoint_sync directory
2. **Data is replaced daily** - don't store custom data here
3. **Backups are automatic** before each sync
4. **Cleaning is mandatory** - ensures data consistency

## 🚨 Troubleshooting

### If sync fails:
1. Check internet connection
2. Verify SharePoint URL is accessible
3. Look for the download in your Downloads folder
4. Run manual sync with the downloaded file

### If data looks wrong:
1. Check the cleaning report
2. Verify source file format
3. Look for validation warnings
4. Check column mapping in data_parser_cleaner.py

### If ERP won't start:
1. Run `python daily_data_sync.py --status`
2. Check for sync errors
3. Manually download and sync if needed
4. Check data cleaning logs

## 📈 Benefits of This System

1. **Data Consistency**: Same column names every day
2. **Type Safety**: Numbers are numbers, dates are dates
3. **No Manual Cleaning**: Automatic standardization
4. **Error Prevention**: Validation catches issues early
5. **Audit Trail**: Complete logs of all changes
6. **Backup Safety**: Never lose data during updates

The system ensures your ERP always has clean, validated, up-to-date data!