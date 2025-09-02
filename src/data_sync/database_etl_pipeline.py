#!/usr/bin/env python3
"""
ETL Pipeline for Beverly Knits ERP Database
Extracts data from Excel/CSV files and loads into PostgreSQL with TimescaleDB
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import logging
import json
import warnings
from typing import Dict, List, Tuple, Optional
import re

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BeverlyKnitsETL:
    """ETL Pipeline for Beverly Knits ERP Data"""
    
    def __init__(self, config_path: str = "database_config.json"):
        """Initialize ETL pipeline with database configuration"""
        self.config = self.load_config(config_path)
        self.conn = None
        self.cursor = None
        self.data_path = Path(self.config.get('data_path', '/mnt/c/Users/psytz/sc data/ERP Data'))
        self.snapshot_date = datetime.now().date()
        self.errors = []
        self.warnings = []
        
    def load_config(self, config_path: str) -> dict:
        """Load database configuration from file or use defaults"""
        default_config = {
            "host": "localhost",
            "port": 5432,
            "database": "beverly_knits_erp",
            "user": "postgres",
            "password": "postgres",
            "data_path": "/mnt/c/Users/psytz/sc data/ERP Data"
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")
    
    def log_etl_run(self, status: str, records_processed: int = 0, 
                    records_inserted: int = 0, records_updated: int = 0):
        """Log ETL run status to database"""
        try:
            self.cursor.execute("""
                INSERT INTO production.etl_log 
                (run_date, start_time, end_time, status, records_processed, 
                 records_inserted, records_updated, errors_count, error_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.snapshot_date,
                datetime.now(),
                datetime.now(),
                status,
                records_processed,
                records_inserted,
                records_updated,
                len(self.errors),
                json.dumps(self.errors) if self.errors else None
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log ETL run: {e}")
    
    def log_data_quality_issue(self, table_name: str, issue_type: str, 
                               description: str, severity: str = 'WARNING'):
        """Log data quality issues"""
        try:
            self.cursor.execute("""
                INSERT INTO production.data_quality_log
                (snapshot_date, table_name, issue_type, issue_description, severity)
                VALUES (%s, %s, %s, %s, %s)
            """, (self.snapshot_date, table_name, issue_type, description, severity))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log data quality issue: {e}")
    
    def standardize_date(self, date_val) -> Optional[str]:
        """Standardize date to MM/DD/YYYY format"""
        if pd.isna(date_val):
            return None
        
        try:
            # Try to parse the date
            if isinstance(date_val, str):
                # Try different date formats
                for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(date_val, fmt)
                        return dt.strftime('%m/%d/%Y')
                    except:
                        continue
            elif isinstance(date_val, (pd.Timestamp, datetime)):
                return date_val.strftime('%m/%d/%Y')
        except:
            pass
        
        return None
    
    def clean_currency(self, value) -> Optional[float]:
        """Clean currency values to decimal"""
        if pd.isna(value):
            return None
        
        try:
            # Remove currency symbols and convert
            if isinstance(value, str):
                value = re.sub(r'[$,]', '', value)
            return float(value)
        except:
            return None
    
    def get_latest_folder(self) -> Path:
        """Get the latest dated folder from ERP Data directory"""
        folders = []
        for item in self.data_path.iterdir():
            if item.is_dir() and re.match(r'\d+-\d+-\d+', item.name):
                # Parse folder date
                parts = item.name.split('-')
                try:
                    folder_date = datetime(2025, int(parts[0]), int(parts[1]))
                    folders.append((folder_date, item))
                except:
                    continue
        
        if folders:
            folders.sort(key=lambda x: x[0], reverse=True)
            return folders[0][1]
        
        return self.data_path / '8-28-2025'  # Default fallback
    
    def load_master_data(self):
        """Load master data files (Yarn_ID, Style_BOM, Supplier_ID)"""
        logger.info("Loading master data files...")
        
        # Load Supplier_ID.csv
        supplier_file = self.data_path / 'Supplier_ID.csv'
        if supplier_file.exists():
            try:
                df = pd.read_csv(supplier_file)
                for _, row in df.iterrows():
                    self.cursor.execute("""
                        INSERT INTO production.suppliers (supplier_code, supplier_name)
                        VALUES (%s, %s)
                        ON CONFLICT (supplier_code) DO UPDATE
                        SET supplier_name = EXCLUDED.supplier_name,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        str(row.get('Supplier_Code', row.get('Supplier', ''))),
                        str(row.get('Supplier_Name', row.get('Name', '')))
                    ))
                self.conn.commit()
                logger.info(f"Loaded {len(df)} suppliers")
            except Exception as e:
                logger.error(f"Failed to load suppliers: {e}")
                self.errors.append(f"Supplier load error: {e}")
        
        # Load Yarn_ID.csv
        yarn_file = self.data_path / 'Yarn_ID.csv'
        if yarn_file.exists():
            try:
                df = pd.read_csv(yarn_file)
                
                # Check for missing yarn IDs
                missing_count = df['Desc#'].isna().sum()
                if missing_count > 0:
                    self.log_data_quality_issue('yarns', 'MISSING_ID', 
                        f"{missing_count} yarns with missing Desc#", 'WARNING')
                
                for _, row in df.iterrows():
                    if pd.notna(row.get('Desc#')):
                        # Get supplier_id if exists
                        supplier_id = None
                        if pd.notna(row.get('Supplier')):
                            self.cursor.execute(
                                "SELECT supplier_id FROM production.suppliers WHERE supplier_name = %s LIMIT 1",
                                (str(row['Supplier']),)
                            )
                            result = self.cursor.fetchone()
                            if result:
                                supplier_id = result[0]
                        
                        self.cursor.execute("""
                            INSERT INTO production.yarns 
                            (desc_id, supplier_id, yarn_description, blend, yarn_type, 
                             color, cost_per_pound)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (desc_id) DO UPDATE
                            SET supplier_id = EXCLUDED.supplier_id,
                                yarn_description = EXCLUDED.yarn_description,
                                blend = EXCLUDED.blend,
                                yarn_type = EXCLUDED.yarn_type,
                                color = EXCLUDED.color,
                                cost_per_pound = EXCLUDED.cost_per_pound,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            int(row['Desc#']),
                            supplier_id,
                            str(row.get('Description', '')),
                            str(row.get('Blend', '')),
                            str(row.get('Type', '')),
                            str(row.get('Color', '')),
                            self.clean_currency(row.get('Cost_Pound'))
                        ))
                self.conn.commit()
                logger.info(f"Loaded {len(df)} yarns")
            except Exception as e:
                logger.error(f"Failed to load yarns: {e}")
                self.errors.append(f"Yarn load error: {e}")
        
        # Load Style_BOM.csv
        bom_file = self.data_path / 'Style_BOM.csv'
        if bom_file.exists():
            try:
                df = pd.read_csv(bom_file)
                
                # First, ensure styles exist
                unique_styles = df['Style#'].dropna().unique()
                for style in unique_styles:
                    self.cursor.execute("""
                        INSERT INTO production.styles (style_number, style_description)
                        VALUES (%s, %s)
                        ON CONFLICT (style_number) DO NOTHING
                    """, (str(style), f"Style {style}"))
                
                self.conn.commit()
                
                # Check for BOM entries with missing yarn references
                missing_yarns = []
                for _, row in df.iterrows():
                    if pd.notna(row.get('Yarn_ID')):
                        self.cursor.execute(
                            "SELECT yarn_id FROM production.yarns WHERE desc_id = %s",
                            (int(row['Yarn_ID']),)
                        )
                        if not self.cursor.fetchone():
                            missing_yarns.append(int(row['Yarn_ID']))
                
                if missing_yarns:
                    self.log_data_quality_issue('style_bom', 'MISSING_YARN_REF',
                        f"{len(set(missing_yarns))} yarn IDs in BOM not found in yarn master",
                        'ERROR')
                
                # Load BOM entries
                for _, row in df.iterrows():
                    if pd.notna(row.get('Style#')) and pd.notna(row.get('Yarn_ID')):
                        # Get style_id
                        self.cursor.execute(
                            "SELECT style_id FROM production.styles WHERE style_number = %s",
                            (str(row['Style#']),)
                        )
                        style_result = self.cursor.fetchone()
                        
                        # Get yarn_id
                        self.cursor.execute(
                            "SELECT yarn_id FROM production.yarns WHERE desc_id = %s",
                            (int(row['Yarn_ID']),)
                        )
                        yarn_result = self.cursor.fetchone()
                        
                        if style_result and yarn_result:
                            self.cursor.execute("""
                                INSERT INTO production.style_bom 
                                (style_id, yarn_id, bom_percent, unit)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (style_id, yarn_id) DO UPDATE
                                SET bom_percent = EXCLUDED.bom_percent,
                                    unit = EXCLUDED.unit,
                                    updated_at = CURRENT_TIMESTAMP
                            """, (
                                style_result[0],
                                yarn_result[0],
                                float(row.get('BOM%', 0)),
                                str(row.get('unit', 'lbs'))
                            ))
                
                self.conn.commit()
                logger.info(f"Loaded {len(df)} BOM entries")
            except Exception as e:
                logger.error(f"Failed to load BOM: {e}")
                self.errors.append(f"BOM load error: {e}")
    
    def load_daily_data(self):
        """Load daily refresh data from latest folder"""
        latest_folder = self.get_latest_folder()
        logger.info(f"Loading daily data from: {latest_folder}")
        
        # Load yarn inventory
        self.load_yarn_inventory(latest_folder)
        
        # Load fabric inventory (all stages)
        self.load_fabric_inventory(latest_folder)
        
        # Load sales orders
        self.load_sales_orders(latest_folder)
        
        # Load knit orders
        self.load_knit_orders(latest_folder)
        
        # Load yarn demand
        self.load_yarn_demand(latest_folder)
    
    def load_yarn_inventory(self, folder: Path):
        """Load yarn inventory data"""
        file_path = folder / 'yarn_inventory.xlsx'
        if not file_path.exists():
            # Try alternative naming
            for alt_name in ['yarn_inventory (1).xlsx', 'yarn_inventory (2).xlsx']:
                alt_path = folder / alt_name
                if alt_path.exists():
                    file_path = alt_path
                    break
        
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                records_inserted = 0
                
                for _, row in df.iterrows():
                    if pd.notna(row.get('Desc#')):
                        # Get yarn_id
                        self.cursor.execute(
                            "SELECT yarn_id FROM production.yarns WHERE desc_id = %s",
                            (int(row['Desc#']),)
                        )
                        yarn_result = self.cursor.fetchone()
                        
                        if yarn_result:
                            self.cursor.execute("""
                                INSERT INTO production.yarn_inventory_ts
                                (snapshot_date, yarn_id, desc_id, theoretical_balance,
                                 allocated, on_order, consumed_qty, weeks_of_supply,
                                 cost_per_pound, data_source)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                self.snapshot_date,
                                yarn_result[0],
                                int(row['Desc#']),
                                self.clean_currency(row.get('Theoretical Balance', 0)),
                                self.clean_currency(row.get('Allocated', 0)),
                                self.clean_currency(row.get('On Order', 0)),
                                self.clean_currency(row.get('Consumed', 0)),
                                float(row.get('Weeks of Supply', 0)) if pd.notna(row.get('Weeks of Supply')) else None,
                                self.clean_currency(row.get('Cost/Pound')),
                                str(file_path.name)
                            ))
                            records_inserted += 1
                
                self.conn.commit()
                logger.info(f"Loaded {records_inserted} yarn inventory records")
            except Exception as e:
                logger.error(f"Failed to load yarn inventory: {e}")
                self.errors.append(f"Yarn inventory load error: {e}")
    
    def load_fabric_inventory(self, folder: Path):
        """Load fabric inventory for all stages"""
        stages = ['F01', 'G00', 'G02', 'I01']
        
        for stage in stages:
            file_path = folder / f'eFab_Inventory_{stage}.xlsx'
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    records_inserted = 0
                    
                    for _, row in df.iterrows():
                        fstyle = str(row.get('fStyle#', ''))
                        if fstyle:
                            # Get or create style
                            self.cursor.execute("""
                                INSERT INTO production.styles (style_number, fstyle_number)
                                VALUES (%s, %s)
                                ON CONFLICT (style_number) DO UPDATE
                                SET fstyle_number = EXCLUDED.fstyle_number
                                RETURNING style_id
                            """, (fstyle, fstyle))
                            style_id = self.cursor.fetchone()[0]
                            
                            # Get customer if exists
                            customer_code = str(row.get('Customer', ''))
                            if customer_code:
                                self.cursor.execute("""
                                    INSERT INTO production.customers (customer_code, customer_name)
                                    VALUES (%s, %s)
                                    ON CONFLICT (customer_code) DO NOTHING
                                """, (customer_code, customer_code))
                            
                            self.cursor.execute("""
                                INSERT INTO production.fabric_inventory_ts
                                (snapshot_date, style_id, fstyle_number, inventory_stage,
                                 quantity_lbs, location, status, customer_code, data_source)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                self.snapshot_date,
                                style_id,
                                fstyle,
                                stage,
                                self.clean_currency(row.get('Quantity', 0)),
                                str(row.get('Location', '')),
                                str(row.get('Status', '')),
                                customer_code,
                                str(file_path.name)
                            ))
                            records_inserted += 1
                    
                    self.conn.commit()
                    logger.info(f"Loaded {records_inserted} {stage} inventory records")
                except Exception as e:
                    logger.error(f"Failed to load {stage} inventory: {e}")
                    self.errors.append(f"{stage} inventory load error: {e}")
    
    def load_sales_orders(self, folder: Path):
        """Load sales orders data"""
        file_path = folder / 'eFab_SO_List.xlsx'
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                records_inserted = 0
                
                for _, row in df.iterrows():
                    so_number = str(row.get('SO #', ''))
                    if so_number:
                        # Get or create customer
                        customer_code = str(row.get('Customer', ''))
                        customer_id = None
                        if customer_code:
                            self.cursor.execute("""
                                INSERT INTO production.customers (customer_code, customer_name)
                                VALUES (%s, %s)
                                ON CONFLICT (customer_code) DO NOTHING
                                RETURNING customer_id
                            """, (customer_code, customer_code))
                            result = self.cursor.fetchone()
                            if result:
                                customer_id = result[0]
                            else:
                                self.cursor.execute(
                                    "SELECT customer_id FROM production.customers WHERE customer_code = %s",
                                    (customer_code,)
                                )
                                customer_id = self.cursor.fetchone()[0]
                        
                        # Get style
                        style_number = str(row.get('Style#', ''))
                        fstyle_number = str(row.get('fStyle#', ''))
                        style_id = None
                        
                        if style_number or fstyle_number:
                            self.cursor.execute("""
                                INSERT INTO production.styles (style_number, fstyle_number)
                                VALUES (%s, %s)
                                ON CONFLICT (style_number) DO UPDATE
                                SET fstyle_number = COALESCE(EXCLUDED.fstyle_number, production.styles.fstyle_number)
                                RETURNING style_id
                            """, (style_number or fstyle_number, fstyle_number))
                            style_id = self.cursor.fetchone()[0]
                        
                        self.cursor.execute("""
                            INSERT INTO production.sales_orders_ts
                            (snapshot_date, so_number, customer_id, style_id,
                             fstyle_number, style_number, order_status, uom,
                             quantity_ordered, quantity_picked, quantity_shipped,
                             balance, available_qty, ship_date, po_number, data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            self.snapshot_date,
                            so_number,
                            customer_id,
                            style_id,
                            fstyle_number,
                            style_number,
                            str(row.get('Status', '')),
                            str(row.get('UOM', '')),
                            self.clean_currency(row.get('Ordered', 0)),
                            self.clean_currency(row.get('Picked', 0)),
                            self.clean_currency(row.get('Shipped', 0)),
                            self.clean_currency(row.get('Balance', 0)),
                            self.clean_currency(row.get('Available', 0)),
                            pd.to_datetime(row.get('Ship Date')) if pd.notna(row.get('Ship Date')) else None,
                            str(row.get('PO#', '')),
                            str(file_path.name)
                        ))
                        records_inserted += 1
                
                self.conn.commit()
                logger.info(f"Loaded {records_inserted} sales order records")
            except Exception as e:
                logger.error(f"Failed to load sales orders: {e}")
                self.errors.append(f"Sales orders load error: {e}")
    
    def load_knit_orders(self, folder: Path):
        """Load knit orders data"""
        file_path = folder / 'eFab_Knit_Orders.xlsx'
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                records_inserted = 0
                
                for _, row in df.iterrows():
                    ko_number = str(row.get('Actions', ''))
                    if ko_number:
                        # Get style
                        style_number = str(row.get('Style#', ''))
                        style_id = None
                        
                        if style_number:
                            self.cursor.execute("""
                                INSERT INTO production.styles (style_number)
                                VALUES (%s)
                                ON CONFLICT (style_number) DO NOTHING
                                RETURNING style_id
                            """, (style_number,))
                            result = self.cursor.fetchone()
                            if result:
                                style_id = result[0]
                            else:
                                self.cursor.execute(
                                    "SELECT style_id FROM production.styles WHERE style_number = %s",
                                    (style_number,)
                                )
                                style_id = self.cursor.fetchone()[0]
                        
                        self.cursor.execute("""
                            INSERT INTO production.knit_orders_ts
                            (snapshot_date, ko_number, style_id, style_number,
                             start_date, quoted_date, qty_ordered_lbs, g00_lbs,
                             shipped_lbs, balance_lbs, seconds_lbs, machine,
                             po_number, data_source)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            self.snapshot_date,
                            ko_number,
                            style_id,
                            style_number,
                            pd.to_datetime(row.get('Start Date')) if pd.notna(row.get('Start Date')) else None,
                            pd.to_datetime(row.get('Quoted Date')) if pd.notna(row.get('Quoted Date')) else None,
                            self.clean_currency(row.get('Qty Ordered (lbs)', 0)),
                            self.clean_currency(row.get('G00 (lbs)', 0)),
                            self.clean_currency(row.get('Shipped (lbs)', 0)),
                            self.clean_currency(row.get('Balance (lbs)', 0)),
                            self.clean_currency(row.get('Seconds (lbs)', 0)),
                            str(row.get('Machine', '')),
                            str(row.get('PO#', '')),
                            str(file_path.name)
                        ))
                        records_inserted += 1
                
                self.conn.commit()
                logger.info(f"Loaded {records_inserted} knit order records")
            except Exception as e:
                logger.error(f"Failed to load knit orders: {e}")
                self.errors.append(f"Knit orders load error: {e}")
    
    def load_yarn_demand(self, folder: Path):
        """Load yarn demand data"""
        file_path = folder / 'Yarn_Demand.xlsx'
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                records_inserted = 0
                
                # Get column names for week data
                week_columns = [col for col in df.columns if 'Week' in str(col) or col.startswith('W')]
                
                for _, row in df.iterrows():
                    desc_id = row.get('Desc#')
                    if pd.notna(desc_id):
                        # Get yarn_id
                        self.cursor.execute(
                            "SELECT yarn_id FROM production.yarns WHERE desc_id = %s",
                            (int(desc_id),)
                        )
                        yarn_result = self.cursor.fetchone()
                        
                        if yarn_result:
                            # Process each week
                            for week_num, week_col in enumerate(week_columns, 1):
                                demand_qty = self.clean_currency(row.get(week_col, 0))
                                if demand_qty and demand_qty > 0:
                                    week_date = self.snapshot_date + timedelta(weeks=week_num-1)
                                    
                                    self.cursor.execute("""
                                        INSERT INTO production.yarn_demand_ts
                                        (snapshot_date, yarn_id, desc_id, week_number,
                                         week_date, demand_qty, data_source)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (
                                        self.snapshot_date,
                                        yarn_result[0],
                                        int(desc_id),
                                        week_num,
                                        week_date,
                                        demand_qty,
                                        str(file_path.name)
                                    ))
                                    records_inserted += 1
                
                self.conn.commit()
                logger.info(f"Loaded {records_inserted} yarn demand records")
            except Exception as e:
                logger.error(f"Failed to load yarn demand: {e}")
                self.errors.append(f"Yarn demand load error: {e}")
    
    def refresh_views(self):
        """Refresh materialized views"""
        try:
            self.cursor.execute("SELECT api.refresh_all_views()")
            self.conn.commit()
            logger.info("Materialized views refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh views: {e}")
    
    def run(self):
        """Execute the complete ETL pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Beverly Knits ETL Pipeline")
        logger.info(f"Snapshot Date: {self.snapshot_date}")
        logger.info("=" * 50)
        
        try:
            # Connect to database
            self.connect()
            
            # Start ETL log
            self.log_etl_run('RUNNING')
            
            # Load master data (no daily refresh)
            logger.info("Phase 1: Loading master data...")
            self.load_master_data()
            
            # Load daily data
            logger.info("Phase 2: Loading daily data...")
            self.load_daily_data()
            
            # Refresh materialized views
            logger.info("Phase 3: Refreshing views...")
            self.refresh_views()
            
            # Log successful completion
            self.log_etl_run('SUCCESS', 
                            records_processed=1000,  # Update with actual counts
                            records_inserted=900,
                            records_updated=100)
            
            logger.info("=" * 50)
            logger.info("ETL Pipeline completed successfully")
            if self.warnings:
                logger.warning(f"Warnings: {len(self.warnings)}")
                for warning in self.warnings[:5]:
                    logger.warning(f"  - {warning}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            self.log_etl_run('FAILED')
            raise
        finally:
            # Disconnect from database
            self.disconnect()

def main():
    """Main entry point"""
    # Check for config file argument
    config_path = "database_config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Run ETL pipeline
    etl = BeverlyKnitsETL(config_path)
    etl.run()

if __name__ == "__main__":
    main()