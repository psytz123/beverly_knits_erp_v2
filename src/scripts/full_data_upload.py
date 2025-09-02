#!/usr/bin/env python3
"""
Full Data Upload Script for Beverly Knits ERP
Uploads all data from files to PostgreSQL database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullDataUploader:
    """Upload all ERP data to PostgreSQL database"""
    
    def __init__(self):
        """Initialize uploader with database connection"""
        self.config = {
            "host": "localhost",
            "port": 5432,
            "database": "beverly_knits_erp",
            "user": "erp_user",
            "password": "erp_password"
        }
        self.data_path = Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5")
        self.snapshot_date = datetime.now().date()
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database disconnected")
    
    def upload_yarns(self):
        """Upload all yarn data"""
        logger.info("Uploading yarn data...")
        try:
            # Load yarn inventory file
            yarn_file = self.data_path / "yarn_inventory.xlsx"
            if not yarn_file.exists():
                yarn_file = self.data_path / "Yarn_Inventory_latest.xlsx"
            
            df = pd.read_excel(yarn_file)
            logger.info(f"Loaded {len(df)} yarn records from {yarn_file}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # First, insert/update yarns master data
            yarn_count = 0
            for _, row in df.iterrows():
                desc_id = str(row.get('Desc#', row.get('desc#', ''))).strip()
                if desc_id and desc_id != 'nan':
                    # Insert or update yarn master
                    self.cursor.execute("""
                        INSERT INTO production.yarns (desc_id, yarn_description, blend, yarn_type, is_active)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (desc_id) DO UPDATE
                        SET yarn_description = EXCLUDED.yarn_description,
                            blend = COALESCE(EXCLUDED.blend, production.yarns.blend),
                            yarn_type = COALESCE(EXCLUDED.yarn_type, production.yarns.yarn_type)
                        RETURNING yarn_id
                    """, (
                        desc_id,
                        str(row.get('Yarn Description', '')),
                        str(row.get('Blend', '')),
                        str(row.get('Type', 'YARN')),
                        True
                    ))
                    yarn_id = self.cursor.fetchone()[0]
                    
                    # Insert inventory data (planning_balance is generated, so exclude it)
                    self.cursor.execute("""
                        INSERT INTO production.yarn_inventory_ts 
                        (yarn_id, theoretical_balance, allocated, on_order,
                         weeks_of_supply, cost_per_pound, snapshot_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (yarn_id, snapshot_date) DO UPDATE
                        SET theoretical_balance = EXCLUDED.theoretical_balance,
                            allocated = EXCLUDED.allocated,
                            on_order = EXCLUDED.on_order,
                            weeks_of_supply = EXCLUDED.weeks_of_supply,
                            cost_per_pound = EXCLUDED.cost_per_pound
                    """, (
                        yarn_id,
                        float(row.get('Theoretical Balance', 0) or 0),
                        float(row.get('Allocated', 0) or 0),
                        float(row.get('On Order', 0) or 0),
                        float(row.get('Weeks of Supply', 0) or 0),
                        float(row.get('Cost/Pound', row.get('cost', 0)) or 0),
                        self.snapshot_date
                    ))
                    yarn_count += 1
            
            self.conn.commit()
            logger.info(f"✓ Uploaded {yarn_count} yarn records")
            return yarn_count
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error uploading yarns: {e}")
            raise
    
    def upload_sales_orders(self):
        """Upload all sales orders"""
        logger.info("Uploading sales orders...")
        try:
            # Load sales activity file
            sales_file = self.data_path / "Sales Activity Report.csv"
            if not sales_file.exists():
                logger.warning(f"Sales file not found: {sales_file}")
                return 0
            
            df = pd.read_csv(sales_file)
            logger.info(f"Loaded {len(df)} sales records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            order_count = 0
            for _, row in df.iterrows():
                so_number = str(row.get('SO#', row.get('Document', ''))).strip()
                if so_number and so_number != 'nan':
                    # Get or create customer
                    customer_name = str(row.get('Sold To', row.get('Customer', 'Unknown')))
                    # First try to find existing customer
                    self.cursor.execute("""
                        SELECT customer_id FROM production.customers 
                        WHERE customer_name = %s
                    """, (customer_name,))
                    result = self.cursor.fetchone()
                    
                    if result:
                        customer_id = result[0]
                    else:
                        # Create new customer with auto-generated ID
                        customer_code = customer_name[:20].upper().replace(' ', '_')
                        self.cursor.execute("""
                            INSERT INTO production.customers (customer_name, customer_code)
                            VALUES (%s, %s)
                            RETURNING customer_id
                        """, (customer_name, customer_code))
                        customer_id = self.cursor.fetchone()[0]
                    
                    # Get or create style
                    style_number = str(row.get('fStyle#', row.get('Style#', '')))
                    self.cursor.execute("""
                        INSERT INTO production.styles (style_number, fstyle_number, style_description)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (style_number) DO UPDATE
                        SET fstyle_number = COALESCE(EXCLUDED.fstyle_number, production.styles.fstyle_number)
                        RETURNING style_id
                    """, (style_number, style_number, style_number))
                    style_id = self.cursor.fetchone()[0]
                    
                    # Parse ship date
                    ship_date = None
                    if pd.notna(row.get('Ship Date')):
                        try:
                            ship_date = pd.to_datetime(row.get('Ship Date')).date()
                        except:
                            pass
                    
                    # Insert sales order
                    self.cursor.execute("""
                        INSERT INTO production.sales_orders_ts
                        (so_number, customer_id, style_id, style_number, quantity_ordered,
                         quantity_shipped, balance, ship_date, available_qty, snapshot_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (so_number, snapshot_date) DO UPDATE
                        SET quantity_ordered = EXCLUDED.quantity_ordered,
                            quantity_shipped = EXCLUDED.quantity_shipped,
                            balance = EXCLUDED.balance,
                            ship_date = EXCLUDED.ship_date
                    """, (
                        so_number,
                        customer_id,
                        style_id,
                        style_number,
                        float(row.get('Ordered', row.get('Yds_ordered', 0)) or 0),
                        float(row.get('Picked/Shipped', 0) or 0),
                        float(row.get('Balance', row.get('Yds_ordered', 0)) or 0),
                        ship_date,
                        float(row.get('Available', 0) or 0),
                        self.snapshot_date
                    ))
                    order_count += 1
            
            self.conn.commit()
            logger.info(f"✓ Uploaded {order_count} sales orders")
            return order_count
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error uploading sales orders: {e}")
            raise
    
    def upload_knit_orders(self):
        """Upload all knit orders"""
        logger.info("Uploading knit orders...")
        try:
            # Load knit orders file
            ko_file = self.data_path / "eFab_Knit_Orders.xlsx"
            if not ko_file.exists():
                logger.warning(f"Knit orders file not found: {ko_file}")
                return 0
            
            df = pd.read_excel(ko_file)
            logger.info(f"Loaded {len(df)} knit orders")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            ko_count = 0
            for _, row in df.iterrows():
                ko_number = str(row.get('KO#', row.get('Order #', ''))).strip()
                # Clean HTML from ko_number
                if '&nbsp;' in ko_number or '<' in ko_number:
                    ko_number = ko_number.split('&nbsp;')[0].split('<')[0].strip()
                if ko_number and ko_number != 'nan' and ko_number != '':
                    # Get or create style
                    style_number = str(row.get('Style#', row.get('Style #', '')))
                    self.cursor.execute("""
                        INSERT INTO production.styles (style_number, fstyle_number, style_description)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (style_number) DO NOTHING
                        RETURNING style_id
                    """, (style_number, style_number, style_number))
                    result = self.cursor.fetchone()
                    
                    if not result:
                        self.cursor.execute("SELECT style_id FROM production.styles WHERE style_number = %s", (style_number,))
                        result = self.cursor.fetchone()
                    
                    style_id = result[0] if result else None
                    
                    # Parse dates
                    start_date = None
                    quoted_date = None
                    if pd.notna(row.get('Start Date')):
                        try:
                            start_date = pd.to_datetime(row.get('Start Date')).date()
                        except:
                            pass
                    if pd.notna(row.get('Quoted Date')):
                        try:
                            quoted_date = pd.to_datetime(row.get('Quoted Date')).date()
                        except:
                            pass
                    
                    # Insert knit order
                    self.cursor.execute("""
                        INSERT INTO production.knit_orders_ts
                        (ko_number, style_id, style_number, qty_ordered_lbs, g00_lbs,
                         shipped_lbs, balance_lbs, machine, start_date, quoted_date, snapshot_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ko_number, snapshot_date) DO UPDATE
                        SET qty_ordered_lbs = EXCLUDED.qty_ordered_lbs,
                            g00_lbs = EXCLUDED.g00_lbs,
                            shipped_lbs = EXCLUDED.shipped_lbs,
                            balance_lbs = EXCLUDED.balance_lbs
                    """, (
                        ko_number,
                        style_id,
                        style_number,
                        float(row.get('Qty Ordered (lbs)', 0) or 0),
                        float(row.get('G00 (lbs)', 0) or 0),
                        float(row.get('Shipped (lbs)', 0) or 0),
                        float(row.get('Balance (lbs)', 0) or 0),
                        str(row.get('Machine', '')),
                        start_date,
                        quoted_date,
                        self.snapshot_date
                    ))
                    ko_count += 1
            
            self.conn.commit()
            logger.info(f"✓ Uploaded {ko_count} knit orders")
            return ko_count
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error uploading knit orders: {e}")
            raise
    
    def upload_bom(self):
        """Upload all BOM data"""
        logger.info("Uploading BOM data...")
        try:
            # Load BOM file
            bom_file = self.data_path / "BOM_updated.csv"
            if not bom_file.exists():
                bom_file = self.data_path / "Style_BOM.csv"
            
            if not bom_file.exists():
                logger.warning(f"BOM file not found")
                return 0
            
            df = pd.read_csv(bom_file)
            logger.info(f"Loaded {len(df)} BOM records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            bom_count = 0
            for _, row in df.iterrows():
                style_number = str(row.get('Style#', '')).strip()
                desc_id = str(row.get('desc#', row.get('Desc#', ''))).strip()
                
                if style_number and style_number != 'nan' and desc_id and desc_id != 'nan':
                    # Get style_id
                    self.cursor.execute("""
                        INSERT INTO production.styles (style_number, fstyle_number, style_description)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (style_number) DO NOTHING
                        RETURNING style_id
                    """, (style_number, style_number, style_number))
                    result = self.cursor.fetchone()
                    
                    if not result:
                        self.cursor.execute("SELECT style_id FROM production.styles WHERE style_number = %s", (style_number,))
                        result = self.cursor.fetchone()
                    
                    if result:
                        style_id = result[0]
                        
                        # Get yarn_id
                        self.cursor.execute("""
                            INSERT INTO production.yarns (desc_id, yarn_description, is_active)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (desc_id) DO NOTHING
                            RETURNING yarn_id
                        """, (desc_id, desc_id, True))
                        yarn_result = self.cursor.fetchone()
                        
                        if not yarn_result:
                            self.cursor.execute("SELECT yarn_id FROM production.yarns WHERE desc_id = %s", (desc_id,))
                            yarn_result = self.cursor.fetchone()
                        
                        if yarn_result:
                            yarn_id = yarn_result[0]
                            
                            # Insert BOM entry
                            percentage = float(row.get('BOM_Percentage', 0) or 0)
                            self.cursor.execute("""
                                INSERT INTO production.style_bom
                                (style_id, yarn_id, percentage, unit)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (style_id, yarn_id) DO UPDATE
                                SET percentage = EXCLUDED.percentage
                            """, (style_id, yarn_id, percentage, 'PCT'))
                            bom_count += 1
            
            self.conn.commit()
            logger.info(f"✓ Uploaded {bom_count} BOM records")
            return bom_count
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error uploading BOM: {e}")
            raise
    
    def verify_upload(self):
        """Verify all data was uploaded"""
        logger.info("\nVerifying data upload...")
        
        queries = [
            ("Yarns", "SELECT COUNT(*) FROM production.yarns"),
            ("Yarn Inventory", "SELECT COUNT(*) FROM production.yarn_inventory_ts WHERE snapshot_date = %s"),
            ("Sales Orders", "SELECT COUNT(*) FROM production.sales_orders_ts WHERE snapshot_date = %s"),
            ("Knit Orders", "SELECT COUNT(*) FROM production.knit_orders_ts WHERE snapshot_date = %s"),
            ("BOM Entries", "SELECT COUNT(*) FROM production.style_bom"),
            ("Styles", "SELECT COUNT(*) FROM production.styles"),
            ("Customers", "SELECT COUNT(*) FROM production.customers")
        ]
        
        for name, query in queries:
            if "%s" in query:
                self.cursor.execute(query, (self.snapshot_date,))
            else:
                self.cursor.execute(query)
            count = self.cursor.fetchone()[0]
            logger.info(f"  {name}: {count} records")
    
    def run(self):
        """Run full data upload"""
        print("\n" + "=" * 60)
        print("FULL DATA UPLOAD TO POSTGRESQL")
        print("=" * 60)
        
        try:
            self.connect()
            
            # Upload all data types
            yarn_count = self.upload_yarns()
            sales_count = self.upload_sales_orders()
            ko_count = self.upload_knit_orders()
            bom_count = self.upload_bom()
            
            # Verify upload
            self.verify_upload()
            
            print("\n" + "=" * 60)
            print("UPLOAD SUMMARY")
            print("=" * 60)
            print(f"✓ Yarns: {yarn_count} records")
            print(f"✓ Sales Orders: {sales_count} records")
            print(f"✓ Knit Orders: {ko_count} records")
            print(f"✓ BOM: {bom_count} records")
            print("\n✓ All data uploaded successfully!")
            
        except Exception as e:
            print(f"\n✗ Upload failed: {e}")
            raise
        finally:
            self.disconnect()


if __name__ == "__main__":
    uploader = FullDataUploader()
    uploader.run()