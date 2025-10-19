"""
Fabric Inventory Sync from eFab
=================================

This script syncs fabric inventory data from eFab API endpoints to the Turso database.

eFab Stages:
- G00: Raw greige off loom
- G02: Dyed/finished greige
- I01: Awaiting QC inspection
- F01: QC passed, ready to ship

Usage:
    python scripts/sync_fabric_inventory_from_efab.py

    # Or with options:
    python scripts/sync_fabric_inventory_from_efab.py --dry-run
    python scripts/sync_fabric_inventory_from_efab.py --stage G00
"""

import os
import sys
import httpx
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.database.turso_client import TursoClient


class EFabInventorySync:
    """Syncs fabric inventory from eFab to Turso database."""

    def __init__(self, efab_base_url: str = "http://localhost:5006", dry_run: bool = False):
        """Initialize sync manager.

        Args:
            efab_base_url: Base URL for eFab API
            dry_run: If True, don't write to database
        """
        self.efab_base_url = efab_base_url.rstrip('/')
        self.dry_run = dry_run
        self.db = TursoClient()
        self.stats = {
            'total_fetched': 0,
            'inserted': 0,
            'updated': 0,
            'errors': 0
        }

    async def fetch_stage_data(self, stage: str) -> List[Dict]:
        """Fetch inventory data from eFab for a specific stage.

        Args:
            stage: One of 'G00', 'G02', 'I01', 'F01'

        Returns:
            List of inventory records from eFab
        """
        # Map stages to eFab endpoints
        endpoint_map = {
            'G00': '/api/greige/g00',
            'G02': '/api/greige/g02',
            'I01': '/api/finished/i01',
            'F01': '/api/finished/f01'
        }

        endpoint = endpoint_map.get(stage)
        if not endpoint:
            raise ValueError(f"Invalid stage: {stage}")

        url = f"{self.efab_base_url}{endpoint}"

        print(f"[FETCH] Fetching {stage} data from {url}...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                # Handle different response formats
                if isinstance(data, dict):
                    # Check for common data keys
                    records = data.get('data', data.get('items', data.get('inventory', [])))
                elif isinstance(data, list):
                    records = data
                else:
                    records = []

                print(f"[OK] Fetched {len(records)} records from {stage}")
                return records

            except httpx.HTTPError as e:
                print(f"[ERROR] HTTP error fetching {stage}: {e}")
                return []
            except Exception as e:
                print(f"[ERROR] Error fetching {stage}: {e}")
                return []

    def transform_record(self, record: Dict, stage: str) -> Optional[Dict]:
        """Transform eFab record to fabric_inventory format.

        Args:
            record: Raw record from eFab
            stage: Stage identifier (G00, G02, I01, F01)

        Returns:
            Transformed record ready for database insertion
        """
        try:
            # Extract fabric ID from nested structure
            # Path: knit_version.knit_style_base.base_style
            fabric_id = None
            if 'knit_version' in record and record['knit_version']:
                knit_version = record['knit_version']
                if 'knit_style_base' in knit_version and knit_version['knit_style_base']:
                    fabric_id = knit_version['knit_style_base'].get('base_style')

            # Fallback to other possible locations
            if not fabric_id:
                fabric_id = record.get('style_number') or record.get('style') or record.get('base_style')

            if not fabric_id:
                return None

            # Clean fabric ID (remove non-numeric characters, keep first 4 digits)
            fabric_id_clean = ''.join(filter(str.isdigit, str(fabric_id)))[:4]
            if not fabric_id_clean:
                return None

            # Determine fabric type based on stage
            # G00/G02 are greige, I01/F01 are finished
            fabric_type = 'greige' if stage in ['G00', 'G02'] else 'finished'

            # Extract quantities from eFab fields
            quantity_yards = float(record.get('qty_yds', 0) or 0)
            quantity_lbs = float(record.get('qty_lbs', 0) or 0)

            # Single roll per record in eFab
            rolls = 1 if quantity_yards > 0 or quantity_lbs > 0 else 0

            # Additional fields
            location = record.get('location', '') or record.get('inspection_machine', '')
            lot_number = record.get('lot_number', '') or record.get('roll_number', '')
            grade = record.get('grade', '')
            color = record.get('color', '')

            # Document reference
            document = record.get('document', '')

            return {
                'fabric_id': fabric_id_clean,
                'fabric_type': fabric_type,
                'stage': stage,
                'quantity_yards': quantity_yards,
                'quantity_lbs': quantity_lbs,
                'rolls': rolls,
                'location': location or None,
                'lot_number': lot_number or None,
                'grade': grade or None,
                'notes': f"Doc: {document}, Color: {color}" if document or color else None,
                'updated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f"[WARN] Error transforming record: {e}")
            return None

    def upsert_inventory(self, record: Dict) -> bool:
        """Insert or update fabric inventory record.

        Args:
            record: Transformed record

        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            print(f"  [DRY RUN] Would upsert: {record['fabric_id']} @ {record['stage']} = {record['quantity_yards']} yards")
            self.stats['inserted'] += 1
            return True

        try:
            # Check if record exists
            existing = self.db.execute(
                """
                SELECT id FROM fabric_inventory
                WHERE fabric_id = ? AND stage = ?
                """,
                [record['fabric_id'], record['stage']]
            )

            if existing:
                # Update existing record
                self.db.execute(
                    """
                    UPDATE fabric_inventory
                    SET quantity_yards = ?,
                        quantity_lbs = ?,
                        rolls = ?,
                        location = ?,
                        lot_number = ?,
                        grade = ?,
                        notes = ?,
                        updated_at = ?
                    WHERE fabric_id = ? AND stage = ?
                    """,
                    [
                        record['quantity_yards'],
                        record['quantity_lbs'],
                        record['rolls'],
                        record['location'],
                        record['lot_number'],
                        record['grade'],
                        record['notes'],
                        record['updated_at'],
                        record['fabric_id'],
                        record['stage']
                    ]
                )
                self.stats['updated'] += 1
                print(f"  [UPDATE] Updated: {record['fabric_id']} @ {record['stage']}")
            else:
                # Insert new record
                self.db.execute(
                    """
                    INSERT INTO fabric_inventory (
                        fabric_id, fabric_type, stage,
                        quantity_yards, quantity_lbs, rolls,
                        location, lot_number, grade, notes,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record['fabric_id'],
                        record['fabric_type'],
                        record['stage'],
                        record['quantity_yards'],
                        record['quantity_lbs'],
                        record['rolls'],
                        record['location'],
                        record['lot_number'],
                        record['grade'],
                        record['notes'],
                        record['updated_at'],
                        record['updated_at']
                    ]
                )
                self.stats['inserted'] += 1
                print(f"  [INSERT] Inserted: {record['fabric_id']} @ {record['stage']}")

            return True

        except Exception as e:
            print(f"  [ERROR] Error upserting record: {e}")
            self.stats['errors'] += 1
            return False

    async def sync_stage(self, stage: str):
        """Sync all inventory for a specific stage.

        Args:
            stage: Stage to sync (G00, G02, I01, F01)
        """
        print(f"\n{'='*60}")
        print(f"[SYNC] Syncing Stage: {stage}")
        print(f"{'='*60}")

        # Fetch data from eFab
        records = await self.fetch_stage_data(stage)
        self.stats['total_fetched'] += len(records)

        if not records:
            print(f"[WARN] No records fetched for {stage}")
            return

        # Transform and upsert each record
        print(f"\n[PROCESS] Processing {len(records)} records...")
        for i, raw_record in enumerate(records, 1):
            transformed = self.transform_record(raw_record, stage)
            if transformed:
                self.upsert_inventory(transformed)

            # Progress indicator every 10 records
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(records)}")

        print(f"[OK] Completed {stage}")

    async def sync_all_stages(self, stages: Optional[List[str]] = None):
        """Sync inventory from all eFab stages.

        Args:
            stages: List of stages to sync, or None for all
        """
        if stages is None:
            stages = ['G00', 'G02', 'I01', 'F01']

        print("=" * 60)
        print("         eFab Inventory Sync")
        print("=" * 60)
        print(f"\n[*] Target: {self.efab_base_url}")
        print(f"[*] Stages: {', '.join(stages)}")
        print(f"[*] Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")

        # Sync each stage
        for stage in stages:
            await self.sync_stage(stage)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print sync summary statistics."""
        print("\n" + "="*60)
        print("[STATS] SYNC SUMMARY")
        print("="*60)
        print(f"  Total Fetched:  {self.stats['total_fetched']:>6} records")
        print(f"  Inserted:       {self.stats['inserted']:>6} records")
        print(f"  Updated:        {self.stats['updated']:>6} records")
        print(f"  Errors:         {self.stats['errors']:>6} errors")
        print("="*60)

        if self.dry_run:
            print("\n[WARN] DRY RUN MODE - No changes were made to the database")
        else:
            print("\n[OK] Sync completed successfully!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sync fabric inventory from eFab to Turso database'
    )
    parser.add_argument(
        '--efab-url',
        default='http://localhost:5006',
        help='eFab API base URL (default: http://localhost:5006)'
    )
    parser.add_argument(
        '--stage',
        choices=['G00', 'G02', 'I01', 'F01'],
        help='Sync only a specific stage (default: all stages)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making database changes'
    )

    args = parser.parse_args()

    # Initialize sync manager
    sync = EFabInventorySync(
        efab_base_url=args.efab_url,
        dry_run=args.dry_run
    )

    # Sync stages
    stages = [args.stage] if args.stage else None
    await sync.sync_all_stages(stages)


if __name__ == '__main__':
    asyncio.run(main())
