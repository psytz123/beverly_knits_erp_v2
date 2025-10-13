#!/usr/bin/env python3
"""
eFab API Data Sync Service
Purpose: Pull data from eFab API and sync to database
Usage: sync_service = EFabAPISync(db_url, session_cookie)
       sync_service.sync_all()
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests
import logging
from decimal import Decimal
import json
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from database.models import (
    init_database, CFVersion, YarnRequirement, YarnInventory,
    ProductionOrder, MachineAssignment, APISync, YarnDemandReport,
    KnitOrder, SalesActivity
)

logger = logging.getLogger(__name__)


class EFabAPISync:
    """
    Service to sync data from eFab API to database.

    Handles authentication, data fetching, and database updates.
    """

    def __init__(self, database_url: str, session_cookie: str):
        """
        Initialize sync service.

        Args:
            database_url: Database connection string
            session_cookie: eFab session cookie for authentication
        """
        self.database_url = database_url
        self.session_cookie = session_cookie
        self.base_url = "https://efab.bkiapps.com"

        # Initialize database
        self.engine, self.SessionLocal = init_database(database_url)

        # API headers
        self.headers = {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": f"dancer.session={session_cookie}",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/json"
        }

        # Track sync session
        self.sync_session: Optional[APISync] = None

    def _start_sync(self, endpoint: str, sync_type: str = "full") -> APISync:
        """Start a new sync session."""
        with self.SessionLocal() as session:
            sync = APISync(
                endpoint=endpoint,
                sync_type=sync_type,
                status="running"
            )
            session.add(sync)
            session.commit()
            session.refresh(sync)
            self.sync_session = sync
            return sync

    def _complete_sync(self, success: bool = True, error_message: str = None) -> None:
        """Complete the sync session."""
        if not self.sync_session:
            return

        with self.SessionLocal() as session:
            sync = session.query(APISync).filter_by(id=self.sync_session.id).first()
            if sync:
                sync.completed_at = datetime.utcnow()
                sync.status = "success" if success else "failed"
                if error_message:
                    sync.error_message = error_message
                session.commit()

    def fetch_cf_versions(self, base_id: int = 1611, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch CF versions from API.

        Args:
            base_id: Starting version ID
            limit: Number of records to fetch

        Returns:
            List of CF version data
        """
        versions = []

        try:
            for i in range(limit):
                version_id = base_id + i
                url = f"{self.base_url}/api/cf_version/from_base/{version_id}"

                response = requests.get(url, headers=self.headers, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if data:
                        versions.append(data)
                        logger.info(f"Fetched CF version {version_id}")
                elif response.status_code == 404:
                    logger.debug(f"CF version {version_id} not found")
                else:
                    logger.warning(f"Error fetching version {version_id}: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching CF versions: {e}")

        return versions

    def sync_cf_versions(self, versions: List[Dict[str, Any]]) -> tuple[int, int]:
        """
        Sync CF versions to database.

        Args:
            versions: List of version data from API

        Returns:
            Tuple of (created_count, updated_count)
        """
        created = 0
        updated = 0

        with self.SessionLocal() as session:
            for version_data in versions:
                try:
                    # Extract key fields
                    version_id = version_data.get('id') or version_data.get('version_id')
                    if not version_id:
                        continue

                    # Check if exists
                    existing = session.query(CFVersion).filter_by(version_id=version_id).first()

                    if existing:
                        # Update existing
                        existing.style_number = version_data.get('style_number')
                        existing.description = version_data.get('description')
                        existing.customer_code = version_data.get('customer_code')
                        existing.fabric_type = version_data.get('fabric_type')
                        existing.construction = version_data.get('construction')
                        existing.width = version_data.get('width')
                        existing.weight = version_data.get('weight')
                        existing.status = version_data.get('status')
                        existing.api_data = version_data
                        existing.updated_at = datetime.utcnow()
                        updated += 1
                    else:
                        # Create new
                        cf_version = CFVersion(
                            version_id=version_id,
                            style_number=version_data.get('style_number'),
                            description=version_data.get('description'),
                            customer_code=version_data.get('customer_code'),
                            fabric_type=version_data.get('fabric_type'),
                            construction=version_data.get('construction'),
                            width=version_data.get('width'),
                            weight=version_data.get('weight'),
                            status=version_data.get('status'),
                            api_data=version_data
                        )
                        session.add(cf_version)
                        created += 1

                    # Sync yarn requirements if present
                    if 'yarns' in version_data:
                        self._sync_yarn_requirements(session, version_id, version_data['yarns'])

                except Exception as e:
                    logger.error(f"Error syncing CF version: {e}")
                    continue

            session.commit()

        logger.info(f"CF Versions sync: {created} created, {updated} updated")
        return created, updated

    def _sync_yarn_requirements(self, session: Session, cf_version_id: int, yarns: List[Dict]) -> None:
        """Sync yarn requirements for a CF version."""
        cf_version = session.query(CFVersion).filter_by(version_id=cf_version_id).first()
        if not cf_version:
            return

        for yarn_data in yarns:
            yarn_code = yarn_data.get('yarn_code')
            if not yarn_code:
                continue

            # Check if exists
            existing = session.query(YarnRequirement).filter_by(
                cf_version_id=cf_version.id,
                yarn_code=yarn_code
            ).first()

            if existing:
                # Update
                existing.yarn_description = yarn_data.get('description')
                existing.supplier = yarn_data.get('supplier')
                existing.color = yarn_data.get('color')
                existing.quantity_required = Decimal(str(yarn_data.get('quantity', 0)))
                existing.unit_of_measure = yarn_data.get('unit')
                existing.cost_per_unit = Decimal(str(yarn_data.get('cost', 0)))
                existing.lead_time_days = yarn_data.get('lead_time')
            else:
                # Create
                yarn_req = YarnRequirement(
                    cf_version_id=cf_version.id,
                    yarn_code=yarn_code,
                    yarn_description=yarn_data.get('description'),
                    supplier=yarn_data.get('supplier'),
                    color=yarn_data.get('color'),
                    quantity_required=Decimal(str(yarn_data.get('quantity', 0))),
                    unit_of_measure=yarn_data.get('unit'),
                    cost_per_unit=Decimal(str(yarn_data.get('cost', 0))),
                    lead_time_days=yarn_data.get('lead_time')
                )
                session.add(yarn_req)

    def fetch_production_orders(self) -> List[Dict[str, Any]]:
        """Fetch production orders from API."""
        orders = []

        try:
            url = f"{self.base_url}/api/production/orders"
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    orders = data
                elif isinstance(data, dict) and 'orders' in data:
                    orders = data['orders']
                logger.info(f"Fetched {len(orders)} production orders")
            else:
                logger.error(f"Error fetching production orders: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching production orders: {e}")

        return orders

    def sync_production_orders(self, orders: List[Dict[str, Any]]) -> tuple[int, int]:
        """Sync production orders to database."""
        created = 0
        updated = 0

        with self.SessionLocal() as session:
            for order_data in orders:
                try:
                    order_number = order_data.get('order_number')
                    if not order_number:
                        continue

                    # Check if exists
                    existing = session.query(ProductionOrder).filter_by(
                        order_number=order_number
                    ).first()

                    # Find related CF version
                    cf_version = None
                    style = order_data.get('style_number')
                    if style:
                        cf_version = session.query(CFVersion).filter_by(
                            style_number=style
                        ).first()

                    if existing:
                        # Update
                        existing.customer_po = order_data.get('customer_po')
                        existing.quantity_ordered = Decimal(str(order_data.get('quantity', 0)))
                        existing.quantity_produced = Decimal(str(order_data.get('produced', 0)))
                        existing.unit_of_measure = order_data.get('unit')
                        existing.due_date = self._parse_date(order_data.get('due_date'))
                        existing.start_date = self._parse_date(order_data.get('start_date'))
                        existing.status = order_data.get('status')
                        existing.priority = order_data.get('priority')
                        existing.work_center = order_data.get('work_center')
                        existing.machine_id = order_data.get('machine_id')
                        if cf_version:
                            existing.cf_version_id = cf_version.id
                        updated += 1
                    else:
                        # Create
                        order = ProductionOrder(
                            order_number=order_number,
                            cf_version_id=cf_version.id if cf_version else None,
                            customer_po=order_data.get('customer_po'),
                            quantity_ordered=Decimal(str(order_data.get('quantity', 0))),
                            quantity_produced=Decimal(str(order_data.get('produced', 0))),
                            unit_of_measure=order_data.get('unit'),
                            due_date=self._parse_date(order_data.get('due_date')),
                            start_date=self._parse_date(order_data.get('start_date')),
                            status=order_data.get('status'),
                            priority=order_data.get('priority'),
                            work_center=order_data.get('work_center'),
                            machine_id=order_data.get('machine_id')
                        )
                        session.add(order)
                        created += 1

                except Exception as e:
                    logger.error(f"Error syncing production order: {e}")
                    continue

            session.commit()

        logger.info(f"Production orders sync: {created} created, {updated} updated")
        return created, updated

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None

        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except:
                    continue
        except:
            return None

        return None

    def sync_all(self) -> Dict[str, Any]:
        """
        Run full sync of all data from eFab API.

        Returns:
            Summary of sync results
        """
        results = {
            'started_at': datetime.utcnow(),
            'success': True,
            'cf_versions': {'created': 0, 'updated': 0},
            'production_orders': {'created': 0, 'updated': 0},
            'errors': []
        }

        try:
            # Start sync session
            self._start_sync("all", "full")

            # Sync CF versions
            logger.info("Starting CF versions sync...")
            versions = self.fetch_cf_versions()
            if versions:
                created, updated = self.sync_cf_versions(versions)
                results['cf_versions']['created'] = created
                results['cf_versions']['updated'] = updated

            # Sync production orders
            logger.info("Starting production orders sync...")
            orders = self.fetch_production_orders()
            if orders:
                created, updated = self.sync_production_orders(orders)
                results['production_orders']['created'] = created
                results['production_orders']['updated'] = updated

            # Complete sync
            self._complete_sync(success=True)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            results['success'] = False
            results['errors'].append(str(e))
            self._complete_sync(success=False, error_message=str(e))

        results['completed_at'] = datetime.utcnow()
        results['duration'] = (results['completed_at'] - results['started_at']).total_seconds()

        return results

    def sync_incremental(self, hours: int = 24) -> Dict[str, Any]:
        """
        Run incremental sync for recent changes.

        Args:
            hours: Sync data from last N hours

        Returns:
            Summary of sync results
        """
        # Similar to sync_all but with date filters
        results = self.sync_all()  # For now, do full sync
        return results


def main():
    """Test the sync service."""
    import os

    # Configuration
    DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/efab_erp")
    SESSION_COOKIE = os.environ.get("EFAB_SESSION", "aNEM2YqIXevF7IvZ13r68JFeSlsVo1Lh")

    # Initialize sync service
    sync_service = EFabAPISync(DATABASE_URL, SESSION_COOKIE)

    # Run full sync
    print("Starting full sync...")
    results = sync_service.sync_all()

    print("\nSync Results:")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results['duration']:.2f} seconds")
    print(f"  CF Versions: {results['cf_versions']['created']} created, {results['cf_versions']['updated']} updated")
    print(f"  Production Orders: {results['production_orders']['created']} created, {results['production_orders']['updated']} updated")

    if results['errors']:
        print(f"  Errors: {', '.join(results['errors'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()