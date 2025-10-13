#!/usr/bin/env python3
"""
Database Models for eFab API Data Storage
Purpose: SQLAlchemy models for storing eFab API data
Usage: Import models for database operations
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, Text, ForeignKey, Index, JSON, DECIMAL, Date,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class CFVersion(Base):
    """Core fabric/style version data from eFab API."""
    __tablename__ = 'cf_versions'

    id = Column(Integer, primary_key=True)
    version_id = Column(Integer, unique=True, index=True)
    style_number = Column(String(50), index=True)
    description = Column(Text)
    customer_code = Column(String(50))
    fabric_type = Column(String(100))
    construction = Column(String(50))
    width = Column(Float)
    weight = Column(Float)
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    api_data = Column(JSON)  # Store raw API response

    # Relationships
    yarn_requirements = relationship("YarnRequirement", back_populates="cf_version")
    production_orders = relationship("ProductionOrder", back_populates="cf_version")

    __table_args__ = (
        Index('idx_cf_style_customer', 'style_number', 'customer_code'),
    )


class YarnRequirement(Base):
    """Yarn requirements for each CF version."""
    __tablename__ = 'yarn_requirements'

    id = Column(Integer, primary_key=True)
    cf_version_id = Column(Integer, ForeignKey('cf_versions.id'))
    yarn_code = Column(String(50), index=True)
    yarn_description = Column(Text)
    supplier = Column(String(100))
    color = Column(String(50))
    quantity_required = Column(DECIMAL(15, 3))
    unit_of_measure = Column(String(10))
    cost_per_unit = Column(DECIMAL(10, 4))
    lead_time_days = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cf_version = relationship("CFVersion", back_populates="yarn_requirements")
    inventory_levels = relationship("YarnInventory", back_populates="yarn")

    __table_args__ = (
        UniqueConstraint('cf_version_id', 'yarn_code', name='uq_cf_yarn'),
        Index('idx_yarn_supplier', 'yarn_code', 'supplier'),
    )


class YarnInventory(Base):
    """Current yarn inventory levels."""
    __tablename__ = 'yarn_inventory'

    id = Column(Integer, primary_key=True)
    yarn_requirement_id = Column(Integer, ForeignKey('yarn_requirements.id'))
    yarn_code = Column(String(50), index=True)
    location = Column(String(50))
    quantity_on_hand = Column(DECIMAL(15, 3))
    quantity_allocated = Column(DECIMAL(15, 3))
    quantity_available = Column(DECIMAL(15, 3))
    reorder_point = Column(DECIMAL(15, 3))
    reorder_quantity = Column(DECIMAL(15, 3))
    last_received_date = Column(Date)
    last_counted_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    yarn = relationship("YarnRequirement", back_populates="inventory_levels")

    __table_args__ = (
        UniqueConstraint('yarn_code', 'location', name='uq_yarn_location'),
        CheckConstraint('quantity_on_hand >= 0', name='check_positive_quantity'),
    )


class ProductionOrder(Base):
    """Production orders from eFab."""
    __tablename__ = 'production_orders'

    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, index=True)
    cf_version_id = Column(Integer, ForeignKey('cf_versions.id'))
    customer_po = Column(String(50))
    quantity_ordered = Column(DECIMAL(15, 3))
    quantity_produced = Column(DECIMAL(15, 3), default=0)
    unit_of_measure = Column(String(10))
    due_date = Column(Date)
    start_date = Column(Date)
    status = Column(String(20))
    priority = Column(Integer)
    work_center = Column(String(50))
    machine_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cf_version = relationship("CFVersion", back_populates="production_orders")
    machine_assignments = relationship("MachineAssignment", back_populates="order")

    __table_args__ = (
        Index('idx_order_status_due', 'status', 'due_date'),
        Index('idx_order_machine', 'machine_id', 'work_center'),
    )


class MachineAssignment(Base):
    """Machine assignments for production orders."""
    __tablename__ = 'machine_assignments'

    id = Column(Integer, primary_key=True)
    production_order_id = Column(Integer, ForeignKey('production_orders.id'))
    machine_id = Column(String(50), index=True)
    work_center = Column(String(50))
    assigned_date = Column(DateTime)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    status = Column(String(20))
    operator_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    order = relationship("ProductionOrder", back_populates="machine_assignments")

    __table_args__ = (
        Index('idx_machine_schedule', 'machine_id', 'start_time', 'end_time'),
    )


class APISync(Base):
    """Track API synchronization history."""
    __tablename__ = 'api_sync'

    id = Column(Integer, primary_key=True)
    sync_id = Column(String(50), default=lambda: str(uuid.uuid4()), unique=True)
    endpoint = Column(String(200))
    sync_type = Column(String(20))  # 'full', 'incremental', 'manual'
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20))  # 'running', 'success', 'failed'
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_sync_status_time', 'status', 'started_at'),
    )


class YarnDemandReport(Base):
    """Store parsed Yarn Demand report data."""
    __tablename__ = 'yarn_demand_reports'

    id = Column(Integer, primary_key=True)
    report_date = Column(Date, index=True)
    yarn_code = Column(String(50), index=True)
    yarn_description = Column(Text)
    current_inventory = Column(DECIMAL(15, 3))
    allocated_quantity = Column(DECIMAL(15, 3))
    on_order_quantity = Column(DECIMAL(15, 3))
    planned_usage = Column(DECIMAL(15, 3))
    projected_balance = Column(DECIMAL(15, 3))
    reorder_suggestion = Column(DECIMAL(15, 3))
    lead_time_days = Column(Integer)
    supplier = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('report_date', 'yarn_code', name='uq_report_yarn'),
        Index('idx_demand_balance', 'projected_balance', 'yarn_code'),
    )


class KnitOrder(Base):
    """Knit orders from eFab."""
    __tablename__ = 'knit_orders'

    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, index=True)
    style_number = Column(String(50), index=True)
    customer_code = Column(String(50))
    po_number = Column(String(50))
    quantity = Column(DECIMAL(15, 3))
    unit = Column(String(10))
    due_date = Column(Date)
    status = Column(String(20))
    work_center = Column(String(50))
    machine_assigned = Column(String(50))
    fabric_type = Column(String(100))
    weight_lbs = Column(DECIMAL(15, 3))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_knit_status_due', 'status', 'due_date'),
        Index('idx_knit_style_customer', 'style_number', 'customer_code'),
    )


class SalesActivity(Base):
    """Historical sales data for forecasting."""
    __tablename__ = 'sales_activity'

    id = Column(Integer, primary_key=True)
    invoice_number = Column(String(50), index=True)
    invoice_date = Column(Date, index=True)
    customer_code = Column(String(50), index=True)
    style_number = Column(String(50), index=True)
    quantity_sold = Column(DECIMAL(15, 3))
    unit_price = Column(DECIMAL(10, 4))
    total_amount = Column(DECIMAL(15, 2))
    currency = Column(String(3), default='USD')
    sales_rep = Column(String(100))
    region = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_sales_date_customer', 'invoice_date', 'customer_code'),
        Index('idx_sales_style_date', 'style_number', 'invoice_date'),
    )


def init_database(database_url: str) -> tuple:
    """
    Initialize database and create tables.

    Args:
        database_url: PostgreSQL connection string

    Returns:
        Tuple of (engine, SessionLocal)
    """
    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal


def drop_all_tables(engine) -> None:
    """Drop all tables for fresh start."""
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    # Test database connection
    from sqlalchemy import text

    # Use PostgreSQL or SQLite for testing
    DATABASE_URL = "postgresql://user:password@localhost/efab_erp"
    # DATABASE_URL = "sqlite:///efab_erp.db"  # Alternative for testing

    try:
        engine, SessionLocal = init_database(DATABASE_URL)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")

        print("✓ All tables created successfully")

    except Exception as e:
        print(f"✗ Database initialization failed: {e}")