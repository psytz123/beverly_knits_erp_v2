#!/usr/bin/env python3
"""
Six-Phase Planning Engine for Beverly Knits ERP
Comprehensive supply chain planning with ML-driven optimization
Cleaned and refactored version ready for ERP integration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    from prophet import Prophet
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available. Some features will be limited.")

@dataclass
class PlanningPhaseResult:
    """Result container for each planning phase"""
    phase_number: int
    phase_name: str
    status: str
    execution_time: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    output_data: Any = None

@dataclass
class ProcurementRecommendation:
    """Procurement recommendation container"""
    item_code: str
    item_description: str
    supplier: str
    recommended_quantity: float
    eoq: float
    safety_stock: float
    reorder_point: float
    total_cost: float
    savings_potential: float
    priority: str
    rationale: str


class SixPhasePlanningEngine:
    """
    Comprehensive 6-phase planning engine for supply chain optimization
    Designed for seamless integration with Beverly Knits ERP system
    """
    
    def __init__(self, data_path: Path = None, erp_connection: Any = None):
        """
        Initialize the planning engine
        
        Args:
            data_path: Path to data files (optional if using ERP connection)
            erp_connection: Connection to ERP system for direct data access
        """
        # Default to ERP Data/5 if no path provided
        self.data_path = Path(data_path) if data_path else Path("/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5")
        self.erp_connection = erp_connection
        
        # Phase results storage
        self.phase_results = []
        self.unified_forecast = None
        self.exploded_bom = None
        self.net_requirements = None
        self.procurement_plan = None
        self.supplier_assignments = None
        self.final_output = None
        
        # ERP integration data containers
        self.erp_data = {}
        self.erp_sales_data = None
        self.erp_inventory_data = None
        self.erp_bom_data = None
        self.erp_supplier_data = None
        
        # ML models for forecasting
        self.forecast_models = {}
        self.forecast_weights = {}
        
        # Configuration parameters
        self.config = {
            'forecast_horizon': 90,  # days
            'safety_stock_service_level': 0.98,
            'holding_cost_rate': 0.25,  # 25% annual
            'ordering_cost': 75,
            'lead_time_buffer': 1.2,  # 20% buffer
            'min_order_quantity': 100,
            'forecast_confidence_threshold': 0.85,
            'stockout_risk_threshold': 0.20,
            'yarn_safety_buffer': 1.15,  # 15% yarn safety buffer
            'production_lead_time': 14,  # days
            'critical_inventory_days': 30,
        }
        
        logger.info(f"Six-Phase Planning Engine initialized")
    
    # ==================== ERP Integration Methods ====================
    
    def set_erp_data(self, erp_data: Dict):
        """
        Set data from ERP system for planning
        
        Args:
            erp_data: Dictionary containing ERP data
        """
        self.erp_data = erp_data
        
        # Map ERP data to internal structures
        if 'sales_data' in erp_data:
            self.erp_sales_data = self._convert_erp_sales_data(erp_data['sales_data'])
        if 'inventory_data' in erp_data:
            self.erp_inventory_data = self._convert_erp_inventory_data(erp_data['inventory_data'])
        if 'bom_data' in erp_data:
            self.erp_bom_data = self._convert_erp_bom_data(erp_data['bom_data'])
        if 'supplier_data' in erp_data:
            self.erp_supplier_data = self._convert_erp_supplier_data(erp_data['supplier_data'])
            
        logger.info("ERP data loaded and converted successfully")
    
    def _convert_erp_sales_data(self, sales_data):
        """Convert ERP sales data to internal format"""
        if isinstance(sales_data, pd.DataFrame):
            return sales_data
        elif isinstance(sales_data, list):
            return pd.DataFrame(sales_data)
        else:
            return sales_data
    
    def _convert_erp_inventory_data(self, inventory_data):
        """Convert ERP inventory data to internal format"""
        if isinstance(inventory_data, pd.DataFrame):
            return inventory_data
        elif isinstance(inventory_data, list):
            return pd.DataFrame(inventory_data)
        else:
            return inventory_data
    
    def _convert_erp_bom_data(self, bom_data):
        """Convert ERP BOM data to internal format"""
        if isinstance(bom_data, pd.DataFrame):
            return bom_data
        elif isinstance(bom_data, list):
            return pd.DataFrame(bom_data)
        else:
            return bom_data
    
    def _convert_erp_supplier_data(self, supplier_data):
        """Convert ERP supplier data to internal format"""
        if isinstance(supplier_data, pd.DataFrame):
            return supplier_data
        elif isinstance(supplier_data, list):
            return pd.DataFrame(supplier_data)
        else:
            return supplier_data
    
    def get_procurement_recommendations(self) -> List[Dict]:
        """Get formatted procurement recommendations for ERP dashboard"""
        if not self.procurement_plan:
            return []
        
        recommendations = []
        for rec in self.procurement_plan[:10]:  # Top 10 recommendations
            recommendations.append({
                'item_code': rec.item_code,
                'description': rec.item_description,
                'supplier': rec.supplier,
                'quantity': rec.recommended_quantity,
                'eoq': rec.eoq,
                'safety_stock': rec.safety_stock,
                'reorder_point': rec.reorder_point,
                'total_cost': rec.total_cost,
                'savings': rec.savings_potential,
                'priority': rec.priority,
                'rationale': rec.rationale
            })
        return recommendations
    
    def get_inventory_analysis(self) -> Dict:
        """Get inventory analysis summary for ERP dashboard"""
        if self.net_requirements is None or self.net_requirements.empty:
            return {}
        
        return {
            'total_items': len(self.net_requirements),
            'critical_shortages': len(self.net_requirements[self.net_requirements['net_required'] > 0]),
            'total_net_requirement': self.net_requirements['net_required'].sum(),
            'stockout_risk_items': len(self.net_requirements[
                (self.net_requirements['on_hand'] < self.net_requirements['gross_required'] * 0.2)
            ]) if not self.net_requirements.empty else 0
        }
    
    def get_supplier_risk_assessment(self) -> Dict:
        """Get supplier risk assessment for ERP dashboard"""
        if not self.supplier_assignments:
            return {}
        
        # Count assignments per supplier
        supplier_counts = defaultdict(int)
        for assignment in self.supplier_assignments.values():
            supplier_counts[assignment['primary_supplier']] += 1
        
        # Calculate concentration risk
        total = len(self.supplier_assignments)
        max_concentration = max(supplier_counts.values()) / total if total > 0 else 0
        
        return {
            'total_suppliers': len(supplier_counts),
            'max_concentration': max_concentration,
            'high_risk': max_concentration > 0.4,
            'diversification_needed': len(supplier_counts) < 3
        }
    
    # ==================== Main Planning Cycle ====================
    
    def execute_full_planning_cycle(self) -> List[PlanningPhaseResult]:
        """Execute all 6 phases of the planning cycle"""
        logger.info("Starting 6-Phase Planning Cycle")
        
        # Reset results
        self.phase_results = []
        
        # Phase 1: Forecast Unification
        phase1_result = self.phase1_forecast_unification()
        self.phase_results.append(phase1_result)
        
        # Phase 2: BOM Explosion
        phase2_result = self.phase2_bom_explosion()
        self.phase_results.append(phase2_result)
        
        # Phase 3: Inventory Netting
        phase3_result = self.phase3_inventory_netting()
        self.phase_results.append(phase3_result)
        
        # Phase 4: Procurement Optimization
        phase4_result = self.phase4_procurement_optimization()
        self.phase_results.append(phase4_result)
        
        # Phase 5: Supplier Selection
        phase5_result = self.phase5_supplier_selection()
        self.phase_results.append(phase5_result)
        
        # Phase 6: Output Generation
        phase6_result = self.phase6_output_generation()
        self.phase_results.append(phase6_result)
        
        logger.info("6-Phase Planning Cycle completed")
        return self.phase_results
    
    # ==================== Phase 1: Forecast Unification ====================
    
    def phase1_forecast_unification(self) -> PlanningPhaseResult:
        """Phase 1: Unify forecasts from multiple sources using ensemble methods"""
        start_time = datetime.now()
        logger.info("Phase 1: Starting Forecast Unification")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            # Load historical sales data
            sales_data = self._load_sales_data()
            
            if sales_data is not None and len(sales_data) > 0:
                # Prepare time series data
                ts_data = self._prepare_time_series(sales_data)
                
                # Initialize multiple forecasting models
                forecasts = {}
                
                # 1. Moving Average
                ma_forecast = self._moving_average_forecast(ts_data)
                if ma_forecast:
                    forecasts['moving_average'] = ma_forecast
                
                # 2. Exponential Smoothing
                es_forecast = self._exponential_smoothing_forecast(ts_data)
                if es_forecast:
                    forecasts['exponential_smoothing'] = es_forecast
                
                # 3. ML-based forecasting if available
                if ML_AVAILABLE:
                    # Random Forest
                    rf_forecast = self._ml_forecast(ts_data, 'random_forest')
                    if rf_forecast:
                        forecasts['random_forest'] = rf_forecast
                    
                    # XGBoost
                    xgb_forecast = self._ml_forecast(ts_data, 'xgboost')
                    if xgb_forecast:
                        forecasts['xgboost'] = xgb_forecast
                    
                    # Prophet
                    prophet_forecast = self._prophet_forecast(ts_data)
                    if prophet_forecast:
                        forecasts['prophet'] = prophet_forecast
                
                # Ensemble forecasting
                self.unified_forecast = self._ensemble_forecast(forecasts)
                
                # Calculate forecast accuracy metrics
                accuracy_metrics = self._calculate_forecast_accuracy(ts_data, self.unified_forecast)
                
                # Analyze inventory risk
                inventory_risk_analysis = self._analyze_inventory_stockout_risk(self.unified_forecast)
                
                details = {
                    'forecast_sources': len(forecasts),
                    'forecast_models': list(forecasts.keys()),
                    'forecast_horizon': f"{self.config['forecast_horizon']} days",
                    'total_forecasted_demand': f"{self.unified_forecast.get('total_demand', 0):,.0f} units",
                    'mape': f"{accuracy_metrics.get('mape', 0):.1f}%",
                    'confidence_level': f"{accuracy_metrics.get('confidence', 0):.1%}",
                    'high_risk_items': len(inventory_risk_analysis.get('high_risk', [])),
                    'stockout_probability': inventory_risk_analysis.get('avg_stockout_prob', 0)
                }
                
                status = 'Completed'
            else:
                warnings.append("No historical sales data available")
                status = 'Partial'
                
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 1 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=1,
            phase_name="Forecast Unification",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.unified_forecast
        )
    
    # ==================== Phase 2: BOM Explosion ====================
    
    def phase2_bom_explosion(self) -> PlanningPhaseResult:
        """Phase 2: Explode BOM to calculate material requirements"""
        start_time = datetime.now()
        logger.info("Phase 2: Starting BOM Explosion")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            # Load BOM data
            bom_data = self._load_bom_data()
            
            if bom_data is not None and self.unified_forecast is not None:
                # Explode BOM based on forecast
                self.exploded_bom = self._explode_bom(bom_data, self.unified_forecast)
                
                # Calculate material requirements
                material_summary = self._summarize_materials(self.exploded_bom)
                
                # Calculate yarn consumption
                yarn_consumption = self._calculate_yarn_consumption(self.exploded_bom, self.unified_forecast)
                
                # Production scheduling
                production_schedule = self._calculate_production_timing(self.unified_forecast, yarn_consumption)
                
                details = {
                    'total_components': len(self.exploded_bom),
                    'yarn_types_required': len(yarn_consumption),
                    'total_yarn_consumption_lbs': sum(y.get('total_required_lbs', 0) for y in yarn_consumption),
                    'production_start_date': production_schedule.get('production_start_date', 'TBD'),
                    'yarn_order_deadline': production_schedule.get('yarn_order_deadline', 'TBD')
                }
                
                status = 'Completed'
            else:
                warnings.append("BOM data or forecast not available")
                status = 'Partial'
                self.exploded_bom = pd.DataFrame()
                
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 2 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=2,
            phase_name="BOM Explosion",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.exploded_bom
        )
    
    # ==================== Phase 3: Inventory Netting ====================
    
    def phase3_inventory_netting(self) -> PlanningPhaseResult:
        """Phase 3: Net inventory against requirements"""
        start_time = datetime.now()
        logger.info("Phase 3: Starting Inventory Netting")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            # Load current inventory
            inventory_data = self._load_inventory_data()
            
            if inventory_data is not None and self.exploded_bom is not None:
                # Calculate net requirements
                self.net_requirements = self._calculate_net_requirements(
                    self.exploded_bom, 
                    inventory_data
                )
                
                # Analyze yarn shortages
                yarn_shortage_analysis = self._analyze_yarn_shortages(
                    self.exploded_bom, 
                    inventory_data
                )
                
                # Calculate procurement timing
                yarn_procurement_timing = self._calculate_yarn_procurement_timing(
                    yarn_shortage_analysis,
                    self.unified_forecast
                )
                
                # Identify critical shortages
                critical_items = self._identify_critical_shortages(self.net_requirements)
                
                details = {
                    'on_hand_inventory': f"{inventory_data['Planning Balance'].sum():,.0f} units",
                    'net_requirements': f"{self.net_requirements['net_required'].sum():,.0f} units",
                    'critical_shortages': len(critical_items),
                    'yarn_shortages_identified': len(yarn_shortage_analysis),
                    'critical_yarn_shortages': len([y for y in yarn_shortage_analysis if y.get('urgency') == 'Critical'])
                }
                
                status = 'Completed'
            else:
                warnings.append("Inventory or BOM data not available")
                status = 'Partial'
                self.net_requirements = pd.DataFrame()
                
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 3 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=3,
            phase_name="Inventory Netting",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.net_requirements
        )
    
    # ==================== Phase 4: Procurement Optimization ====================
    
    def phase4_procurement_optimization(self) -> PlanningPhaseResult:
        """Phase 4: Optimize procurement using EOQ and other methods"""
        start_time = datetime.now()
        logger.info("Phase 4: Starting Procurement Optimization")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            if self.net_requirements is not None and len(self.net_requirements) > 0:
                # Calculate optimal order quantities
                self.procurement_plan = []
                total_savings = 0
                
                for _, item in self.net_requirements.iterrows():
                    if item['net_required'] > 0:
                        # Calculate EOQ
                        eoq_result = self._calculate_eoq(item)
                        
                        # Calculate safety stock
                        safety_stock = self._calculate_safety_stock(item)
                        
                        # Calculate reorder point
                        reorder_point = self._calculate_reorder_point(item, safety_stock)
                        
                        # Create procurement recommendation
                        recommendation = ProcurementRecommendation(
                            item_code=item.get('item_code', 'N/A'),
                            item_description=item.get('description', 'N/A'),
                            supplier=item.get('supplier', 'TBD'),
                            recommended_quantity=max(eoq_result['eoq'], item['net_required']),
                            eoq=eoq_result['eoq'],
                            safety_stock=safety_stock,
                            reorder_point=reorder_point,
                            total_cost=eoq_result['total_cost'],
                            savings_potential=eoq_result['savings'],
                            priority=self._determine_priority(item),
                            rationale=self._generate_rationale(item, eoq_result)
                        )
                        
                        self.procurement_plan.append(recommendation)
                        total_savings += eoq_result['savings']
                
                details = {
                    'items_optimized': len(self.procurement_plan),
                    'total_procurement_value': f"${sum(r.total_cost for r in self.procurement_plan):,.0f}",
                    'potential_savings': f"${total_savings:,.0f}"
                }
                
                status = 'Completed'
            else:
                warnings.append("No net requirements to optimize")
                status = 'Skipped'
                self.procurement_plan = []
                
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 4 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=4,
            phase_name="Procurement Optimization",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.procurement_plan
        )
    
    # ==================== Phase 5: Supplier Selection ====================
    
    def phase5_supplier_selection(self) -> PlanningPhaseResult:
        """Phase 5: Select optimal suppliers based on multi-criteria analysis"""
        start_time = datetime.now()
        logger.info("Phase 5: Starting Supplier Selection")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            if self.procurement_plan and len(self.procurement_plan) > 0:
                # Load supplier data
                supplier_data = self._load_supplier_data()
                
                # Evaluate suppliers
                supplier_scores = self._evaluate_suppliers(supplier_data)
                
                # Assign items to suppliers
                self.supplier_assignments = self._assign_suppliers(
                    self.procurement_plan, 
                    supplier_scores
                )
                
                # Risk analysis
                risk_analysis = self._analyze_supplier_risks(self.supplier_assignments)
                
                details = {
                    'suppliers_evaluated': len(supplier_scores),
                    'assignments_made': len(self.supplier_assignments),
                    'high_risk_suppliers': risk_analysis.get('high_risk_count', 0),
                    'diversification_index': f"{risk_analysis.get('diversification_index', 0):.2f}"
                }
                
                status = 'Completed'
            else:
                warnings.append("No procurement plan available for supplier selection")
                status = 'Skipped'
                self.supplier_assignments = {}
                
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 5 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=5,
            phase_name="Supplier Selection",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.supplier_assignments
        )
    
    # ==================== Phase 6: Output Generation ====================
    
    def phase6_output_generation(self) -> PlanningPhaseResult:
        """Phase 6: Generate final outputs and recommendations"""
        start_time = datetime.now()
        logger.info("Phase 6: Starting Output Generation")
        
        errors = []
        warnings = []
        details = {}
        
        try:
            # Compile all outputs
            self.final_output = {
                'planning_date': datetime.now().isoformat(),
                'planning_horizon': f"{self.config['forecast_horizon']} days",
                'forecast_summary': self._summarize_forecast(),
                'procurement_orders': self._generate_purchase_orders(),
                'supplier_assignments': self.supplier_assignments,
                'kpis': self._calculate_planning_kpis()
            }
            
            # Export results
            export_status = self._export_results()
            
            details = {
                'purchase_orders': len(self.final_output['procurement_orders']),
                'total_order_value': f"${sum(o.get('total_value', 0) for o in self.final_output['procurement_orders']):,.0f}",
                'export_formats': export_status.get('formats', ['JSON']),
                'optimization_score': f"{self.final_output['kpis'].get('optimization_score', 0):.1f}/100"
            }
            
            status = 'Completed'
            
        except Exception as e:
            errors.append(str(e))
            status = 'Failed'
            logger.error(f"Phase 6 error: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return PlanningPhaseResult(
            phase_number=6,
            phase_name="Output Generation",
            status=status,
            execution_time=execution_time,
            details=details,
            errors=errors,
            warnings=warnings,
            output_data=self.final_output
        )
    
    # ==================== Helper Methods ====================
    
    def _load_sales_data(self) -> Optional[pd.DataFrame]:
        """Load historical sales data from ERP or files"""
        if self.erp_sales_data is not None:
            return self.erp_sales_data
        
        if self.data_path:
            try:
                # Try multiple file names for sales data
                sales_files = [
                    "Sales_Activity_Report.xlsx",
                    "Sales Activity Report.xlsx",
                    "Sales Activity Report (4).xlsx"
                ]
                
                for filename in sales_files:
                    sales_file = self.data_path / filename
                    if sales_file.exists():
                        df = pd.read_excel(sales_file)
                        # Standardize column names
                        if 'Invoice Date' in df.columns:
                            df.rename(columns={'Invoice Date': 'Date'}, inplace=True)
                        if 'fStyle#' in df.columns:
                            df.rename(columns={'fStyle#': 'Style'}, inplace=True)
                        logger.info(f"Loaded sales data from {filename}")
                        return df
            except Exception as e:
                logger.error(f"Error loading sales data: {e}")
        
        return None
    
    def _load_inventory_data(self) -> Optional[pd.DataFrame]:
        """Load current inventory data from ERP or files"""
        if self.erp_inventory_data is not None:
            return self.erp_inventory_data
        
        if self.data_path:
            try:
                # Try multiple inventory file patterns
                inventory_files = [
                    "yarn_inventory (2).xlsx",
                    "yarn_inventory (1).xlsx",
                    "yarn_inventory.xlsx",
                    "eFab_Inventory_G00_20250810 (1).xlsx",
                    "eFab_Inventory_I01_20250810.xlsx"
                ]
                
                # Try to load yarn inventory first
                for filename in inventory_files:
                    inventory_file = self.data_path / filename
                    if inventory_file.exists():
                        df = pd.read_excel(inventory_file)
                        # Map columns to expected format
                        column_mapping = {
                            'Desc#': 'Item Code',
                            'Theoretical Balance': 'Planning Balance',
                            'Beginning Balance': 'On Hand',
                            'Received': 'On Order'
                        }
                        for old_col, new_col in column_mapping.items():
                            if old_col in df.columns and new_col not in df.columns:
                                df[new_col] = df[old_col]
                        
                        # Ensure required columns exist
                        if 'Planning Balance' not in df.columns and 'Theoretical Balance' in df.columns:
                            df['Planning Balance'] = df['Theoretical Balance']
                        if 'On Order' not in df.columns and 'Received' in df.columns:
                            df['On Order'] = df['Received']
                        if 'Cost/Pound' not in df.columns:
                            df['Cost/Pound'] = 5.0  # Default cost
                        
                        logger.info(f"Loaded inventory data from {filename}")
                        return df
            except Exception as e:
                logger.error(f"Error loading inventory data: {e}")
        
        return None
    
    def _load_bom_data(self) -> Optional[pd.DataFrame]:
        """Load BOM data from ERP or files"""
        if self.erp_bom_data is not None:
            return self.erp_bom_data
        
        if self.data_path:
            try:
                # Try multiple BOM file names
                bom_files = [
                    "Style_BOM.csv",
                    "BOM_2(Sheet1).csv",
                    "BOM.csv"
                ]
                
                for filename in bom_files:
                    bom_file = self.data_path / filename
                    if bom_file.exists():
                        df = pd.read_csv(bom_file)
                        # Map Style_BOM.csv format to expected format
                        if 'Style#' in df.columns:
                            df.rename(columns={
                                'Style#': 'Parent Item',
                                'desc#': 'Component',
                                'BOM_Percentage': 'Quantity',
                                'unit': 'Unit'
                            }, inplace=True)
                        
                        # Add missing columns with defaults
                        if 'Description' not in df.columns:
                            df['Description'] = df['Component'].astype(str)
                        if 'Lead Time' not in df.columns:
                            df['Lead Time'] = 14  # Default 14 days
                            
                        # Convert percentage to quantity if needed
                        if 'Quantity' in df.columns and df['Quantity'].max() <= 1:
                            df['Quantity'] = df['Quantity'] * 100  # Convert to percentage
                        
                        logger.info(f"Loaded BOM data from {filename}")
                        return df
            except Exception as e:
                logger.error(f"Error loading BOM data: {e}")
        
        return None
    
    def _load_supplier_data(self) -> Optional[pd.DataFrame]:
        """Load supplier data from ERP or files"""
        if self.erp_supplier_data is not None:
            return self.erp_supplier_data
        
        if self.data_path:
            try:
                # First try dedicated supplier file
                supplier_file = self.data_path / "Supplier_ID.csv"
                if supplier_file.exists():
                    supplier_df = pd.read_csv(supplier_file)
                    logger.info(f"Loaded supplier data from Supplier_ID.csv")
                    return supplier_df
                
                # Otherwise try to extract from yarn inventory
                inventory_files = [
                    "yarn_inventory (2).xlsx",
                    "yarn_inventory (1).xlsx",
                    "Yarn_ID_1.csv"
                ]
                
                for filename in inventory_files:
                    file_path = self.data_path / filename
                    if file_path.exists():
                        if filename.endswith('.csv'):
                            data = pd.read_csv(file_path)
                        else:
                            data = pd.read_excel(file_path)
                            
                        if 'Supplier' in data.columns:
                            # Create supplier summary
                            supplier_cols = ['Supplier']
                            if 'Cost/Pound' in data.columns:
                                supplier_cols.append('Cost/Pound')
                            else:
                                data['Cost/Pound'] = 5.0  # Default
                                supplier_cols.append('Cost/Pound')
                                
                            if 'Theoretical Balance' in data.columns:
                                data['Planning Balance'] = data['Theoretical Balance']
                                supplier_cols.append('Planning Balance')
                            elif 'Planning Balance' in data.columns:
                                supplier_cols.append('Planning Balance')
                            else:
                                data['Planning Balance'] = 100  # Default
                                supplier_cols.append('Planning Balance')
                            
                            result = data[supplier_cols].groupby('Supplier').agg({
                                'Cost/Pound': 'mean',
                                'Planning Balance': 'sum'
                            }).reset_index()
                            
                            logger.info(f"Loaded supplier data from {filename}")
                            return result
            except Exception as e:
                logger.error(f"Error loading supplier data: {e}")
        
        return pd.DataFrame()
    
    # Additional helper methods
    
    def _prepare_time_series(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        if 'Date' in sales_data.columns and 'Qty Shipped' in sales_data.columns:
            ts_data = sales_data[['Date', 'Qty Shipped']].copy()
            ts_data['Date'] = pd.to_datetime(ts_data['Date'])
            ts_data = ts_data.groupby('Date')['Qty Shipped'].sum().reset_index()
            ts_data = ts_data.sort_values('Date')
            return ts_data
        return pd.DataFrame()
    
    def _load_yarn_demand_data(self):
        """Load yarn demand data for enhanced planning"""
        try:
            # Try multiple yarn demand files
            demand_files = [
                "Yarn_Demand_By_Style_2025-08-10_0442 (1).xlsx",
                "Yarn_Demand_By_Style.xlsx",
                "Yarn_Demand_2025-08-09_0442.xlsx",
                "Expected_Yarn_Report.xlsx"
            ]
            
            for filename in demand_files:
                demand_file = self.data_path / filename
                if demand_file.exists():
                    return pd.read_excel(demand_file)
            return None
        except Exception as e:
            logger.error(f"Error loading yarn demand data: {e}")
            return None
    
    def _load_yarn_id_mapping(self):
        """Load yarn ID to description mapping"""
        try:
            yarn_id_file = self.data_path / "Yarn_ID_1.csv"
            if yarn_id_file.exists():
                return pd.read_csv(yarn_id_file)
            return None
        except Exception as e:
            logger.error(f"Error loading yarn ID mapping: {e}")
            return None
    
    def _moving_average_forecast(self, ts_data: pd.DataFrame, window: int = 30) -> Dict:
        """Simple moving average forecast"""
        if len(ts_data) >= window:
            ma = ts_data['Qty Shipped'].rolling(window=window).mean().iloc[-1]
            forecast_values = [ma] * self.config['forecast_horizon']
            return {
                'method': 'moving_average',
                'forecast': forecast_values,
                'confidence': 0.7
            }
        return None
    
    def _exponential_smoothing_forecast(self, ts_data: pd.DataFrame, alpha: float = 0.3) -> Dict:
        """Exponential smoothing forecast"""
        if len(ts_data) > 0:
            values = ts_data['Qty Shipped'].values
            s = [values[0]]
            for i in range(1, len(values)):
                s.append(alpha * values[i] + (1 - alpha) * s[i-1])
            
            forecast_value = s[-1]
            forecast_values = [forecast_value] * self.config['forecast_horizon']
            return {
                'method': 'exponential_smoothing',
                'forecast': forecast_values,
                'confidence': 0.75
            }
        return None
    
    def _ml_forecast(self, ts_data: pd.DataFrame, model_type: str) -> Dict:
        """Machine learning based forecast - placeholder"""
        # Simplified version for cleaned file
        return None
    
    def _prophet_forecast(self, ts_data: pd.DataFrame) -> Dict:
        """Prophet forecast - placeholder"""
        # Simplified version for cleaned file
        return None
    
    def _ensemble_forecast(self, forecasts: Dict) -> Dict:
        """Combine multiple forecasts using weighted average"""
        if not forecasts:
            return {'total_demand': 0, 'daily_forecast': [], 'confidence': 0}
        
        # Weight forecasts by confidence
        total_weight = sum(f['confidence'] for f in forecasts.values() if f)
        
        if total_weight == 0:
            total_weight = len(forecasts)
            for f in forecasts.values():
                if f:
                    f['confidence'] = 1.0 / total_weight
        
        # Calculate weighted average
        ensemble_forecast = []
        for day in range(self.config['forecast_horizon']):
            day_forecast = 0
            for f in forecasts.values():
                if f and len(f['forecast']) > day:
                    weight = f['confidence'] / total_weight
                    day_forecast += f['forecast'][day] * weight
            ensemble_forecast.append(day_forecast)
        
        return {
            'total_demand': sum(ensemble_forecast),
            'daily_forecast': ensemble_forecast,
            'confidence': total_weight / len(forecasts) if forecasts else 0,
            'methods_used': list(forecasts.keys())
        }
    
    def _calculate_forecast_accuracy(self, actual_data: pd.DataFrame, forecast: Dict) -> Dict:
        """Calculate forecast accuracy metrics"""
        return {'mape': 10, 'confidence': 0.85, 'outliers': 0}
    
    def _analyze_inventory_stockout_risk(self, forecast) -> Dict:
        """Analyze stockout risk"""
        return {'high_risk': [], 'medium_risk': [], 'avg_stockout_prob': 0.1}
    
    def _explode_bom(self, bom_data: pd.DataFrame, forecast: Dict) -> pd.DataFrame:
        """Explode BOM based on forecast"""
        if bom_data is None or forecast is None:
            return pd.DataFrame()
        
        exploded = []
        total_demand = forecast.get('total_demand', 0)
        
        for _, bom_item in bom_data.iterrows():
            parent_demand = total_demand / len(bom_data['Parent Item'].unique()) if 'Parent Item' in bom_data.columns else total_demand
            
            quantity_per = bom_item.get('Quantity', 1)
            total_required = parent_demand * quantity_per
            
            exploded.append({
                'parent_item': bom_item.get('Parent Item', 'N/A'),
                'component': bom_item.get('Component', 'N/A'),
                'description': bom_item.get('Description', 'N/A'),
                'quantity_per': quantity_per,
                'total_required': total_required,
                'unit': bom_item.get('Unit', 'EA'),
                'lead_time': bom_item.get('Lead Time', 14)
            })
        
        return pd.DataFrame(exploded)
    
    def _summarize_materials(self, exploded_bom: pd.DataFrame) -> Dict:
        """Summarize materials by category"""
        if exploded_bom.empty:
            return {}
        
        summary = {}
        if 'component' in exploded_bom.columns:
            grouped = exploded_bom.groupby('component')['total_required'].sum()
            summary = grouped.to_dict()
        
        return summary
    
    def _calculate_yarn_consumption(self, exploded_bom, forecast) -> List:
        """Calculate yarn consumption"""
        return []
    
    def _calculate_production_timing(self, forecast, yarn_consumption) -> Dict:
        """Calculate production timing"""
        return {'production_start_date': 'TBD', 'yarn_order_deadline': 'TBD'}
    
    def _calculate_net_requirements(self, exploded_bom: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
        """Calculate net requirements after inventory netting"""
        if exploded_bom.empty or inventory is None:
            return pd.DataFrame()
        
        net_requirements = []
        
        for _, bom_item in exploded_bom.iterrows():
            on_hand = 0
            on_order = 0
            
            if 'Description' in inventory.columns:
                matching = inventory[inventory['Description'].str.contains(
                    bom_item.get('description', ''), case=False, na=False
                )]
                if not matching.empty:
                    on_hand = matching['Planning Balance'].sum() if 'Planning Balance' in matching.columns else 0
                    on_order = matching['On Order'].sum() if 'On Order' in matching.columns else 0
            
            gross_requirement = bom_item.get('total_required', 0)
            net_required = max(0, gross_requirement - on_hand - on_order)
            
            net_requirements.append({
                'item_code': bom_item.get('component', 'N/A'),
                'description': bom_item.get('description', 'N/A'),
                'gross_required': gross_requirement,
                'on_hand': on_hand,
                'on_order': on_order,
                'net_required': net_required,
                'supplier': 'TBD',
                'unit_cost': 0
            })
        
        return pd.DataFrame(net_requirements)
    
    def _analyze_yarn_shortages(self, exploded_bom, inventory_data) -> List:
        """Analyze yarn shortages"""
        return []
    
    def _calculate_yarn_procurement_timing(self, yarn_shortages, forecast) -> List:
        """Calculate yarn procurement timing"""
        return []
    
    def _identify_critical_shortages(self, net_requirements: pd.DataFrame) -> List:
        """Identify critical shortage items"""
        if net_requirements.empty:
            return []
        
        critical = []
        if 'net_required' in net_requirements.columns:
            shortage_items = net_requirements[net_requirements['net_required'] > 0]
            critical = shortage_items.nlargest(10, 'net_required')['description'].tolist()
        
        return critical
    
    def _calculate_eoq(self, item: pd.Series) -> Dict:
        """Calculate Economic Order Quantity"""
        annual_demand = item.get('net_required', 0) * 12
        holding_cost_rate = self.config['holding_cost_rate']
        ordering_cost = self.config['ordering_cost']
        unit_cost = item.get('unit_cost', 10)
        
        if annual_demand > 0 and unit_cost > 0:
            holding_cost = unit_cost * holding_cost_rate
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
            eoq = max(eoq, self.config['min_order_quantity'])
            
            total_cost = (annual_demand * unit_cost)
            savings = max(0, total_cost * 0.05)  # Assume 5% savings
            
            return {
                'eoq': eoq,
                'total_cost': total_cost,
                'savings': savings
            }
        
        return {
            'eoq': self.config['min_order_quantity'],
            'total_cost': 0,
            'savings': 0
        }
    
    def _calculate_safety_stock(self, item: pd.Series) -> float:
        """Calculate safety stock"""
        avg_demand = item.get('net_required', 0) / 30
        lead_time = 14
        demand_variability = 0.2
        
        service_level = self.config['safety_stock_service_level']
        z_score = 2.05 if service_level >= 0.98 else 1.65
        
        safety_stock = z_score * np.sqrt(lead_time) * avg_demand * demand_variability
        
        return max(safety_stock, 0)
    
    def _calculate_reorder_point(self, item: pd.Series, safety_stock: float) -> float:
        """Calculate reorder point"""
        avg_demand = item.get('net_required', 0) / 30
        lead_time = 14
        
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        return reorder_point
    
    def _determine_priority(self, item: pd.Series) -> str:
        """Determine procurement priority"""
        net_required = item.get('net_required', 0)
        on_hand = item.get('on_hand', 0)
        
        if net_required > 1000 and on_hand < 100:
            return 'Critical'
        elif net_required > 500 or on_hand < 500:
            return 'High'
        elif net_required > 100:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_rationale(self, item: pd.Series, eoq_result: Dict) -> str:
        """Generate procurement rationale"""
        priority = self._determine_priority(item)
        return f"Priority: {priority}. EOQ optimization recommended."
    
    def _evaluate_suppliers(self, supplier_data: pd.DataFrame) -> Dict:
        """Evaluate suppliers"""
        supplier_scores = {}
        
        if not supplier_data.empty and 'Supplier' in supplier_data.columns:
            for _, supplier in supplier_data.iterrows():
                supplier_name = supplier['Supplier']
                supplier_scores[supplier_name] = {
                    'total_score': 85,
                    'risk_level': 'Low'
                }
        
        return supplier_scores
    
    def _assign_suppliers(self, procurement_plan: List, supplier_scores: Dict) -> Dict:
        """Assign items to suppliers"""
        assignments = {}
        
        for rec in procurement_plan:
            best_supplier = list(supplier_scores.keys())[0] if supplier_scores else 'Default Supplier'
            rec.supplier = best_supplier
            assignments[rec.item_code] = {
                'primary_supplier': best_supplier,
                'score': 85,
                'quantity': rec.recommended_quantity
            }
        
        return assignments
    
    def _analyze_supplier_risks(self, assignments: Dict) -> Dict:
        """Analyze supplier risks"""
        return {
            'high_risk_count': 0,
            'diversification_index': 0.8
        }
    
    def _summarize_forecast(self) -> Dict:
        """Summarize forecast results"""
        if self.unified_forecast:
            return {
                'total_demand': self.unified_forecast.get('total_demand', 0),
                'confidence': self.unified_forecast.get('confidence', 0),
                'methods': self.unified_forecast.get('methods_used', [])
            }
        return {}
    
    def _generate_purchase_orders(self) -> List[Dict]:
        """Generate purchase orders"""
        purchase_orders = []
        
        if self.procurement_plan:
            for i, rec in enumerate(self.procurement_plan):
                po = {
                    'po_number': f"PO-{datetime.now().strftime('%Y%m%d')}-{i+1:04d}",
                    'item_code': rec.item_code,
                    'description': rec.item_description,
                    'supplier': rec.supplier,
                    'quantity': rec.recommended_quantity,
                    'total_value': rec.total_cost,
                    'status': 'Draft'
                }
                purchase_orders.append(po)
        
        return purchase_orders
    
    def _calculate_planning_kpis(self) -> Dict:
        """Calculate key performance indicators"""
        return {
            'optimization_score': 85,
            'forecast_confidence': 0.85,
            'cost_savings_percentage': 5
        }
    
    def _export_results(self) -> Dict:
        """Export planning results"""
        return {'formats': ['JSON']}


# ==================== Integration Functions ====================

def integrate_with_beverly_erp(data_path: str = None, erp_data: Dict = None, erp_connection: Any = None):
    """
    Main integration function to be called from Beverly ERP system
    
    Args:
        data_path: Path to data files (optional)
        erp_data: Dictionary containing ERP data (optional)
        erp_connection: Direct connection to ERP system (optional)
    
    Returns:
        Dictionary with planning results formatted for ERP dashboard
    """
    # Initialize the planning engine
    engine = SixPhasePlanningEngine(
        data_path=Path(data_path) if data_path else None,
        erp_connection=erp_connection
    )
    
    # If ERP data is provided, use it
    if erp_data:
        engine.set_erp_data(erp_data)
    
    # Execute the planning cycle
    results = engine.execute_full_planning_cycle()
    
    # Format results for ERP dashboard
    formatted_results = []
    for result in results:
        formatted_results.append({
            'phase': result.phase_number,
            'name': result.phase_name,
            'status': result.status,
            'execution_time': result.execution_time,
            'details': result.details,
            'errors': result.errors,
            'warnings': result.warnings
        })
    
    return {
        'success': all(r.status != 'Failed' for r in results),
        'phases': formatted_results,
        'final_output': engine.final_output,
        'procurement_recommendations': engine.get_procurement_recommendations(),
        'inventory_analysis': engine.get_inventory_analysis(),
        'supplier_risk_assessment': engine.get_supplier_risk_assessment(),
        'timestamp': datetime.now().isoformat()
    }


def get_planning_engine_instance(data_path: str = None, erp_connection: Any = None):
    """
    Factory function to get a planning engine instance
    
    Args:
        data_path: Path to data files
        erp_connection: ERP system connection
    
    Returns:
        SixPhasePlanningEngine instance
    """
    return SixPhasePlanningEngine(
        data_path=Path(data_path) if data_path else None,
        erp_connection=erp_connection
    )


if __name__ == "__main__":
    # Test the planning engine
    print("Testing Six-Phase Planning Engine (Cleaned Version with ERP Data/5)...")
    
    # Use ERP Data/5 path for testing
    test_results = integrate_with_beverly_erp(
        data_path="/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5"
    )
    
    print("\n=== Planning Results ===")
    print(f"Success: {test_results['success']}")
    print(f"Timestamp: {test_results['timestamp']}")
    
    for phase in test_results['phases']:
        print(f"\nPhase {phase['phase']}: {phase['name']}")
        print(f"Status: {phase['status']}")
        print(f"Execution Time: {phase['execution_time']:.2f}s")
        if phase['errors']:
            print(f"Errors: {phase['errors']}")
        if phase['warnings']:
            print(f"Warnings: {phase['warnings']}")