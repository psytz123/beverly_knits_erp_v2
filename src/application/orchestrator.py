"""Production orchestrator that coordinates complex workflows across services."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.infrastructure.container.container import Container


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class ProductionOrderRequest:
    """Request for creating a production order."""
    order_id: str
    style_id: str
    customer_name: str
    quantity: float
    due_date: datetime
    priority: int = 3
    notes: Optional[str] = None


@dataclass
class ProductionWorkflowResult:
    """Result of production workflow execution."""
    status: WorkflowStatus
    plan: Optional[Dict[str, Any]] = None
    machine_assignment: Optional[Dict[str, Any]] = None
    material_requirements: Optional[List[Dict[str, Any]]] = None
    shortages: Optional[List[Dict[str, Any]]] = None
    forecast: Optional[Dict[str, Any]] = None
    purchase_orders: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = None
    execution_time_ms: float = 0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ProductionOrchestrator:
    """
    Orchestrates complex production workflows across multiple services.
    Implements the complete production planning workflow from the plan.
    """
    
    def __init__(self, container: Container):
        """Initialize orchestrator with DI container."""
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Get services from container
        self.inventory = container.inventory_analyzer()
        self.forecasting = container.enhanced_forecasting()
        self.production = container.production_pipeline()
        self.yarn_service = container.yarn_intelligence()
        self.planning_engine = container.six_phase_planning()
        self.business_rules = container.business_rules()
        self.cache = container.cache_manager()
    
    async def complete_production_workflow(
        self, 
        order_request: ProductionOrderRequest
    ) -> ProductionWorkflowResult:
        """
        Execute complete production workflow:
        1. Check yarn availability
        2. Generate demand forecast
        3. Calculate material requirements
        4. Create production plan
        5. Assign to machines
        """
        start_time = datetime.utcnow()
        result = ProductionWorkflowResult(status=WorkflowStatus.IN_PROGRESS)
        
        try:
            # Step 1: Check current inventory status
            self.logger.info(f"Step 1: Checking inventory for order {order_request.order_id}")
            inventory_status = await self._check_inventory_status()
            
            # Step 2: Generate demand forecast
            self.logger.info(f"Step 2: Generating forecast for style {order_request.style_id}")
            forecast = await self._generate_forecast(
                order_request.style_id,
                order_request.due_date
            )
            result.forecast = forecast
            
            # Step 3: Calculate material requirements
            self.logger.info(f"Step 3: Calculating material requirements")
            requirements = await self._calculate_requirements(
                order_request.style_id,
                order_request.quantity,
                forecast
            )
            result.material_requirements = requirements
            
            # Step 4: Check material availability and identify shortages
            self.logger.info(f"Step 4: Checking material availability")
            availability_check = await self._check_availability(
                requirements,
                inventory_status
            )
            result.shortages = availability_check['shortages']
            
            # Step 5: Handle shortages if any
            if result.shortages:
                self.logger.warning(f"Found {len(result.shortages)} material shortages")
                
                # Generate purchase orders for shortages
                purchase_orders = await self._generate_purchase_orders(result.shortages)
                result.purchase_orders = purchase_orders
                
                # Check for substitutes
                substitutes = await self._find_substitutes(result.shortages)
                
                # Create partial production plan if possible
                if self._can_produce_partial(availability_check):
                    result.plan = await self._create_partial_plan(
                        order_request,
                        availability_check,
                        substitutes
                    )
                    result.warnings.append("Partial production plan created due to material shortages")
                else:
                    result.status = WorkflowStatus.FAILED
                    result.warnings.append("Cannot create production plan due to critical material shortages")
                    return result
            else:
                # Create full production plan
                self.logger.info(f"Step 5: Creating production plan")
                result.plan = await self._create_production_plan(order_request)
            
            # Step 6: Machine assignment
            if result.plan:
                self.logger.info(f"Step 6: Assigning to machines")
                machine_assignment = await self._assign_to_machines(result.plan)
                result.machine_assignment = machine_assignment
                
                if machine_assignment['assigned']:
                    result.status = WorkflowStatus.COMPLETED
                else:
                    result.status = WorkflowStatus.PARTIALLY_COMPLETED
                    result.warnings.append("Production plan created but machine assignment pending")
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            self.logger.info(f"Workflow completed in {execution_time:.2f}ms with status: {result.status.value}")
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            result.status = WorkflowStatus.FAILED
            result.warnings.append(f"Workflow error: {str(e)}")
        
        return result
    
    async def _check_inventory_status(self) -> Dict[str, Any]:
        """Check current inventory status."""
        try:
            return self.inventory.get_inventory_summary()
        except Exception as e:
            self.logger.error(f"Failed to get inventory status: {e}")
            return {}
    
    async def _generate_forecast(
        self, 
        style_id: str, 
        due_date: datetime
    ) -> Dict[str, Any]:
        """Generate demand forecast for the style."""
        try:
            horizon_days = (due_date - datetime.utcnow()).days
            horizon_days = max(30, min(180, horizon_days))  # Clamp between 30-180 days
            
            forecast = self.forecasting.generate_forecast(
                style_id=style_id,
                horizon_days=horizon_days,
                model_type='ensemble'
            )
            
            return {
                'style_id': style_id,
                'horizon_days': horizon_days,
                'predicted_demand': forecast.get('total_demand', 0),
                'confidence': forecast.get('confidence', 0.95),
                'model': forecast.get('model', 'ensemble')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast: {e}")
            return {
                'style_id': style_id,
                'horizon_days': 30,
                'predicted_demand': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    async def _calculate_requirements(
        self,
        style_id: str,
        quantity: float,
        forecast: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate material requirements including forecast."""
        try:
            # Get base requirements from BOM
            base_requirements = self.yarn_service.calculate_yarn_requirements(
                style_id=style_id,
                quantity=quantity
            )
            
            # Adjust for forecast if available
            if forecast and forecast.get('predicted_demand', 0) > 0:
                forecast_multiplier = 1 + (forecast['predicted_demand'] / quantity * 0.1)  # 10% buffer
                
                for req in base_requirements:
                    req['quantity_with_forecast'] = req['quantity'] * forecast_multiplier
                    req['forecast_adjustment'] = forecast_multiplier - 1
            
            return base_requirements
            
        except Exception as e:
            self.logger.error(f"Failed to calculate requirements: {e}")
            return []
    
    async def _check_availability(
        self,
        requirements: List[Dict[str, Any]],
        inventory_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check material availability against requirements."""
        shortages = []
        available_materials = []
        total_required = 0
        total_available = 0
        
        for req in requirements:
            yarn_id = req.get('yarn_id')
            required_qty = req.get('quantity_with_forecast', req.get('quantity', 0))
            
            # Get current balance
            current_balance = self.inventory.get_yarn_balance(yarn_id)
            
            total_required += required_qty
            total_available += max(0, current_balance)
            
            if current_balance < required_qty:
                shortage = {
                    'yarn_id': yarn_id,
                    'required': required_qty,
                    'available': max(0, current_balance),
                    'shortage': required_qty - max(0, current_balance),
                    'percentage_available': (max(0, current_balance) / required_qty * 100) if required_qty > 0 else 0
                }
                shortages.append(shortage)
            else:
                available_materials.append({
                    'yarn_id': yarn_id,
                    'required': required_qty,
                    'available': current_balance
                })
        
        return {
            'shortages': shortages,
            'available_materials': available_materials,
            'can_produce_full': len(shortages) == 0,
            'can_produce_partial': total_available >= (total_required * 0.5),  # Can produce if 50% available
            'availability_percentage': (total_available / total_required * 100) if total_required > 0 else 100
        }
    
    def _can_produce_partial(self, availability_check: Dict[str, Any]) -> bool:
        """Check if partial production is possible."""
        return availability_check.get('can_produce_partial', False)
    
    async def _generate_purchase_orders(
        self,
        shortages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate purchase orders for material shortages."""
        purchase_orders = []
        
        for shortage in shortages:
            yarn_id = shortage['yarn_id']
            shortage_qty = shortage['shortage']
            
            # Get yarn details
            yarn_info = self.yarn_service.get_yarn_details(yarn_id)
            
            # Calculate order quantity (add buffer)
            order_qty = shortage_qty * 1.2  # 20% buffer
            
            # Get supplier info
            supplier = yarn_info.get('supplier', 'Default Supplier')
            lead_time = yarn_info.get('lead_time_days', 14)
            cost_per_unit = yarn_info.get('cost_per_unit', 0)
            
            po = {
                'po_number': f"PO-{datetime.utcnow().strftime('%Y%m%d')}-{yarn_id}",
                'yarn_id': yarn_id,
                'description': yarn_info.get('description', ''),
                'quantity': order_qty,
                'supplier': supplier,
                'lead_time_days': lead_time,
                'estimated_cost': order_qty * cost_per_unit,
                'expected_delivery': (datetime.utcnow() + timedelta(days=lead_time)).isoformat(),
                'status': 'pending',
                'created_at': datetime.utcnow().isoformat()
            }
            
            purchase_orders.append(po)
        
        return purchase_orders
    
    async def _find_substitutes(
        self,
        shortages: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find substitute yarns for shortages."""
        substitutes = {}
        
        yarn_matcher = self.container.intelligent_yarn_matcher()
        
        for shortage in shortages:
            yarn_id = shortage['yarn_id']
            
            try:
                # Find potential substitutes
                substitute_options = yarn_matcher.find_substitutes(
                    yarn_id=yarn_id,
                    required_quantity=shortage['shortage']
                )
                
                substitutes[yarn_id] = substitute_options
                
            except Exception as e:
                self.logger.error(f"Failed to find substitutes for {yarn_id}: {e}")
                substitutes[yarn_id] = []
        
        return substitutes
    
    async def _create_partial_plan(
        self,
        order_request: ProductionOrderRequest,
        availability_check: Dict[str, Any],
        substitutes: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Create partial production plan with available materials."""
        # Calculate producible quantity
        availability_pct = availability_check.get('availability_percentage', 0) / 100
        producible_qty = order_request.quantity * availability_pct
        
        plan = {
            'order_id': order_request.order_id,
            'style_id': order_request.style_id,
            'original_quantity': order_request.quantity,
            'producible_quantity': producible_qty,
            'production_type': 'partial',
            'availability_percentage': availability_check['availability_percentage'],
            'shortages': availability_check['shortages'],
            'substitute_options': substitutes,
            'estimated_completion_date': order_request.due_date.isoformat(),
            'status': 'partial_ready',
            'created_at': datetime.utcnow().isoformat()
        }
        
        return plan
    
    async def _create_production_plan(
        self,
        order_request: ProductionOrderRequest
    ) -> Dict[str, Any]:
        """Create full production plan."""
        try:
            # Use six-phase planning engine
            plan = self.planning_engine.create_production_plan(
                order_id=order_request.order_id,
                style_id=order_request.style_id,
                quantity=order_request.quantity,
                due_date=order_request.due_date,
                priority=order_request.priority
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create production plan: {e}")
            
            # Fallback to simple plan
            return {
                'order_id': order_request.order_id,
                'style_id': order_request.style_id,
                'quantity': order_request.quantity,
                'production_type': 'full',
                'phases': ['planning', 'preparation', 'production', 'finishing', 'quality', 'shipping'],
                'estimated_completion_date': order_request.due_date.isoformat(),
                'status': 'ready',
                'created_at': datetime.utcnow().isoformat()
            }
    
    async def _assign_to_machines(
        self,
        production_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign production plan to available machines."""
        try:
            # Get machine suggestions
            suggestions = self.production.get_machine_assignment_suggestions(
                style_id=production_plan['style_id'],
                quantity=production_plan.get('producible_quantity', production_plan['quantity'])
            )
            
            if suggestions and len(suggestions) > 0:
                # Assign to best available machine
                best_machine = suggestions[0]
                
                assignment = {
                    'assigned': True,
                    'machine_id': best_machine['machine_id'],
                    'work_center': best_machine['work_center'],
                    'estimated_hours': best_machine.get('estimated_hours', 0),
                    'utilization': best_machine.get('utilization', 0),
                    'assignment_time': datetime.utcnow().isoformat()
                }
            else:
                assignment = {
                    'assigned': False,
                    'reason': 'No suitable machines available',
                    'suggestions': suggestions
                }
            
            return assignment
            
        except Exception as e:
            self.logger.error(f"Failed to assign to machines: {e}")
            return {
                'assigned': False,
                'error': str(e)
            }