#!/usr/bin/env python3
"""
Production Flow Tracker API Endpoints
Provides API routes for production flow tracking functionality
"""

from flask import jsonify, request
from datetime import datetime, timedelta
import json

def register_production_flow_endpoints(app, production_tracker):
    """Register all production flow tracking endpoints"""
    
    @app.route('/api/production-flow/status')
    def production_flow_status():
        """Get current production pipeline status"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            status = production_tracker.get_production_pipeline_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/batch/<batch_id>')
    def get_batch_tracking(batch_id):
        """Track specific batch through stages"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            tracking = production_tracker.get_batch_tracking(batch_id)
            return jsonify(tracking)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/batches')
    def get_all_batches():
        """Get all active production batches"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            batches = []
            for batch_id, batch in production_tracker.production_batches.items():
                batches.append({
                    "batch_id": batch.batch_id,
                    "style_id": batch.style_id,
                    "current_stage": batch.current_stage.value,
                    "current_quantity": batch.current_quantity,
                    "start_quantity": batch.start_quantity,
                    "yield_rate": batch.yield_rate,
                    "days_in_production": batch.days_in_production,
                    "target_date": batch.target_date.isoformat()
                })
            return jsonify(batches)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/create-order', methods=['POST'])
    def create_production_order():
        """Create new production order"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            data = request.json
            
            # Validate required fields
            required = ['style_id', 'quantity', 'bom_data']
            for field in required:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Parse target date if provided
            target_date = datetime.now() + timedelta(days=14)  # Default 2 weeks
            if 'target_date' in data:
                try:
                    target_date = datetime.fromisoformat(data['target_date'])
                except:
                    pass
            
            # Create production order
            batch = production_tracker.create_production_order(
                style_id=data['style_id'],
                quantity=float(data['quantity']),
                bom_data=data['bom_data'],
                target_date=target_date
            )
            
            return jsonify({
                "success": True,
                "batch_id": batch.batch_id,
                "yarn_consumed": batch.yarn_consumed,
                "target_date": batch.target_date.isoformat()
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/move-batch', methods=['POST'])
    def move_batch():
        """Move batch to next stage"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            data = request.json
            
            # Validate required fields
            if 'batch_id' not in data or 'target_stage' not in data:
                return jsonify({"error": "Missing batch_id or target_stage"}), 400
            
            # Import ProductionStage enum
            from production.production_flow_tracker import ProductionStage
            
            # Parse target stage
            try:
                target_stage = ProductionStage[data['target_stage']]
            except KeyError:
                return jsonify({"error": f"Invalid stage: {data['target_stage']}"}), 400
            
            # Get optional parameters
            quantity = data.get('quantity')
            quality_pass_rate = float(data.get('quality_pass_rate', 1.0))
            
            # Move batch
            transition = production_tracker.move_batch_to_stage(
                batch_id=data['batch_id'],
                target_stage=target_stage,
                quantity=quantity,
                quality_pass_rate=quality_pass_rate
            )
            
            return jsonify({
                "success": True,
                "from_stage": transition.from_stage.value,
                "to_stage": transition.to_stage.value,
                "quantity": transition.quantity,
                "timestamp": transition.timestamp.isoformat()
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/f01-replenishment')
    def get_f01_replenishment():
        """Calculate F01 replenishment needs"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            # Get safety stock days from query params
            safety_stock_days = int(request.args.get('safety_stock_days', 20))
            
            replenishment = production_tracker.calculate_f01_replenishment_needs(
                safety_stock_days=safety_stock_days
            )
            
            return jsonify({
                "safety_stock_days": safety_stock_days,
                "replenishment_needs": replenishment
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/production-plan', methods=['POST'])
    def generate_production_plan():
        """Generate production plan based on replenishment needs"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            data = request.json
            
            # Get replenishment needs and BOM data
            replenishment_needs = data.get('replenishment_needs', {})
            bom_data = data.get('bom_data', {})
            
            # If not provided, calculate replenishment needs
            if not replenishment_needs:
                replenishment_needs = production_tracker.calculate_f01_replenishment_needs()
            
            # Generate production plan
            plan = production_tracker.generate_production_plan(
                replenishment_needs=replenishment_needs,
                bom_data=bom_data
            )
            
            return jsonify({
                "production_plan": plan,
                "total_orders": len(plan)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/yarn-impact', methods=['POST'])
    def get_yarn_impact():
        """Analyze yarn impact of production plan"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            data = request.json
            production_plan = data.get('production_plan', [])
            
            if not production_plan:
                return jsonify({"error": "No production plan provided"}), 400
            
            # Analyze yarn impact
            impact = production_tracker.get_yarn_impact_analysis(production_plan)
            
            return jsonify(impact)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/metrics')
    def get_production_metrics():
        """Export comprehensive production metrics"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            metrics = production_tracker.export_production_metrics()
            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/production-flow/capacity')
    def get_capacity_status():
        """Get current capacity utilization by stage"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            capacity_status = {}
            for stage, capacity in production_tracker.capacity_constraints.items():
                current_wip = sum(production_tracker.stage_inventory.get(stage, {}).values())
                utilization = (current_wip / capacity * 100) if capacity < float('inf') else 0
                
                capacity_status[stage.value] = {
                    "capacity": capacity if capacity < float('inf') else "Unlimited",
                    "current_wip": current_wip,
                    "utilization": utilization,
                    "available": capacity - current_wip if capacity < float('inf') else "Unlimited"
                }
            
            return jsonify(capacity_status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Add test data endpoint for development
    @app.route('/api/production-flow/test-data', methods=['POST'])
    def create_test_data():
        """Create test production data for development"""
        try:
            if not production_tracker:
                return jsonify({"error": "Production Flow Tracker not initialized"}), 503
            
            # Create sample BOM data
            test_bom = {
                "STYLE001": {"19003": 0.45, "19004": 0.35, "19005": 0.20},
                "STYLE002": {"19003": 0.30, "19006": 0.40, "19007": 0.30},
                "STYLE003": {"19004": 0.50, "19008": 0.50}
            }
            
            created_batches = []
            
            # Create test batches
            for style_id, bom in test_bom.items():
                try:
                    batch = production_tracker.create_production_order(
                        style_id=style_id,
                        quantity=1000,  # 1000 yards
                        bom_data=bom,
                        target_date=datetime.now() + timedelta(days=14)
                    )
                    created_batches.append(batch.batch_id)
                    
                    # Move first batch through some stages
                    if style_id == "STYLE001":
                        from production.production_flow_tracker import ProductionStage
                        production_tracker.move_batch_to_stage(batch.batch_id, ProductionStage.G00)
                        production_tracker.move_batch_to_stage(batch.batch_id, ProductionStage.G02)
                except Exception as e:
                    print(f"Error creating test batch for {style_id}: {e}")
            
            return jsonify({
                "success": True,
                "created_batches": created_batches,
                "message": f"Created {len(created_batches)} test batches"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    print(f"[OK] Registered {11} production flow tracking endpoints")