#!/usr/bin/env python3
"""
Data Consistency API - Endpoints for validating and reconciling data consistency
"""

from flask import Blueprint, jsonify, request
import logging
import pandas as pd
from typing import Dict, Any

# Import consistency manager and validation rules
try:
    from data_consistency.consistency_manager import DataConsistencyManager
    from data_consistency.validation_rules import DataValidationRules
    CONSISTENCY_AVAILABLE = True
except ImportError:
    try:
        from src.data_consistency.consistency_manager import DataConsistencyManager
        from src.data_consistency.validation_rules import DataValidationRules
        CONSISTENCY_AVAILABLE = True
    except ImportError:
        CONSISTENCY_AVAILABLE = False
        DataConsistencyManager = None
        DataValidationRules = None

logger = logging.getLogger(__name__)

# Create the blueprint
data_consistency_bp = Blueprint('data_consistency', __name__)

# Get analyzer from global scope (set by main ERP app)
def get_analyzer():
    """Get the analyzer from the global scope"""
    try:
        import sys
        # Look for the analyzer in the main module's globals
        for module_name in ['__main__', 'beverly_comprehensive_erp']:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, 'analyzer'):
                    return module.analyzer
        return None
    except Exception:
        return None


@data_consistency_bp.route("/data-consistency-check")
def data_consistency_check():
    """
    Comprehensive data consistency check across all modules
    Returns discrepancies and validation results
    """
    try:
        if not CONSISTENCY_AVAILABLE:
            return jsonify({
                'error': 'Data consistency tools not available',
                'available': False
            }), 503
        
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({
                'error': 'Analyzer not available',
                'available': False
            }), 503
        
        # Get the data
        inventory_df = analyzer.raw_materials_data if hasattr(analyzer, 'raw_materials_data') else pd.DataFrame()
        bom_df = analyzer.bom_data if hasattr(analyzer, 'bom_data') else pd.DataFrame()
        production_df = analyzer.knit_orders_data if hasattr(analyzer, 'knit_orders_data') else pd.DataFrame()
        
        if inventory_df.empty:
            return jsonify({
                'error': 'No inventory data available for consistency check',
                'available': False
            }), 404
        
        # Generate comprehensive reconciliation report
        report = DataConsistencyManager.create_reconciliation_report(
            inventory_df, bom_df, production_df
        )
        
        # Add individual validation results
        validation_results = {
            'inventory': DataValidationRules.validate_yarn_inventory(inventory_df),
            'bom': DataValidationRules.validate_bom(bom_df) if not bom_df.empty else {'is_valid': True, 'warnings': ['No BOM data available']},
            'production': DataValidationRules.validate_production_orders(production_df) if not production_df.empty else {'is_valid': True, 'warnings': ['No production data available']}
        }
        
        # Cross-validate data between sources
        cross_validation = DataValidationRules.cross_validate_data(
            inventory_df, bom_df, production_df
        ) if not bom_df.empty and not production_df.empty else {'is_valid': True, 'cross_checks': []}
        
        # Combine all results
        consistency_report = {
            'summary': {
                'overall_consistent': report['validation']['is_consistent'] and all(v['is_valid'] for v in validation_results.values()),
                'total_discrepancies': len(report['validation']['discrepancies']),
                'total_warnings': sum(len(v.get('warnings', [])) for v in validation_results.values()),
                'data_sources_checked': 3,
                'data_sources_available': sum(1 for df in [inventory_df, bom_df, production_df] if not df.empty)
            },
            'reconciliation': report,
            'validation': validation_results,
            'cross_validation': cross_validation,
            'recommendations': report['recommendations']
        }
        
        return jsonify(consistency_report)
        
    except Exception as e:
        logger.error(f"Error in data consistency check: {e}")
        return jsonify({
            'error': f'Data consistency check failed: {str(e)}',
            'available': False
        }), 500


@data_consistency_bp.route("/shortage-consistency")
def shortage_consistency():
    """
    Check consistency of shortage calculations across different methods
    """
    try:
        if not CONSISTENCY_AVAILABLE:
            return jsonify({
                'error': 'Data consistency tools not available'
            }), 503
        
        analyzer = get_analyzer()
        if not analyzer or not hasattr(analyzer, 'raw_materials_data'):
            return jsonify({
                'error': 'No inventory data available'
            }), 404
        
        inventory_df = analyzer.raw_materials_data.copy()
        
        # Method 1: Use consistency manager
        consistent_shortages = []
        inventory_standardized = DataConsistencyManager.standardize_columns(inventory_df)
        
        for _, yarn_row in inventory_standardized.iterrows():
            shortage_info = DataConsistencyManager.calculate_yarn_shortage(yarn_row)
            if shortage_info['has_shortage']:
                consistent_shortages.append({
                    'yarn_id': shortage_info['yarn_id'],
                    'shortage_amount': shortage_info['shortage_amount'],
                    'risk_level': shortage_info['risk_level'],
                    'method': 'consistent'
                })
        
        # Method 2: Legacy planning balance only
        legacy_shortages = []
        for _, yarn in inventory_df.iterrows():
            planning_balance = yarn.get('Planning Balance', 0)
            if planning_balance < 0:
                yarn_id = str(yarn.get('Desc#', ''))
                legacy_shortages.append({
                    'yarn_id': yarn_id,
                    'shortage_amount': abs(planning_balance),
                    'risk_level': 'CRITICAL' if planning_balance < -1000 else 'HIGH' if planning_balance < -500 else 'MEDIUM',
                    'method': 'legacy'
                })
        
        # Compare results
        consistent_yarn_ids = {s['yarn_id'] for s in consistent_shortages}
        legacy_yarn_ids = {s['yarn_id'] for s in legacy_shortages}
        
        discrepancies = []
        
        # Check for yarns that appear in one method but not the other
        only_in_consistent = consistent_yarn_ids - legacy_yarn_ids
        only_in_legacy = legacy_yarn_ids - consistent_yarn_ids
        
        if only_in_consistent:
            discrepancies.append({
                'type': 'MISSING_FROM_LEGACY',
                'message': f"{len(only_in_consistent)} yarns show shortage in consistent method but not legacy",
                'yarn_ids': list(only_in_consistent)[:10]
            })
        
        if only_in_legacy:
            discrepancies.append({
                'type': 'MISSING_FROM_CONSISTENT',
                'message': f"{len(only_in_legacy)} yarns show shortage in legacy method but not consistent",
                'yarn_ids': list(only_in_legacy)[:10]
            })
        
        # Check for different shortage amounts for same yarn
        amount_differences = []
        for consistent_yarn in consistent_shortages:
            legacy_yarn = next((y for y in legacy_shortages if y['yarn_id'] == consistent_yarn['yarn_id']), None)
            if legacy_yarn:
                diff = abs(consistent_yarn['shortage_amount'] - legacy_yarn['shortage_amount'])
                if diff > 1:  # More than 1 lb difference
                    amount_differences.append({
                        'yarn_id': consistent_yarn['yarn_id'],
                        'consistent_amount': consistent_yarn['shortage_amount'],
                        'legacy_amount': legacy_yarn['shortage_amount'],
                        'difference': diff
                    })
        
        if amount_differences:
            discrepancies.append({
                'type': 'AMOUNT_DIFFERENCES',
                'message': f"{len(amount_differences)} yarns have different shortage amounts",
                'differences': amount_differences[:10]
            })
        
        return jsonify({
            'comparison': {
                'consistent_method': {
                    'total_shortages': len(consistent_shortages),
                    'shortages': consistent_shortages[:20]
                },
                'legacy_method': {
                    'total_shortages': len(legacy_shortages),
                    'shortages': legacy_shortages[:20]
                }
            },
            'consistency_check': {
                'is_consistent': len(discrepancies) == 0,
                'discrepancies': discrepancies,
                'match_percentage': len(consistent_yarn_ids & legacy_yarn_ids) / max(len(consistent_yarn_ids | legacy_yarn_ids), 1) * 100
            },
            'recommendations': [
                'Use consistent method for all shortage calculations',
                'Update legacy code to use DataConsistencyManager',
                'Implement automated consistency checks'
            ] if discrepancies else ['Methods are consistent']
        })
        
    except Exception as e:
        logger.error(f"Error in shortage consistency check: {e}")
        return jsonify({
            'error': f'Shortage consistency check failed: {str(e)}'
        }), 500


@data_consistency_bp.route("/data-quality-metrics")
def data_quality_metrics():
    """
    Get comprehensive data quality metrics
    """
    try:
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({'error': 'Analyzer not available'}), 503
        
        metrics = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_sources': {},
            'overall_quality': 'UNKNOWN'
        }
        
        # Inventory data quality
        if hasattr(analyzer, 'raw_materials_data') and analyzer.raw_materials_data is not None:
            inventory_df = analyzer.raw_materials_data
            
            # Column completeness
            required_columns = ['Desc#', 'Planning Balance', 'Theoretical Balance', 'Allocated']
            column_completeness = {}
            for col in required_columns:
                if col in inventory_df.columns:
                    non_null_count = inventory_df[col].count()
                    total_count = len(inventory_df)
                    column_completeness[col] = (non_null_count / total_count * 100) if total_count > 0 else 0
                else:
                    column_completeness[col] = 0
            
            # Data consistency metrics
            duplicate_count = inventory_df.duplicated(subset=['Desc#'], keep=False).sum() if 'Desc#' in inventory_df.columns else 0
            negative_balance_count = (inventory_df['Planning Balance'] < 0).sum() if 'Planning Balance' in inventory_df.columns else 0
            
            metrics['data_sources']['inventory'] = {
                'total_records': len(inventory_df),
                'column_completeness': column_completeness,
                'duplicate_records': int(duplicate_count),
                'negative_balances': int(negative_balance_count),
                'quality_score': sum(column_completeness.values()) / len(column_completeness) if column_completeness else 0
            }
        
        # BOM data quality
        if hasattr(analyzer, 'bom_data') and analyzer.bom_data is not None:
            bom_df = analyzer.bom_data
            
            # Check BOM completeness
            unique_styles = bom_df['Style#'].nunique() if 'Style#' in bom_df.columns else 0
            total_mappings = len(bom_df)
            avg_yarns_per_style = total_mappings / unique_styles if unique_styles > 0 else 0
            
            metrics['data_sources']['bom'] = {
                'total_records': len(bom_df),
                'unique_styles': unique_styles,
                'total_mappings': total_mappings,
                'avg_yarns_per_style': round(avg_yarns_per_style, 2),
                'quality_score': 90 if unique_styles > 0 and total_mappings > 0 else 50
            }
        
        # Production data quality
        if hasattr(analyzer, 'knit_orders_data') and analyzer.knit_orders_data is not None:
            production_df = analyzer.knit_orders_data
            
            assigned_orders = 0
            total_orders = len(production_df)
            if 'Machine' in production_df.columns:
                assigned_orders = (production_df['Machine'].notna() & (production_df['Machine'] != '')).sum()
            
            metrics['data_sources']['production'] = {
                'total_records': total_orders,
                'assigned_orders': int(assigned_orders),
                'unassigned_orders': total_orders - assigned_orders,
                'assignment_rate': (assigned_orders / total_orders * 100) if total_orders > 0 else 0,
                'quality_score': (assigned_orders / total_orders * 100) if total_orders > 0 else 0
            }
        
        # Calculate overall quality score
        quality_scores = [source['quality_score'] for source in metrics['data_sources'].values() if 'quality_score' in source]
        if quality_scores:
            overall_score = sum(quality_scores) / len(quality_scores)
            if overall_score >= 90:
                metrics['overall_quality'] = 'EXCELLENT'
            elif overall_score >= 75:
                metrics['overall_quality'] = 'GOOD'
            elif overall_score >= 60:
                metrics['overall_quality'] = 'FAIR'
            else:
                metrics['overall_quality'] = 'POOR'
            
            metrics['overall_score'] = round(overall_score, 2)
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting data quality metrics: {e}")
        return jsonify({
            'error': f'Data quality metrics failed: {str(e)}'
        }), 500