#!/usr/bin/env python3
"""
Legacy System Integration Framework
Intelligent connectors for discovering, mapping, and migrating data from legacy ERP systems
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
import asyncio

logger = logging.getLogger(__name__)


class LegacySystemType(Enum):
    """Types of legacy systems we can integrate with"""
    SAP = "SAP"
    ORACLE_ERP = "ORACLE_ERP"
    QUICKBOOKS = "QUICKBOOKS"
    EXCEL_FILES = "EXCEL_FILES"
    CSV_FILES = "CSV_FILES"
    DATABASE_GENERIC = "DATABASE_GENERIC"
    REST_API = "REST_API"
    CUSTOM_ERP = "CUSTOM_ERP"


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "EXCELLENT"  # >95% complete, consistent
    GOOD = "GOOD"           # 85-95% complete, mostly consistent
    FAIR = "FAIR"           # 70-85% complete, some issues
    POOR = "POOR"           # <70% complete, significant issues


@dataclass
class ColumnMapping:
    """Maps legacy column to framework standard"""
    legacy_name: str
    standard_name: str
    data_type: str
    transformation_rule: Optional[str] = None
    confidence_score: float = 1.0
    validation_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "legacy_name": self.legacy_name,
            "standard_name": self.standard_name,
            "data_type": self.data_type,
            "transformation_rule": self.transformation_rule,
            "confidence_score": self.confidence_score,
            "validation_rules": self.validation_rules
        }


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    table_name: str
    total_rows: int
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    quality_level: DataQualityLevel
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score"""
        return (self.completeness_score + self.consistency_score + self.accuracy_score) / 3


@dataclass
class SchemaAnalysisResult:
    """Results of legacy schema analysis"""
    system_type: LegacySystemType
    tables_found: List[str]
    total_columns: int
    column_mappings: Dict[str, List[ColumnMapping]]
    confidence_score: float
    data_quality: Dict[str, DataQualityReport]
    migration_complexity: str  # LOW, MEDIUM, HIGH
    estimated_migration_hours: int


class AutoSchemaAnalyzer:
    """
    Automatically analyzes legacy system schemas and identifies data patterns
    Handles the 300+ column variations mentioned in AI.MD
    """
    
    def __init__(self):
        self.logger = logging.getLogger("AutoSchemaAnalyzer")
        
        # Standard framework schema patterns
        self.framework_schema = {
            "inventory": {
                "item_id": ["id", "item_id", "part_number", "sku", "product_id", "material_id"],
                "description": ["description", "desc", "item_name", "product_name", "name", "title"],
                "on_hand_qty": ["on_hand", "physical_inventory", "current_stock", "available", "qty_available"],
                "allocated_qty": ["allocated", "committed", "reserved", "on_order_allocated"],
                "on_order_qty": ["on_order", "purchase_order", "po_qty", "ordered", "incoming"],
                "unit_cost": ["cost", "unit_cost", "standard_cost", "average_cost", "price"],
                "reorder_point": ["reorder_point", "min_stock", "safety_stock", "minimum_qty"],
                "lead_time": ["lead_time", "delivery_time", "procurement_time", "supplier_lead_time"]
            },
            "production": {
                "order_id": ["order_id", "work_order", "production_order", "job_number", "wo_id"],
                "item_id": ["item_id", "product_id", "part_number", "sku", "style_id"],
                "quantity": ["quantity", "qty", "order_qty", "production_qty", "target_qty"],
                "due_date": ["due_date", "target_date", "completion_date", "delivery_date"],
                "status": ["status", "state", "phase", "production_status", "order_status"],
                "machine_id": ["machine", "machine_id", "work_center", "resource", "equipment"]
            },
            "bom": {
                "parent_id": ["parent", "parent_id", "finished_good", "assembly", "style_id"],
                "child_id": ["child", "component", "material", "ingredient", "raw_material_id"],
                "quantity": ["quantity", "qty", "usage", "consumption", "requirement"],
                "unit": ["unit", "uom", "unit_measure", "measurement_unit"]
            }
        }
        
        # Common transformation patterns
        self.transformation_patterns = {
            "currency_cleanup": r'\$|,',
            "percentage_conversion": r'%',
            "date_formats": ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"],
            "boolean_patterns": {
                "true": ["true", "yes", "y", "1", "active", "enabled"],
                "false": ["false", "no", "n", "0", "inactive", "disabled"]
            }
        }
    
    async def analyze_schema(self, connection_params: Dict[str, Any]) -> SchemaAnalysisResult:
        """Analyze legacy system schema and generate mapping recommendations"""
        try:
            system_type = self._detect_system_type(connection_params)
            self.logger.info(f"Analyzing {system_type.value} system")
            
            # Discover tables and columns
            tables_found = await self._discover_tables(connection_params)
            column_mappings = {}
            total_columns = 0
            
            for table in tables_found:
                columns = await self._discover_columns(connection_params, table)
                total_columns += len(columns)
                
                # Generate mappings for this table
                mappings = self._generate_column_mappings(table, columns)
                if mappings:
                    column_mappings[table] = mappings
            
            # Assess data quality
            data_quality = {}
            for table in tables_found:
                quality_report = await self._assess_data_quality(connection_params, table)
                data_quality[table] = quality_report
            
            # Calculate overall confidence and complexity
            confidence_score = self._calculate_mapping_confidence(column_mappings)
            migration_complexity, estimated_hours = self._estimate_migration_complexity(
                column_mappings, data_quality
            )
            
            result = SchemaAnalysisResult(
                system_type=system_type,
                tables_found=tables_found,
                total_columns=total_columns,
                column_mappings=column_mappings,
                confidence_score=confidence_score,
                data_quality=data_quality,
                migration_complexity=migration_complexity,
                estimated_migration_hours=estimated_hours
            )
            
            self.logger.info(f"Schema analysis completed: {total_columns} columns across {len(tables_found)} tables")
            return result
            
        except Exception as e:
            self.logger.error(f"Schema analysis failed: {str(e)}")
            raise
    
    def _detect_system_type(self, connection_params: Dict[str, Any]) -> LegacySystemType:
        """Detect the type of legacy system based on connection parameters"""
        if "file_path" in connection_params:
            file_path = Path(connection_params["file_path"])
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                return LegacySystemType.EXCEL_FILES
            elif file_path.suffix.lower() == '.csv':
                return LegacySystemType.CSV_FILES
        
        elif "database_type" in connection_params:
            db_type = connection_params["database_type"].upper()
            if "SAP" in db_type:
                return LegacySystemType.SAP
            elif "ORACLE" in db_type:
                return LegacySystemType.ORACLE_ERP
            else:
                return LegacySystemType.DATABASE_GENERIC
        
        elif "api_url" in connection_params:
            return LegacySystemType.REST_API
        
        return LegacySystemType.CUSTOM_ERP
    
    async def _discover_tables(self, connection_params: Dict[str, Any]) -> List[str]:
        """Discover available tables/sheets in legacy system"""
        system_type = self._detect_system_type(connection_params)
        
        if system_type == LegacySystemType.EXCEL_FILES:
            file_path = connection_params["file_path"]
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
        
        elif system_type == LegacySystemType.CSV_FILES:
            # For CSV, assume single table with filename
            file_path = Path(connection_params["file_path"])
            return [file_path.stem]
        
        elif system_type == LegacySystemType.DATABASE_GENERIC:
            # Would connect to database and get table list
            # For now, return common ERP tables
            return ["inventory", "production_orders", "bom", "sales", "purchases"]
        
        else:
            return ["main_data"]  # Fallback
    
    async def _discover_columns(self, connection_params: Dict[str, Any], table_name: str) -> List[str]:
        """Discover columns in a specific table"""
        system_type = self._detect_system_type(connection_params)
        
        if system_type == LegacySystemType.EXCEL_FILES:
            file_path = connection_params["file_path"]
            df = pd.read_excel(file_path, sheet_name=table_name, nrows=0)
            return df.columns.tolist()
        
        elif system_type == LegacySystemType.CSV_FILES:
            file_path = connection_params["file_path"]
            df = pd.read_csv(file_path, nrows=0)
            return df.columns.tolist()
        
        else:
            # Would query database schema
            return []  # Placeholder
    
    def _generate_column_mappings(self, table_name: str, columns: List[str]) -> List[ColumnMapping]:
        """Generate intelligent column mappings using pattern matching"""
        mappings = []
        
        # Determine which framework schema to use based on table name
        framework_table = self._classify_table(table_name)
        if not framework_table:
            return mappings
        
        schema = self.framework_schema.get(framework_table, {})
        
        for column in columns:
            best_match = self._find_best_column_match(column, schema)
            if best_match:
                standard_name, confidence = best_match
                
                # Determine transformation rule
                transformation_rule = self._determine_transformation(column, standard_name)
                
                # Create mapping
                mapping = ColumnMapping(
                    legacy_name=column,
                    standard_name=standard_name,
                    data_type=self._infer_data_type(column, standard_name),
                    transformation_rule=transformation_rule,
                    confidence_score=confidence,
                    validation_rules=self._generate_validation_rules(standard_name)
                )
                
                mappings.append(mapping)
        
        return mappings
    
    def _classify_table(self, table_name: str) -> Optional[str]:
        """Classify table into framework category"""
        name_lower = table_name.lower()
        
        inventory_keywords = ["inventory", "stock", "item", "material", "raw_material", "part"]
        production_keywords = ["production", "order", "work", "manufacturing", "job", "process"]
        bom_keywords = ["bom", "bill", "material", "recipe", "component"]
        
        if any(keyword in name_lower for keyword in inventory_keywords):
            return "inventory"
        elif any(keyword in name_lower for keyword in production_keywords):
            return "production"
        elif any(keyword in name_lower for keyword in bom_keywords):
            return "bom"
        
        return None
    
    def _find_best_column_match(self, column: str, schema: Dict[str, List[str]]) -> Optional[Tuple[str, float]]:
        """Find best matching framework column with confidence score"""
        column_lower = column.lower().replace("_", "").replace(" ", "")
        best_match = None
        best_score = 0.0
        
        for standard_name, patterns in schema.items():
            for pattern in patterns:
                pattern_lower = pattern.lower().replace("_", "").replace(" ", "")
                
                # Exact match
                if column_lower == pattern_lower:
                    return (standard_name, 1.0)
                
                # Contains match
                if pattern_lower in column_lower or column_lower in pattern_lower:
                    score = min(len(pattern_lower), len(column_lower)) / max(len(pattern_lower), len(column_lower))
                    if score > best_score:
                        best_match = standard_name
                        best_score = score
                
                # Fuzzy match using common abbreviations
                if self._fuzzy_match(column_lower, pattern_lower) > best_score:
                    best_match = standard_name
                    best_score = self._fuzzy_match(column_lower, pattern_lower)
        
        return (best_match, best_score) if best_match else None
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Simple fuzzy matching algorithm"""
        if str1 == str2:
            return 1.0
        
        # Check for common abbreviations
        abbrev_map = {
            "qty": "quantity",
            "desc": "description", 
            "id": "identifier",
            "num": "number",
            "amt": "amount"
        }
        
        # Normalize strings
        norm1 = abbrev_map.get(str1, str1)
        norm2 = abbrev_map.get(str2, str2)
        
        if norm1 == norm2:
            return 0.9
        
        # Simple similarity based on common characters
        common_chars = set(norm1) & set(norm2)
        total_chars = len(set(norm1) | set(norm2))
        
        if total_chars == 0:
            return 0.0
        
        return len(common_chars) / total_chars
    
    def _determine_transformation(self, legacy_column: str, standard_name: str) -> Optional[str]:
        """Determine if column needs data transformation"""
        legacy_lower = legacy_column.lower()
        
        # Currency fields
        if "cost" in legacy_lower or "price" in legacy_lower or "$" in legacy_column:
            return "remove_currency_symbols"
        
        # Percentage fields  
        if "%" in legacy_column or "percent" in legacy_lower:
            return "convert_percentage_to_decimal"
        
        # Date fields
        if "date" in legacy_lower or "time" in legacy_lower:
            return "standardize_date_format"
        
        # Boolean fields
        if "flag" in legacy_lower or "active" in legacy_lower or "enabled" in legacy_lower:
            return "convert_to_boolean"
        
        return None
    
    def _infer_data_type(self, legacy_column: str, standard_name: str) -> str:
        """Infer data type from column name and standard mapping"""
        legacy_lower = legacy_column.lower()
        
        if "qty" in legacy_lower or "quantity" in legacy_lower or "count" in legacy_lower:
            return "INTEGER"
        elif "cost" in legacy_lower or "price" in legacy_lower or "amount" in legacy_lower:
            return "DECIMAL"
        elif "date" in legacy_lower or "time" in legacy_lower:
            return "DATETIME"
        elif "flag" in legacy_lower or "active" in legacy_lower:
            return "BOOLEAN"
        elif "id" in legacy_lower:
            return "STRING"
        else:
            return "STRING"
    
    def _generate_validation_rules(self, standard_name: str) -> List[str]:
        """Generate validation rules for standard column"""
        rules = []
        
        if "qty" in standard_name or "quantity" in standard_name:
            rules.append("value >= 0")
        
        if "cost" in standard_name or "price" in standard_name:
            rules.append("value >= 0")
            rules.append("currency_format_valid")
        
        if "date" in standard_name:
            rules.append("valid_date_format")
            rules.append("date_not_future") 
        
        if "id" in standard_name:
            rules.append("not_null")
            rules.append("unique_values")
        
        return rules
    
    async def _assess_data_quality(self, connection_params: Dict[str, Any], table_name: str) -> DataQualityReport:
        """Assess data quality for a table"""
        try:
            # Load sample data
            sample_data = await self._load_sample_data(connection_params, table_name)
            
            if sample_data.empty:
                return DataQualityReport(
                    table_name=table_name,
                    total_rows=0,
                    completeness_score=0.0,
                    consistency_score=0.0,
                    accuracy_score=0.0,
                    quality_level=DataQualityLevel.POOR,
                    issues_found=["No data found"],
                    recommendations=["Verify data source"]
                )
            
            total_rows = len(sample_data)
            issues = []
            recommendations = []
            
            # Completeness: percentage of non-null values
            completeness_score = 1.0 - (sample_data.isnull().sum().sum() / (sample_data.shape[0] * sample_data.shape[1]))
            
            # Consistency: check for data type consistency
            consistency_score = self._calculate_consistency_score(sample_data, issues)
            
            # Accuracy: check for data value accuracy
            accuracy_score = self._calculate_accuracy_score(sample_data, issues)
            
            # Generate recommendations
            if completeness_score < 0.9:
                recommendations.append("Address missing data issues")
            if consistency_score < 0.8:
                recommendations.append("Standardize data formats")
            if accuracy_score < 0.8:
                recommendations.append("Validate data accuracy")
            
            # Determine quality level
            overall_score = (completeness_score + consistency_score + accuracy_score) / 3
            if overall_score >= 0.95:
                quality_level = DataQualityLevel.EXCELLENT
            elif overall_score >= 0.85:
                quality_level = DataQualityLevel.GOOD
            elif overall_score >= 0.7:
                quality_level = DataQualityLevel.FAIR
            else:
                quality_level = DataQualityLevel.POOR
            
            return DataQualityReport(
                table_name=table_name,
                total_rows=total_rows,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                quality_level=quality_level,
                issues_found=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed for {table_name}: {str(e)}")
            return DataQualityReport(
                table_name=table_name,
                total_rows=0,
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                quality_level=DataQualityLevel.POOR,
                issues_found=[f"Assessment failed: {str(e)}"],
                recommendations=["Review data source connectivity"]
            )
    
    async def _load_sample_data(self, connection_params: Dict[str, Any], table_name: str, sample_size: int = 1000) -> pd.DataFrame:
        """Load sample data from legacy system"""
        system_type = self._detect_system_type(connection_params)
        
        if system_type == LegacySystemType.EXCEL_FILES:
            file_path = connection_params["file_path"]
            return pd.read_excel(file_path, sheet_name=table_name, nrows=sample_size)
        
        elif system_type == LegacySystemType.CSV_FILES:
            file_path = connection_params["file_path"]
            return pd.read_csv(file_path, nrows=sample_size)
        
        else:
            # Would connect to database and sample data
            return pd.DataFrame()  # Placeholder
    
    def _calculate_consistency_score(self, data: pd.DataFrame, issues: List[str]) -> float:
        """Calculate data consistency score"""
        consistency_issues = 0
        total_checks = 0
        
        for column in data.columns:
            total_checks += 1
            
            # Check for mixed data types
            if data[column].dtype == 'object':
                # Try to identify if column should be numeric
                non_null_values = data[column].dropna()
                if len(non_null_values) > 0:
                    numeric_count = 0
                    for value in non_null_values:
                        try:
                            float(str(value).replace('$', '').replace(',', ''))
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                    
                    # If more than 80% are numeric, flag inconsistency
                    if numeric_count / len(non_null_values) > 0.8 and numeric_count < len(non_null_values):
                        consistency_issues += 1
                        issues.append(f"Column '{column}' has mixed numeric/text data")
        
        return 1.0 - (consistency_issues / max(total_checks, 1))
    
    def _calculate_accuracy_score(self, data: pd.DataFrame, issues: List[str]) -> float:
        """Calculate data accuracy score"""
        accuracy_issues = 0
        total_checks = 0
        
        for column in data.columns:
            total_checks += 1
            
            # Check for obvious data issues
            if 'date' in column.lower():
                # Check for future dates in historical data
                try:
                    date_series = pd.to_datetime(data[column], errors='coerce')
                    future_dates = date_series > datetime.now()
                    if future_dates.sum() > 0:
                        accuracy_issues += 1
                        issues.append(f"Column '{column}' contains {future_dates.sum()} future dates")
                except:
                    pass
            
            if 'qty' in column.lower() or 'quantity' in column.lower():
                # Check for negative quantities
                try:
                    numeric_data = pd.to_numeric(data[column], errors='coerce')
                    negative_values = numeric_data < 0
                    if negative_values.sum() > 0:
                        accuracy_issues += 1
                        issues.append(f"Column '{column}' contains {negative_values.sum()} negative values")
                except:
                    pass
        
        return 1.0 - (accuracy_issues / max(total_checks, 1))
    
    def _calculate_mapping_confidence(self, column_mappings: Dict[str, List[ColumnMapping]]) -> float:
        """Calculate overall mapping confidence score"""
        if not column_mappings:
            return 0.0
        
        total_confidence = 0.0
        total_mappings = 0
        
        for table_mappings in column_mappings.values():
            for mapping in table_mappings:
                total_confidence += mapping.confidence_score
                total_mappings += 1
        
        return total_confidence / max(total_mappings, 1)
    
    def _estimate_migration_complexity(
        self,
        column_mappings: Dict[str, List[ColumnMapping]],
        data_quality: Dict[str, DataQualityReport]
    ) -> Tuple[str, int]:
        """Estimate migration complexity and time requirements"""
        
        # Count mappings and quality issues
        total_mappings = sum(len(mappings) for mappings in column_mappings.values())
        low_confidence_mappings = sum(
            1 for table_mappings in column_mappings.values()
            for mapping in table_mappings 
            if mapping.confidence_score < 0.7
        )
        
        poor_quality_tables = sum(
            1 for report in data_quality.values() 
            if report.quality_level in [DataQualityLevel.POOR, DataQualityLevel.FAIR]
        )
        
        # Determine complexity
        if low_confidence_mappings / max(total_mappings, 1) > 0.3 or poor_quality_tables > len(data_quality) / 2:
            complexity = "HIGH"
            base_hours = 80
        elif low_confidence_mappings / max(total_mappings, 1) > 0.1 or poor_quality_tables > 0:
            complexity = "MEDIUM" 
            base_hours = 40
        else:
            complexity = "LOW"
            base_hours = 20
        
        # Adjust for number of tables and columns
        estimated_hours = base_hours + (len(column_mappings) * 5) + (total_mappings * 0.5)
        
        return complexity, int(estimated_hours)


class IntelligentDataMapper:
    """
    Intelligent data mapping and transformation engine
    Handles the complex transformations needed for various legacy formats
    """
    
    def __init__(self):
        self.logger = logging.getLogger("IntelligentDataMapper")
        self.transformation_cache = {}
    
    async def transform_data(
        self,
        source_data: pd.DataFrame,
        column_mappings: List[ColumnMapping]
    ) -> pd.DataFrame:
        """Transform legacy data using intelligent mappings"""
        
        try:
            transformed_data = source_data.copy()
            
            for mapping in column_mappings:
                if mapping.legacy_name not in source_data.columns:
                    self.logger.warning(f"Column '{mapping.legacy_name}' not found in source data")
                    continue
                
                # Apply transformation rule if specified
                if mapping.transformation_rule:
                    transformed_data[mapping.standard_name] = await self._apply_transformation(
                        transformed_data[mapping.legacy_name],
                        mapping.transformation_rule
                    )
                else:
                    # Simple rename
                    transformed_data[mapping.standard_name] = transformed_data[mapping.legacy_name]
                
                # Apply validation rules
                if mapping.validation_rules:
                    transformed_data = await self._apply_validation_rules(
                        transformed_data,
                        mapping.standard_name,
                        mapping.validation_rules
                    )
                
                # Remove original column if renamed
                if mapping.legacy_name != mapping.standard_name:
                    transformed_data = transformed_data.drop(columns=[mapping.legacy_name])
            
            self.logger.info(f"Data transformation completed: {len(column_mappings)} mappings applied")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            raise
    
    async def _apply_transformation(self, data_series: pd.Series, rule: str) -> pd.Series:
        """Apply specific transformation rule to data series"""
        
        if rule == "remove_currency_symbols":
            return data_series.astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
        
        elif rule == "convert_percentage_to_decimal":
            return data_series.astype(str).str.replace('%', '').astype(float) / 100
        
        elif rule == "standardize_date_format":
            return pd.to_datetime(data_series, errors='coerce')
        
        elif rule == "convert_to_boolean":
            return data_series.astype(str).str.lower().map({
                'true': True, 'yes': True, 'y': True, '1': True, 'active': True,
                'false': False, 'no': False, 'n': False, '0': False, 'inactive': False
            }).fillna(False)
        
        else:
            self.logger.warning(f"Unknown transformation rule: {rule}")
            return data_series
    
    async def _apply_validation_rules(
        self,
        data: pd.DataFrame,
        column: str,
        rules: List[str]
    ) -> pd.DataFrame:
        """Apply validation rules and handle violations"""
        
        for rule in rules:
            if rule == "value >= 0":
                # Set negative values to 0 with warning
                negative_mask = data[column] < 0
                if negative_mask.any():
                    self.logger.warning(f"Found {negative_mask.sum()} negative values in {column}, setting to 0")
                    data.loc[negative_mask, column] = 0
            
            elif rule == "not_null":
                # Fill null values with appropriate defaults
                null_mask = data[column].isnull()
                if null_mask.any():
                    self.logger.warning(f"Found {null_mask.sum()} null values in {column}")
                    if data[column].dtype in ['int64', 'float64']:
                        data.loc[null_mask, column] = 0
                    else:
                        data.loc[null_mask, column] = ""
        
        return data


class LegacySystemConnector:
    """
    Main connector class that orchestrates legacy system integration
    This is the primary interface for framework users
    """
    
    def __init__(self):
        self.logger = logging.getLogger("LegacySystemConnector")
        self.schema_analyzer = AutoSchemaAnalyzer()
        self.data_mapper = IntelligentDataMapper()
        self.connection_cache = {}
    
    async def analyze_legacy_system(self, connection_params: Dict[str, Any]) -> SchemaAnalysisResult:
        """
        Analyze legacy system and generate integration recommendations
        This handles the 300+ column variations mentioned in AI.MD
        """
        self.logger.info("Starting legacy system analysis")
        
        try:
            # Validate connection parameters
            if not self._validate_connection_params(connection_params):
                raise ValueError("Invalid connection parameters")
            
            # Perform schema analysis
            analysis_result = await self.schema_analyzer.analyze_schema(connection_params)
            
            # Cache connection for later use
            connection_id = self._generate_connection_id(connection_params)
            self.connection_cache[connection_id] = {
                "params": connection_params,
                "analysis": analysis_result,
                "analyzed_at": datetime.now()
            }
            
            self.logger.info(f"Legacy system analysis completed: {analysis_result.confidence_score:.2f} confidence")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Legacy system analysis failed: {str(e)}")
            raise
    
    async def migrate_data(
        self,
        connection_params: Dict[str, Any],
        column_mappings: Dict[str, List[ColumnMapping]],
        target_tables: List[str] = None
    ) -> Dict[str, Any]:
        """Migrate data from legacy system using intelligent mappings"""
        
        self.logger.info("Starting data migration")
        migration_results = {
            "started_at": datetime.now().isoformat(),
            "tables_migrated": {},
            "total_rows_migrated": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Migrate each table
            for table_name, mappings in column_mappings.items():
                if target_tables and table_name not in target_tables:
                    continue
                
                self.logger.info(f"Migrating table: {table_name}")
                
                try:
                    # Load source data
                    source_data = await self.schema_analyzer._load_sample_data(
                        connection_params, table_name, sample_size=None  # Load all data
                    )
                    
                    if source_data.empty:
                        migration_results["warnings"].append(f"No data found in table {table_name}")
                        continue
                    
                    # Transform data
                    transformed_data = await self.data_mapper.transform_data(source_data, mappings)
                    
                    # Store migration results
                    migration_results["tables_migrated"][table_name] = {
                        "source_rows": len(source_data),
                        "transformed_rows": len(transformed_data),
                        "columns_mapped": len(mappings),
                        "data_sample": transformed_data.head().to_dict('records') if len(transformed_data) > 0 else []
                    }
                    
                    migration_results["total_rows_migrated"] += len(transformed_data)
                    
                except Exception as table_error:
                    error_msg = f"Failed to migrate table {table_name}: {str(table_error)}"
                    migration_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            migration_results["completed_at"] = datetime.now().isoformat()
            self.logger.info(f"Data migration completed: {migration_results['total_rows_migrated']} total rows migrated")
            
            return migration_results
            
        except Exception as e:
            self.logger.error(f"Data migration failed: {str(e)}")
            migration_results["fatal_error"] = str(e)
            return migration_results
    
    def _validate_connection_params(self, params: Dict[str, Any]) -> bool:
        """Validate connection parameters"""
        if not params:
            return False
        
        # Check for required parameters based on connection type
        if "file_path" in params:
            return Path(params["file_path"]).exists()
        elif "database_type" in params:
            return "connection_string" in params or "host" in params
        elif "api_url" in params:
            return True
        
        return False
    
    def _generate_connection_id(self, params: Dict[str, Any]) -> str:
        """Generate unique connection ID"""
        import hashlib
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    async def get_migration_preview(
        self,
        connection_params: Dict[str, Any],
        table_name: str,
        mappings: List[ColumnMapping],
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Get preview of data migration results"""
        
        try:
            # Load sample data
            source_data = await self.schema_analyzer._load_sample_data(
                connection_params, table_name, sample_size
            )
            
            # Transform sample
            transformed_data = await self.data_mapper.transform_data(source_data, mappings)
            
            return {
                "source_sample": source_data.head(10).to_dict('records'),
                "transformed_sample": transformed_data.head(10).to_dict('records'),
                "mapping_summary": [mapping.to_dict() for mapping in mappings],
                "row_count": len(source_data)
            }
            
        except Exception as e:
            self.logger.error(f"Migration preview failed: {str(e)}")
            return {"error": str(e)}


# Export key components
__all__ = [
    "LegacySystemType",
    "DataQualityLevel",
    "ColumnMapping",
    "DataQualityReport", 
    "SchemaAnalysisResult",
    "AutoSchemaAnalyzer",
    "IntelligentDataMapper",
    "LegacySystemConnector"
]