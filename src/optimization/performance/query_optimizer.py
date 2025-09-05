"""
QueryOptimizer - Optimize database queries for performance
Reduces query time and memory usage through intelligent optimization
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
import hashlib
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Optimize database queries for maximum performance"""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.query_cache = {}
        self.query_stats = {}
        self.index_suggestions = []
        
    def optimize_yarn_query(
        self, 
        conditions: Dict[str, Any] = None,
        columns: List[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Optimize yarn inventory queries
        
        BEFORE:
        SELECT * FROM yarn_inventory WHERE status = 'active'
        
        AFTER:
        SELECT specific columns with indexes and limits
        """
        # Define essential columns if not specified
        if columns is None:
            columns = [
                'yarn_id', 'description', 'theoretical_balance',
                'allocated', 'on_order', 'min_stock', 'lead_time',
                'last_updated'
            ]
        
        # Build optimized query
        query = f"SELECT {', '.join(columns)} FROM yarn_inventory"
        
        # Add WHERE clause with optimized conditions
        where_clauses = []
        params = []
        
        if conditions:
            for key, value in conditions.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        placeholders = ','.join(['%s'] * len(value))
                        where_clauses.append(f"{key} IN ({placeholders})")
                        params.extend(value)
                    else:
                        where_clauses.append(f"{key} = %s")
                        params.append(value)
        
        # Always filter inactive records
        where_clauses.append("status = 'active'")
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add index hints for better performance
        query += " USE INDEX (idx_yarn_status)"
        
        # Add sorting for consistent results
        query += " ORDER BY yarn_id"
        
        # Add limit to prevent memory issues
        if limit:
            query += f" LIMIT {limit}"
        elif not conditions:  # Default limit for broad queries
            query += " LIMIT 10000"
        
        return query, params
    
    def optimize_bom_query(
        self,
        style_ids: List[str] = None,
        yarn_ids: List[str] = None
    ) -> Tuple[str, List]:
        """
        Optimize BOM queries with proper indexing
        """
        columns = ['style_id', 'yarn_id', 'quantity_per', 'percentage', 'active']
        
        query = f"SELECT {', '.join(columns)} FROM bom"
        where_clauses = []
        params = []
        
        if style_ids:
            placeholders = ','.join(['%s'] * len(style_ids))
            where_clauses.append(f"style_id IN ({placeholders})")
            params.extend(style_ids)
        
        if yarn_ids:
            placeholders = ','.join(['%s'] * len(yarn_ids))
            where_clauses.append(f"yarn_id IN ({placeholders})")
            params.extend(yarn_ids)
        
        where_clauses.append("active = 1")
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Use composite index for style and yarn
        query += " USE INDEX (idx_bom_style_yarn)"
        query += " ORDER BY style_id, yarn_id"
        
        return query, params
    
    def optimize_production_order_query(
        self,
        status: str = None,
        date_range: Tuple[datetime, datetime] = None
    ) -> Tuple[str, List]:
        """
        Optimize production order queries
        """
        columns = [
            'order_id', 'style_id', 'quantity', 'status',
            'scheduled_date', 'completion_date', 'priority',
            'work_center_id', 'machine_id'
        ]
        
        query = f"SELECT {', '.join(columns)} FROM production_orders"
        where_clauses = []
        params = []
        
        if status:
            where_clauses.append("status = %s")
            params.append(status)
        
        if date_range:
            where_clauses.append("scheduled_date BETWEEN %s AND %s")
            params.extend(date_range)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Use appropriate index based on conditions
        if status and date_range:
            query += " USE INDEX (idx_order_status_date)"
        elif status:
            query += " USE INDEX (idx_order_status)"
        elif date_range:
            query += " USE INDEX (idx_order_date)"
        
        query += " ORDER BY priority DESC, scheduled_date ASC"
        
        return query, params
    
    def batch_fetch(
        self,
        table: str,
        ids: List[str],
        id_column: str = 'id',
        batch_size: int = 1000,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch records in optimized batches to prevent memory issues
        """
        results = []
        
        # Select specific columns or all
        select_clause = ', '.join(columns) if columns else '*'
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch_ids))
            
            query = f"""
            SELECT {select_clause} FROM {table} 
            WHERE {id_column} IN ({placeholders})
            """
            
            if self.db:
                batch_result = self.db.fetch_df(query, batch_ids)
                results.append(batch_result)
            
            logger.info(f"Fetched batch {i//batch_size + 1} of {len(ids)//batch_size + 1}")
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def create_missing_indexes(self) -> List[str]:
        """
        Create missing database indexes for performance
        """
        indexes = [
            # Yarn inventory indexes
            "CREATE INDEX IF NOT EXISTS idx_yarn_status ON yarn_inventory(status)",
            "CREATE INDEX IF NOT EXISTS idx_yarn_planning ON yarn_inventory(planning_balance)",
            "CREATE INDEX IF NOT EXISTS idx_yarn_shortage ON yarn_inventory(planning_balance, min_stock)",
            
            # BOM indexes
            "CREATE INDEX IF NOT EXISTS idx_bom_style ON bom(style_id)",
            "CREATE INDEX IF NOT EXISTS idx_bom_yarn ON bom(yarn_id)",
            "CREATE INDEX IF NOT EXISTS idx_bom_style_yarn ON bom(style_id, yarn_id)",
            
            # Production order indexes
            "CREATE INDEX IF NOT EXISTS idx_order_status ON production_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_order_date ON production_orders(scheduled_date)",
            "CREATE INDEX IF NOT EXISTS idx_order_status_date ON production_orders(status, scheduled_date)",
            "CREATE INDEX IF NOT EXISTS idx_order_priority ON production_orders(priority DESC)",
            
            # Work center and machine indexes
            "CREATE INDEX IF NOT EXISTS idx_machine_center ON machines(work_center_id)",
            "CREATE INDEX IF NOT EXISTS idx_machine_status ON machines(status)",
            
            # Sales activity indexes
            "CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_activity(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_sales_style ON sales_activity(style_id)",
            "CREATE INDEX IF NOT EXISTS idx_sales_customer ON sales_activity(customer_id)"
        ]
        
        created_indexes = []
        
        if self.db:
            for index_sql in indexes:
                try:
                    self.db.execute(index_sql)
                    created_indexes.append(index_sql)
                    logger.info(f"Created index: {index_sql[:50]}...")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
        
        return created_indexes
    
    @lru_cache(maxsize=128)
    def get_cached_query_result(
        self,
        query_hash: str,
        ttl_seconds: int = 300
    ) -> Optional[pd.DataFrame]:
        """
        Get cached query result if available and not expired
        """
        if query_hash in self.query_cache:
            cached_data, timestamp = self.query_cache[query_hash]
            
            if datetime.now() - timestamp < timedelta(seconds=ttl_seconds):
                logger.info(f"Cache hit for query {query_hash[:8]}")
                return cached_data
            else:
                # Remove expired cache
                del self.query_cache[query_hash]
        
        return None
    
    def cache_query_result(
        self,
        query: str,
        params: List,
        result: pd.DataFrame
    ) -> str:
        """
        Cache query result with hash key
        """
        # Create hash of query and params
        query_data = json.dumps({'query': query, 'params': str(params)})
        query_hash = hashlib.md5(query_data.encode()).hexdigest()
        
        # Store in cache
        self.query_cache[query_hash] = (result, datetime.now())
        
        # Track statistics
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                'count': 0,
                'total_time': 0,
                'avg_rows': 0
            }
        
        self.query_stats[query_hash]['count'] += 1
        self.query_stats[query_hash]['avg_rows'] = len(result)
        
        return query_hash
    
    def optimize_join_query(
        self,
        tables: List[Dict[str, Any]],
        join_conditions: List[str],
        columns: List[str] = None
    ) -> str:
        """
        Optimize complex JOIN queries
        
        Args:
            tables: List of {'name': 'table', 'alias': 't1'}
            join_conditions: List of join conditions
        """
        # Select specific columns to reduce data transfer
        if columns:
            select_clause = ', '.join(columns)
        else:
            # Avoid SELECT * in joins
            select_clause = ', '.join([f"{t['alias']}.*" for t in tables[:1]])
        
        # Start with smallest table for better performance
        query = f"SELECT {select_clause} FROM {tables[0]['name']} {tables[0]['alias']}"
        
        # Add joins with proper indexing hints
        for i, table in enumerate(tables[1:], 1):
            join_type = table.get('join_type', 'INNER')
            query += f"\n{join_type} JOIN {table['name']} {table['alias']}"
            
            if i <= len(join_conditions):
                query += f" ON {join_conditions[i-1]}"
            
            # Add index hints for join columns
            if 'index_hint' in table:
                query += f" USE INDEX ({table['index_hint']})"
        
        return query
    
    def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """
        Analyze and identify slow queries for optimization
        """
        slow_queries = []
        
        for query_hash, stats in self.query_stats.items():
            if stats['count'] > 10:  # Frequently executed
                slow_queries.append({
                    'query_hash': query_hash,
                    'execution_count': stats['count'],
                    'avg_rows': stats['avg_rows'],
                    'optimization_suggestion': self._suggest_optimization(stats)
                })
        
        return sorted(slow_queries, key=lambda x: x['execution_count'], reverse=True)
    
    def _suggest_optimization(self, stats: Dict) -> str:
        """
        Suggest optimization based on query statistics
        """
        suggestions = []
        
        if stats['avg_rows'] > 10000:
            suggestions.append("Consider pagination or limiting results")
        
        if stats['count'] > 100:
            suggestions.append("High frequency - increase cache TTL")
        
        if stats.get('avg_time', 0) > 1000:  # > 1 second
            suggestions.append("Query is slow - review indexes and query structure")
        
        return '; '.join(suggestions) if suggestions else "Query performing well"
    
    def optimize_aggregation_query(
        self,
        table: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        conditions: Dict[str, Any] = None
    ) -> Tuple[str, List]:
        """
        Optimize aggregation queries for performance
        """
        # Build aggregation expressions
        agg_expressions = []
        for column, func in aggregations.items():
            agg_expressions.append(f"{func}({column}) AS {column}_{func.lower()}")
        
        # Build query
        select_clause = ', '.join(group_by + agg_expressions)
        query = f"SELECT {select_clause} FROM {table}"
        
        # Add conditions
        params = []
        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = %s")
                params.append(value)
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add GROUP BY
        query += f" GROUP BY {', '.join(group_by)}"
        
        # Add HAVING clause for aggregation filtering if needed
        # This would be based on additional parameters
        
        # Order by first group column for consistent results
        query += f" ORDER BY {group_by[0]}"
        
        return query, params
    
    def optimize_subquery(
        self,
        main_query: str,
        subquery: str,
        correlation_column: str
    ) -> str:
        """
        Convert correlated subqueries to JOINs for better performance
        """
        # This is a simplified example - real implementation would parse SQL
        optimized = main_query.replace(
            f"WHERE EXISTS ({subquery})",
            f"INNER JOIN ({subquery}) sq ON main.{correlation_column} = sq.{correlation_column}"
        )
        
        return optimized
    
    def get_query_execution_plan(self, query: str) -> Dict[str, Any]:
        """
        Get query execution plan for analysis
        """
        if self.db:
            explain_query = f"EXPLAIN {query}"
            plan = self.db.fetch(explain_query)
            
            return {
                'query': query,
                'execution_plan': plan,
                'estimated_cost': self._estimate_cost(plan)
            }
        
        return {'query': query, 'execution_plan': None, 'estimated_cost': 0}
    
    def _estimate_cost(self, plan: Any) -> float:
        """
        Estimate query cost from execution plan
        """
        # This would parse the execution plan to estimate cost
        # Simplified for demonstration
        return 0.0
    
    def clear_cache(self):
        """
        Clear query cache
        """
        self.query_cache.clear()
        logger.info("Query cache cleared")