
# Vectorized DataFrame Operations Helper Functions

def vectorize_bom_processing(bom_data):
    """Process BOM data using vectorized operations."""
    if bom_data.empty:
        return {}, set()
    
    # Ensure columns exist
    yarn_col = 'Yarn_ID' if 'Yarn_ID' in bom_data.columns else 'Component_ID'
    qty_col = 'Quantity' if 'Quantity' in bom_data.columns else 'Usage'
    prod_col = 'Product_ID' if 'Product_ID' in bom_data.columns else 'Style'
    
    # Vectorized aggregation
    yarn_requirements = {}
    unique_yarns = set()
    
    # Group by yarn and aggregate
    grouped = bom_data.groupby(yarn_col).agg({
        qty_col: 'sum',
        prod_col: lambda x: list(set(x))
    })
    
    for yarn_id, row in grouped.iterrows():
        yarn_id_str = str(yarn_id)
        unique_yarns.add(yarn_id_str)
        yarn_requirements[yarn_id_str] = {
            'total_required': float(row[qty_col]),
            'products_using': row[prod_col],
            'average_usage': float(row[qty_col]) / len(row[prod_col]) if row[prod_col] else 0
        }
    
    return yarn_requirements, unique_yarns

def vectorize_inventory_conversion(inventory_df):
    """Convert inventory DataFrame to dictionary using vectorized operations."""
    if inventory_df.empty:
        return {}
    
    # Identify columns
    id_col = 'Yarn ID' if 'Yarn ID' in inventory_df.columns else 'ID'
    balance_col = 'Balance' if 'Balance' in inventory_df.columns else 'Quantity'
    
    # Vectorized conversion
    inventory_df['yarn_id_str'] = inventory_df[id_col].astype(str)
    inventory_df['balance_float'] = inventory_df[balance_col].astype(float)
    
    return dict(zip(inventory_df['yarn_id_str'], inventory_df['balance_float']))

def vectorize_alert_creation(df, alert_type, severity='High'):
    """Create alerts from DataFrame using vectorized operations."""
    if df.empty:
        return []
    
    desc_col = 'Description' if 'Description' in df.columns else 'Item'
    
    alerts = df.apply(lambda row: {
        'type': alert_type,
        'severity': severity,
        'item': str(row.get(desc_col, 'Unknown'))[:50],
        'value': row.get('Planning Balance', row.get('Balance', 0)),
        'threshold': row.get('Min_Stock', 0)
    }, axis=1).tolist()
    
    return alerts

def vectorize_shortage_detection(df, threshold_col='Min_Stock', balance_col='Planning Balance'):
    """Detect shortages using vectorized operations."""
    if df.empty or threshold_col not in df.columns or balance_col not in df.columns:
        return pd.DataFrame()
    
    # Vectorized comparison
    shortage_mask = df[balance_col] < df[threshold_col]
    return df[shortage_mask].copy()

def vectorize_yarn_lookup(yarn_data, material_id, value_col='Planning Balance'):
    """Lookup yarn value using vectorized operations."""
    if yarn_data.empty:
        return 0
    
    desc_col = 'Description' if 'Description' in yarn_data.columns else 'Yarn ID'
    
    # Vectorized lookup
    mask = yarn_data[desc_col].astype(str) == str(material_id)
    matched = yarn_data.loc[mask, value_col]
    
    return float(matched.iloc[0]) if not matched.empty else 0

def vectorize_production_summary(production_df, group_by='Style'):
    """Create production summary using vectorized operations."""
    if production_df.empty or group_by not in production_df.columns:
        return {}
    
    # Vectorized grouping and aggregation
    summary = production_df.groupby(group_by).agg({
        'Quantity': 'sum',
        'Order ID': 'count'
    }).to_dict('index')
    
    # Convert to desired format
    result = {}
    for style, metrics in summary.items():
        result[style] = {
            'total_quantity': float(metrics.get('Quantity', 0)),
            'order_count': int(metrics.get('Order ID', 0))
        }
    
    return result
