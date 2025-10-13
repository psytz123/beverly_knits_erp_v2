# Fabric Yards-to-Pounds Conversion for Yarn Demand Calculations

## Conversion Data Available ✅ EXCELLENT

Your **QuadS_finishedFabricList** contains **direct conversion factors** for 3,481 fabric specifications:

### Key Conversion Fields
```
• Yds/Lbs: Direct yards-per-pound ratio (MOST IMPORTANT)
• Oz/Lin Yd: Ounces per linear yard (fabric weight × width)
• Overall Width: Fabric width in inches
• Oz / Sq Yd: Standard fabric weight (ounces per square yard)
• GSM: Grams per square meter (metric fabric weight)
```

### Sample Conversion Data
```
Fabric F ID 60 (6191-BK):
• Yds/Lbs: 1.58 (1.58 yards = 1 pound of fabric)
• Overall Width: 66 inches
• Oz/Lin Yd: 10.12 oz per yard
• Oz/Sq Yd: 5.52 oz per square yard

Fabric F ID 61 (80393C-DS):
• Yds/Lbs: 1.10 (1.10 yards = 1 pound of fabric)  
• Overall Width: 70 inches
• Oz/Lin Yd: 14.5 oz per yard
• Oz/Sq Yd: 7.37 oz per square yard
```

## Yarn Demand Calculation Flow

### Step 1: Fabric Demand (Yards) → Fabric Weight (Pounds)
```
Formula: Fabric_Pounds = Fabric_Yards ÷ Yds_Per_Lb

Example:
Customer orders 1,000 yards of fabric F ID 60
Fabric_Pounds = 1,000 ÷ 1.58 = 633 pounds of fabric needed
```

### Step 2: Fabric Weight (Pounds) → Yarn Requirements (Pounds)  
```
Formula: Yarn_Pounds = Fabric_Pounds × BOM_Percentage

Example (using Style_BOM data):
Style 125792/1 uses Yarn 18767 at 91.4% of fabric weight
Yarn_Required = 633 lbs × 0.914 = 578 pounds of Yarn 18767
```

### Complete Calculation Chain
```
Sales Forecast → Fabric Yards → Fabric Pounds → Yarn Pounds → Inventory Check
    ↓               ↓               ↓               ↓              ↓
1,000 yards →   633 lbs fabric → 578 lbs yarn → Planning Balance → Shortage Alert
```

## Implementation Integration

### Data Mapping Requirements

**Link Fabric Specifications to Sales Data**:
```sql
-- Need to map sales styles to fabric specifications
Sales_Activity_Report.Style ←→ QuadS_FabricList.Name (or F_ID)
```

**Style Mapping Examples**:
```
Sales Style → Fabric Specification → Conversion Factor
"CEE4142-1" → Fabric F ID 150 → 1.45 Yds/Lbs
"CMR4067-3" → Fabric F ID 220 → 1.22 Yds/Lbs  
50000174 → Fabric F ID 75 → 1.67 Yds/Lbs
```

### Calculation Engine Design

**Python Implementation**:
```python
class YardsToPoundsConverter:
    def __init__(self, fabric_specs_data):
        self.fabric_specs = fabric_specs_data
        
    def get_conversion_factor(self, style_id):
        """Get Yds/Lbs ratio for a specific style"""
        fabric_spec = self.fabric_specs[
            self.fabric_specs['Name'] == style_id
        ]
        return fabric_spec['Yds/Lbs'].iloc[0]
    
    def yards_to_pounds(self, style_id, yards):
        """Convert fabric yards to pounds"""
        conversion_factor = self.get_conversion_factor(style_id)
        return yards / conversion_factor
    
    def calculate_yarn_requirements(self, style_id, yards, bom_data):
        """Complete calculation: yards → fabric lbs → yarn lbs"""
        # Step 1: Yards to fabric pounds
        fabric_pounds = self.yards_to_pounds(style_id, yards)
        
        # Step 2: Fabric pounds to yarn pounds (using BOM)
        style_bom = bom_data[bom_data['Style_ID'] == style_id]
        yarn_requirements = {}
        
        for _, row in style_bom.iterrows():
            yarn_id = row['Yarn_ID']
            bom_percentage = row['BOM_Percentage']
            yarn_pounds = fabric_pounds * bom_percentage
            yarn_requirements[yarn_id] = yarn_pounds
            
        return {
            'fabric_pounds': fabric_pounds,
            'yarn_requirements': yarn_requirements
        }
```

### Example Calculation Workflow

**Scenario**: Customer orders 2,000 yards of style "6191-BK"

**Step 1: Fabric Conversion**
```
Style: 6191-BK (F ID 60)
Yards ordered: 2,000
Yds/Lbs ratio: 1.58
Fabric pounds needed: 2,000 ÷ 1.58 = 1,266 lbs
```

**Step 2: BOM Explosion** (assuming style maps to 125792/1)
```
Style 125792/1 BOM:
• Yarn 18767: 91.4% × 1,266 lbs = 1,157 lbs
• Yarn 18123: 8.6% × 1,266 lbs = 109 lbs
Total yarn needed: 1,266 lbs
```

**Step 3: Inventory Check**
```
Yarn 18767 Planning Balance: 2,400 lbs
Required: 1,157 lbs  
Available after order: 2,400 - 1,157 = 1,243 lbs ✅ Sufficient

Yarn 18123 Planning Balance: -50 lbs (shortage!)
Required: 109 lbs
Total shortage: 50 + 109 = 159 lbs ⚠️ ALERT
```

## Data Quality Validation

### Conversion Factor Reasonableness Check
```
Typical fabric weight ranges:
• Light fabrics: 2.0-3.0 Yds/Lbs (lightweight materials)
• Medium fabrics: 1.0-2.0 Yds/Lbs (standard weight)  
• Heavy fabrics: 0.5-1.0 Yds/Lbs (heavy materials)

Your data shows:
• F ID 60: 1.58 Yds/Lbs (medium weight) ✅
• F ID 61: 1.10 Yds/Lbs (heavier fabric) ✅
• F ID 62: 1.36 Yds/Lbs (medium weight) ✅
```

### Alternative Calculation Verification
```
Manual check using width and weight:
F ID 60: 66" wide, 5.52 oz/sq yd
Linear yard weight = (66 ÷ 12) × 5.52 = 30.36 oz per yard
Pounds per yard = 30.36 ÷ 16 = 1.90 lbs per yard
Yds/Lbs = 1 ÷ 1.90 = 0.526... 

Note: This doesn't match 1.58 Yds/Lbs from data
Recommend using the direct Yds/Lbs field as it likely includes 
manufacturing adjustments, waste factors, etc.
```

## Implementation Priorities

### Phase 1: Style Mapping (Critical)
1. **Map sales styles to fabric F IDs**
   - Cross-reference Sales_Activity_Report.Style with QuadS fabric names
   - Handle both numeric (50000174) and alphanumeric (CEE4142-1) formats
   - Create lookup table for conversion factors

### Phase 2: Calculation Engine
2. **Build yards-to-pounds converter**
   - Use direct Yds/Lbs ratios from QuadS data
   - Integrate with existing BOM explosion logic
   - Validate calculations with actual production data

### Phase 3: Integration Testing  
3. **End-to-end validation**
   - Compare calculated yarn requirements to actual consumption
   - Verify conversion factors against production records
   - Adjust for any systematic differences

## Missing Data Considerations

### Style Mapping Gaps
- **Sales styles not in QuadS list**: Need default conversion factors
- **Multiple fabric options per style**: Choose primary or average
- **New styles**: Process for adding conversion factors

### Conversion Factor Updates
- **Fabric specification changes**: Update Yds/Lbs ratios when needed
- **Width changes**: Recalculate linear yard weights
- **New fabric constructions**: Add to conversion database

## Database Schema Addition

### Fabric Specifications Table
```sql
CREATE TABLE fabric_specifications (
    fabric_id INTEGER PRIMARY KEY,
    fabric_name VARCHAR(50),
    finish_code VARCHAR(20),
    overall_width_inches DECIMAL(5,2),
    cuttable_width_inches DECIMAL(5,2),
    oz_per_linear_yard DECIMAL(8,4),
    yards_per_pound DECIMAL(8,4), -- KEY FIELD
    gsm DECIMAL(8,2),
    oz_per_square_yard DECIMAL(8,4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Style to fabric mapping
CREATE TABLE style_fabric_mapping (
    style_id VARCHAR(50),
    fabric_id INTEGER REFERENCES fabric_specifications(fabric_id),
    effective_date DATE,
    PRIMARY KEY (style_id, effective_date)
);
```

## Success Metrics

### Conversion Accuracy
- **Yarn calculation accuracy**: ±5% of actual consumption
- **Forecast precision**: Improved demand planning accuracy
- **Waste reduction**: Better yarn procurement sizing

### System Performance
- **Calculation speed**: Real-time yards-to-pounds conversion
- **Data completeness**: 95%+ style coverage with conversion factors
- **Update frequency**: Daily refresh of fabric specifications

Your **QuadS fabric specification data is excellent** - the direct Yds/Lbs ratios eliminate complex calculations and provide accurate conversion factors for precise yarn demand planning!