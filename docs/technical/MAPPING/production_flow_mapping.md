# Production Flow Mapping & Data Tracking

## Production Stage Flow
```
YARN INVENTORY → G00 (Knit) → G02 (Finishing) → I01 (Inspection) → F01 (Available) → P01 (Allocated)
```

## Detailed Stage Mapping

### Stage 1: G00 - Fabric Knitting (Raw Knitted Fabric)
**Process**: Raw yarn converted to knitted fabric
**Inventory Status**: Work-in-process (WIP)
**Yarn Consumption**: **HAPPENS HERE** - yarn physically consumed

**Data Tracking Requirements**:
```
Input:  Raw Yarn (from Yarn Inventory)
Output: Knitted fabric (gray goods)
Status: G00 inventory location

Key Metrics:
• Yarn consumed (by Yarn_ID and quantity)
• Fabric produced (by Style_ID and yards)
• Conversion efficiency (yarn lbs → fabric yards)
• Production date and batch tracking
```

**Critical for Planning**:
- **Yarn allocation point**: When yarn moves from inventory to G00
- **BOM consumption**: Actual yarn usage vs planned BOM percentages
- **Production scheduling**: Which styles are being knitted when

### Stage 2: G02 - Fabric Finishing 
**Process**: Knitted fabric undergoes dyeing, finishing processes
**Inventory Status**: Work-in-process (WIP)
**Yarn Consumption**: None (yarn already consumed at G00)

**Data Tracking Requirements**:
```
Input:  Gray fabric from G00
Output: Finished fabric (dyed/processed)
Status: G02 inventory location

Key Metrics:
• Fabric in finishing process
• Processing time (lead time tracking)
• Quality issues/rework needs
• Chemical/dye inventory consumption
```

**Critical for Planning**:
- **Processing capacity**: Bottleneck identification
- **Lead time tracking**: G00 → G02 cycle time
- **Quality yield**: Fabric loss during finishing

### Stage 3: I01 - Final Inspection
**Process**: Quality inspection and approval
**Inventory Status**: Work-in-process (awaiting QC approval)
**Yarn Consumption**: None

**Data Tracking Requirements**:
```
Input:  Finished fabric from G02
Output: Approved fabric OR rework/reject
Status: I01 inventory location

Key Metrics:
• Fabric awaiting inspection
• Pass/fail rates by style
• Inspection lead time
• Rework requirements
```

**Critical for Planning**:
- **Quality gates**: Fabric that fails inspection
- **Release timing**: When fabric becomes available for sale
- **Yield rates**: Final fabric output vs yarn input

### Stage 4: F01 - Available Fabric (Highest $ Value)
**Process**: Approved, saleable finished fabric
**Inventory Status**: **FINISHED GOODS** - ready to ship
**Yarn Consumption**: Complete (all yarn costs captured)

**Data Tracking Requirements**:
```
Input:  Approved fabric from I01
Output: Fabric available for customer orders
Status: F01 inventory location (PRIMARY FOCUS)

Key Metrics:
• Available fabric by Style_ID
• Fabric aging (how long in F01)
• Customer allocation decisions
• Inventory turns
```

**Critical for Planning**:
- **Safety stock calculations**: 20-day buffer at F01 level
- **Customer promising**: Available to promise (ATP)
- **Sales order fulfillment**: Matching orders to F01 inventory

### Stage 5: P01 - Allocated Fabric
**Process**: Fabric selected and allocated to specific customer orders
**Inventory Status**: **COMMITTED** - reserved for shipment
**Yarn Consumption**: Complete

**Data Tracking Requirements**:
```
Input:  Available fabric from F01
Output: Fabric staged for customer shipment
Status: P01 inventory location

Key Metrics:
• Fabric allocated by customer/order
• Ship date commitments
• Pick/pack efficiency
• Shipping lead times
```

**Critical for Planning**:
- **Order fulfillment tracking**: Customer delivery performance
- **Allocation logic**: First-in-first-out vs customer priority
- **F01 availability**: Reduces available fabric for new orders

## Yarn Consumption & Planning Points

### Yarn Planning Balance Impact by Stage

**At G00 Entry** (Yarn Consumption Point):
```
When fabric production starts at G00:
• Yarn Inventory: Reduces by BOM requirements
• Yarn Planning_Balance: Reduces by allocated amount
• WIP Tracking: Begins for fabric production
```

**Planning Formula**:
```
Yarn Available = Current_Inventory + On_Order - Allocated_to_G00_WIP
```

### Fabric Flow Impact on Yarn Planning

**Forward Planning** (Yarn → Fabric availability):
```
Yarn Order → 8-14 weeks delivery → G00 knitting → 4 weeks → F01 available

Total Lead Time: Yarn supplier lead time + 4 week production cycle
```

**Backward Planning** (Fabric demand → Yarn requirements):
```
F01 demand forecast → BOM explosion → Yarn requirements → Procurement timing
```

## Data Integration Points

### Current Data Files Mapping to Production Stages

**Style_BOM.csv**:
- **Usage Point**: G00 stage (yarn consumption)
- **Calculation**: Fabric yards × BOM_Percentage = Yarn pounds needed

**Yarn_ID.csv (Planning_Balance)**:
- **Updates**: When yarn allocated to G00 production
- **Formula**: Inventory + On_Order - G00_Allocated = Available

**eFab_Inventory_[Location].xlsx Files**:
- **G00**: Raw knitted fabric WIP
- **G02**: Fabric in finishing WIP  
- **I01**: Fabric awaiting inspection
- **F01**: Available finished fabric (**PRIMARY FOCUS**)
- **P01**: Allocated fabric for shipment

**Sales Activity Report.csv**:
- **Impact Point**: F01 → P01 (fabric allocation)
- **Planning Trigger**: F01 depletion drives new G00 production needs

**cfab_Yarn_Demand_By_Style.csv**:
- **Source**: Projected F01 demand working backwards through production
- **Purpose**: Drive G00 production scheduling and yarn procurement

## Production Planning Logic Flow

### 1. Demand Signal (F01 Level)
```
Customer orders + forecast → F01 demand → Safety stock check
If F01 < 20-day safety stock → Trigger production
```

### 2. Production Trigger (G00 Level)  
```
F01 replenishment need → G00 production order → Yarn allocation check
If yarn available → Allocate yarn and start G00 production
If yarn shortage → Procurement alert
```

### 3. Yarn Planning Impact
```
G00 production plan → BOM explosion → Yarn requirements
Compare to Planning_Balance → Identify shortages → Purchase orders
```

## Critical Production Metrics for Supply Chain Planning

### Lead Time Tracking
- **G00 → G02**: Finishing lead time
- **G02 → I01**: Inspection queue time  
- **I01 → F01**: Quality approval time
- **F01 → P01**: Order allocation time

### Capacity Constraints
- **G00 knitting capacity**: Limit on new production starts
- **G02 finishing capacity**: Bottleneck identification
- **I01 inspection capacity**: Quality gate throughput

### Yield Factors
- **Yarn to fabric conversion**: Efficiency at G00
- **Finishing yield**: Fabric loss during G02 processing
- **Quality pass rate**: Approval rate at I01

## Data Requirements for Production Flow Tracking

### Real-time Updates Needed
```
1. G00 production starts → Update yarn Planning_Balance
2. Stage transfers (G00→G02→I01→F01) → Update WIP positions  
3. F01 sales → Update available fabric inventory
4. P01 allocations → Update committed inventory
```

### Weekly Production Planning
```
1. F01 demand forecast → G00 production schedule
2. G00 schedule → Yarn requirement calculation
3. Yarn requirements vs Planning_Balance → Procurement plan
```

### Exception Reporting
```
1. Yarn shortages blocking G00 production
2. Quality failures at I01 reducing F01 availability  
3. F01 below safety stock levels
4. Long WIP aging (G00/G02/I01 stage times)
```

## Supply Chain Integration Points

### F01 as Central Planning Hub
**Why F01 is Primary Focus**:
- Highest $ value inventory stage
- Customer promising point (available to sell)
- Safety stock calculation base
- Drives backward yarn planning

### Yarn Planning Connection
**From F01 demand back to yarn**:
```
F01 forecast → Style requirements → BOM explosion → Yarn needs → Planning_Balance check → Procurement
```

This production flow mapping provides the foundation for yarn requirement planning by clearly showing where yarn gets consumed (G00) and how demand flows backward from customer requirements (F01) to yarn procurement decisions.