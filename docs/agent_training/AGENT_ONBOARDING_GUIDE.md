# Agent Onboarding Guide for Beverly Knits ERP v2

## Table of Contents
1. [System Overview](#system-overview)
2. [Agent Roles & Responsibilities](#agent-roles--responsibilities)
3. [Training Process](#training-process)
4. [Performance Standards](#performance-standards)
5. [Deployment Phases](#deployment-phases)
6. [Continuous Learning](#continuous-learning)

## System Overview

Beverly Knits ERP v2 is a production-ready textile manufacturing ERP system that manages:
- **1,199 yarn items** with real-time inventory tracking
- **28,653 BOM entries** mapping styles to yarns
- **194 production orders** across 91 work centers
- **285 machines** with capacity planning
- **557,671 lbs** total production workload

### Core Technologies
- **Backend**: Python/Flask (Port 5006)
- **ML Models**: ARIMA, Prophet, LSTM, XGBoost
- **Data Storage**: PostgreSQL + Redis caching
- **Performance**: <200ms API response, 70-90% cache hit rate

## Agent Roles & Responsibilities

### 1. Inventory Intelligence Agent
**Primary Responsibilities:**
- Calculate Planning Balance (Physical Inventory - Allocated + On Order)
- Detect yarn shortages and overstock situations
- Perform multi-level BOM netting calculations
- Generate inventory optimization recommendations

**Key Metrics:**
- Accuracy: >95% for balance calculations
- Response time: <100ms for queries
- Error tolerance: <1% calculation errors

**Training Data:**
- `yarn_inventory.xlsx` - 1,199 yarn items with Planning Balance
- `BOM_updated.csv` - 28,653 style-to-yarn mappings
- Historical inventory snapshots (6 months)

### 2. Forecast Intelligence Agent
**Primary Responsibilities:**
- Train and maintain ML forecasting models
- Generate demand predictions (9-week to 90-day horizons)
- Monitor forecast accuracy and trigger retraining
- Manage ensemble model weights

**Key Metrics:**
- MAPE: <15% for 30-day forecasts
- Model training time: <5 minutes
- Retraining frequency: Weekly for critical models

**Training Data:**
- `Sales Activity Report.csv` - 10,338+ historical records
- Seasonal patterns from past 2 years
- Market trend indicators

### 3. Production Planning Agent
**Primary Responsibilities:**
- Schedule production across 285 machines
- Optimize capacity utilization (target >85%)
- Manage 6-phase production planning
- Generate machine assignment suggestions

**Key Metrics:**
- Schedule efficiency: >90%
- On-time delivery: >95%
- Machine utilization: 85-95%

**Training Data:**
- `eFab_Knit_Orders.csv` - 194 production orders
- `Machine Report fin1.csv` - Machine capabilities
- `QuadS_greigeFabricList.xlsx` - Work center mappings

### 4. Yarn Substitution Agent
**Primary Responsibilities:**
- Identify compatible yarn substitutes
- Analyze color, weight, and material similarities
- Calculate substitution impact on quality
- Recommend procurement alternatives

**Key Metrics:**
- Substitution accuracy: >90%
- Quality impact: <5% variance
- Cost optimization: 10-15% savings

**Training Data:**
- `Yarn_ID_Master.csv` - Complete yarn specifications
- Historical substitution success/failure records
- Supplier reliability scores

### 5. Quality Assurance Agent
**Primary Responsibilities:**
- Validate data integrity across systems
- Monitor KPI performance
- Detect anomalies in production flows
- Generate quality reports

**Key Metrics:**
- Data validation accuracy: >99%
- Anomaly detection rate: >85%
- False positive rate: <10%

## Training Process

### Phase 1: Knowledge Acquisition (Days 1-3)
```python
# Load domain knowledge
agent.load_knowledge_base([
    'CLAUDE.md',
    'PRESERVED_CONTENT.md',
    'API_MAPPING_DOCUMENTATION.md'
])

# Configure business rules
agent.configure_rules({
    'planning_balance': 'inventory - allocated + on_order',
    'minimum_stock': '2_weeks_demand',
    'reorder_point': '4_weeks_demand'
})
```

### Phase 2: Skill Development (Days 4-10)
```python
# Train on historical data
training_data = load_training_data('data/agent_training/scenarios/')
agent.train(training_data, epochs=100, validation_split=0.2)

# Practice specific scenarios
for scenario in ['normal_ops', 'stockout', 'rush_order', 'machine_failure']:
    agent.practice_scenario(scenario)
    evaluate_performance(agent, scenario)
```

### Phase 3: Integration Testing (Days 11-14)
```python
# Test API interactions
test_endpoints = [
    '/api/inventory-intelligence-enhanced',
    '/api/ml-forecast-detailed',
    '/api/production-planning',
    '/api/yarn-substitution-intelligent'
]

for endpoint in test_endpoints:
    response = agent.query_api(endpoint)
    validate_response(response)
```

### Phase 4: Certification (Day 15)
```python
# Run certification suite
certification_results = run_certification_tests(agent, test_count=100)

# Required passing scores
requirements = {
    'accuracy': 0.85,
    'speed': 200,  # ms
    'reliability': 0.95,
    'error_rate': 0.05
}

if all(certification_results[metric] >= requirements[metric] for metric in requirements):
    agent.certify()
```

## Performance Standards

### Response Time Requirements
| Operation | Target | Maximum |
|-----------|--------|---------|
| Simple Query | 50ms | 100ms |
| Complex Calculation | 100ms | 200ms |
| ML Prediction | 200ms | 500ms |
| Report Generation | 1s | 3s |

### Accuracy Standards
| Metric | Minimum | Target | Excellence |
|--------|---------|--------|------------|
| Data Validation | 95% | 98% | 99.5% |
| Forecast MAPE | 20% | 15% | 10% |
| Planning Accuracy | 85% | 90% | 95% |
| Substitution Success | 80% | 90% | 95% |

### Learning Progression
```
Week 1: 60% accuracy → Basic operations
Week 2: 75% accuracy → Complex scenarios
Week 3: 85% accuracy → Edge cases
Week 4: 90% accuracy → Production ready
Week 6: 95% accuracy → Expert level
```

## Deployment Phases

### Phase 1: Shadow Mode (Week 1-2)
- Run parallel to existing systems
- Log all decisions without execution
- Compare with human operators
- Collect performance metrics

### Phase 2: Advisory Mode (Week 3-4)
- Provide recommendations to users
- Require human approval for actions
- Track acceptance/rejection rates
- Refine decision algorithms

### Phase 3: Supervised Autonomy (Week 5-6)
- Execute low-risk decisions automatically
- Flag high-risk decisions for review
- Monitor error rates closely
- Implement rollback procedures

### Phase 4: Full Autonomy (Week 7+)
- Operate independently within defined boundaries
- Self-monitor performance
- Request human intervention when uncertain
- Continuous learning from feedback

## Continuous Learning

### Daily Learning Cycle
```python
# Morning: Review overnight operations
agent.review_logs(period='last_24_hours')
agent.identify_patterns()

# Midday: Update models if needed
if agent.performance_degraded():
    agent.retrain_models()

# Evening: Generate improvement suggestions
suggestions = agent.analyze_improvement_opportunities()
agent.propose_optimizations(suggestions)
```

### Weekly Improvement Process
1. **Monday**: Analyze previous week's performance
2. **Tuesday**: Identify learning opportunities
3. **Wednesday**: Update training datasets
4. **Thursday**: Retrain specialized models
5. **Friday**: Validate improvements
6. **Weekend**: Run comprehensive testing

### Monthly Evaluation
- Review all KPIs against targets
- Identify systematic issues
- Update business rules if needed
- Plan next month's focus areas

## Error Handling & Recovery

### Common Error Scenarios
1. **Data Quality Issues**
   - Detection: Validation failures >5%
   - Response: Switch to conservative mode
   - Recovery: Request data cleanup

2. **Model Drift**
   - Detection: Accuracy drop >10%
   - Response: Trigger immediate retraining
   - Recovery: Rollback to previous model

3. **System Overload**
   - Detection: Response time >500ms
   - Response: Prioritize critical operations
   - Recovery: Scale resources or defer non-critical tasks

### Escalation Protocol
```python
if error_severity == 'LOW':
    agent.log_and_continue()
elif error_severity == 'MEDIUM':
    agent.alert_supervisor()
    agent.switch_to_safe_mode()
elif error_severity == 'HIGH':
    agent.stop_operations()
    agent.trigger_emergency_protocol()
    human_intervention_required()
```

## Integration Points

### API Endpoints
Agents must integrate with these critical endpoints:
- `/api/inventory-intelligence-enhanced` - Inventory analysis
- `/api/ml-forecast-detailed` - Demand forecasting
- `/api/production-planning` - Production scheduling
- `/api/yarn-substitution-intelligent` - Substitution recommendations
- `/api/comprehensive-kpis` - Performance metrics

### Database Connections
```python
# Primary data sources
PRODUCTION_DB = 'postgresql://localhost:5432/beverly_knits'
CACHE_DB = 'redis://localhost:6379'
FILE_STORAGE = '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/'
```

### Communication Protocols
- **REST API**: Primary communication method
- **WebSocket**: Real-time updates
- **Message Queue**: Async task processing
- **File System**: Large data transfers

## Security & Compliance

### Access Control
```python
agent_permissions = {
    'read': ['all_data'],
    'write': ['recommendations', 'forecasts'],
    'execute': ['approved_actions'],
    'restricted': ['financial_data', 'personnel_records']
}
```

### Audit Trail
All agent actions must be logged with:
- Timestamp
- Agent ID
- Action type
- Input data
- Decision rationale
- Output/result
- Performance metrics

### Compliance Requirements
- GDPR: No personal data processing without consent
- SOX: Financial calculations must be auditable
- Industry: Maintain textile industry standards
- Internal: Follow company policies and procedures

## Support Resources

### Documentation
- Technical: `/docs/technical/`
- Data Mapping: `/docs/technical/DATA_MAPPING_REFERENCE.md`
- Technical Archive: `/docs/technical/PRESERVED_CONTENT.md`

### Training Materials
- Scenarios: `/data/agent_training/scenarios/`
- Test Data: `/tests/agent_tests/`
- Benchmarks: `/data/agent_training/benchmarks/`

### Contact Points
- Technical Lead: ERP System Administrator
- Business Owner: Production Manager
- Data Steward: Database Administrator
- ML Expert: Data Science Team

## Success Metrics

### Key Performance Indicators
1. **Operational Efficiency**: 20% reduction in manual tasks
2. **Decision Speed**: 50% faster than manual process
3. **Error Rate**: <5% compared to human baseline
4. **Cost Savings**: 15% through optimization
5. **User Satisfaction**: >4.0/5.0 rating

### Monitoring Dashboard
Access the agent performance dashboard at:
```
http://localhost:5006/agent-performance
```

Monitor real-time metrics including:
- Active agents and their status
- Decision accuracy trends
- Response time distributions
- Error logs and alerts
- Learning progression curves

## Conclusion

This onboarding guide provides the foundation for training and deploying intelligent agents in the Beverly Knits ERP system. Success depends on:
1. Thorough training with representative data
2. Gradual deployment with careful monitoring
3. Continuous learning and improvement
4. Strong integration with existing systems
5. Clear communication with human operators

Following this guide will ensure agents are properly prepared to enhance the ERP system's capabilities while maintaining reliability and performance standards.