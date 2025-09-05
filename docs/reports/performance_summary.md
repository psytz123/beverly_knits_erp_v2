# Beverly Knits ERP Performance Analysis Summary

Generated: 2025-09-05T00:49:23.190678

## Critical Bottlenecks

- **/api/inventory-intelligence-enhanced**: High error rate (Impact: CRITICAL)
- **/api/comprehensive-kpis**: High error rate (Impact: CRITICAL)
- **/api/yarn-substitution-intelligent**: High error rate (Impact: CRITICAL)
- **/api/machine-assignment-suggestions**: High error rate (Impact: CRITICAL)
- **/api/factory-floor-ai-dashboard**: High error rate (Impact: CRITICAL)
- **/api/six-phase-supply-chain**: High error rate (Impact: CRITICAL)
- **/api/health**: High error rate (Impact: CRITICAL)
- **/api/production-planning**: Slow response (Impact: MEDIUM)
- **/api/inventory-intelligence-enhanced**: Slow response (Impact: MEDIUM)
- **/api/ml-forecast-detailed**: Slow response (Impact: MEDIUM)

## Top Recommendations

### HIGH - Performance
Implement caching for slow endpoints
*Found 19 slow endpoints. Consider Redis caching.*

### CRITICAL - Stability
Fix error-prone endpoints immediately
*Found 7 endpoints with high error rates.*


## Endpoint Performance

| Endpoint | Avg Time (s) | Error Rate | Memory Delta (MB) |
|----------|-------------|------------|------------------|
| /api/po-risk-analysis | 2.09 | 0% | 0.0 |
| /api/knit-orders | 2.08 | 0% | 0.1 |
| /api/production-pipeline | 2.07 | 0% | 0.0 |
| /api/debug-data | 2.06 | 0% | 0.0 |
| /api/inventory-netting | 2.06 | 0% | 0.0 |
| /api/production-planning | 2.06 | 0% | 0.0 |
| /api/ml-forecast-detailed | 2.06 | 0% | 0.0 |
| /api/comprehensive-kpis | 2.06 | 100% | 0.0 |
| /api/reload-data | 2.06 | 0% | 0.0 |
| /api/health | 2.06 | 100% | 0.0 |
| /api/production-suggestions | 2.05 | 0% | 0.0 |
| /api/yarn-substitution-intelligent | 2.05 | 100% | 0.0 |
| /api/factory-floor-ai-dashboard | 2.05 | 100% | 0.0 |
| /api/inventory-intelligence-enhanced | 2.05 | 100% | 0.0 |
| /api/yarn-intelligence | 2.05 | 0% | -0.0 |
| /api/execute-planning | 2.05 | 0% | 0.0 |
| /api/production-recommendations-ml | 2.05 | 0% | 0.0 |
| /api/machine-assignment-suggestions | 2.04 | 100% | -0.0 |
| /api/six-phase-supply-chain | 2.04 | 100% | 0.0 |
