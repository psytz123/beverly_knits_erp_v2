# Beverly Knits ERP Validation Summary

Generated: 2025-09-05T01:33:23.908648

## Overall Results

- Total Tests: 16
- Passed: 5
- Failed: 11
- Success Rate: 31.2%

## Failed Tests

- **Production Planning API**: Missing required fields: ['production_plan', 'capacity']
- **Inventory Intelligence API**: Expected status 200, got 500
- **ML Forecast API**: Missing required fields: ['forecast']
- **Inventory Netting API**: Missing required fields: ['netting_results']
- **KPIs API**: Expected status 200, got 500
- **Yarn Intelligence API**: Missing required fields: ['yarn_analysis']
- **Yarn Substitution API**: Expected status 200, got 500
- **Health Check API**: Expected status 200, got 404
- **Planning Balance Formula**: No yarn_analysis in response
- **Negative Allocated Values**: 'yarn_analysis'
- **Yarn Shortage Detection**: 'yarn_analysis'

## Endpoint Results

| Endpoint | Status | Message |
|----------|--------|----------|
