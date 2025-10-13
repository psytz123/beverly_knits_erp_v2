# Codebase Prevention Rules & Standards

## Executive Summary

This document outlines comprehensive prevention rules and coding standards designed to prevent technical debt accumulation, security vulnerabilities, and maintainability issues in the Beverly Knits ERP v2 codebase. These rules address current issues identified in the codebase analysis and establish guardrails for future development.

## Current Issues Identified

### Critical Problems
1. **Monolithic Architecture**: Core file with 18,314 lines containing 132 API routes
2. **Error Handling**: 348 exception handlers with 28 bare except clauses
3. **Security Vulnerabilities**: Hardcoded credentials and missing authentication
4. **Data Loading Chaos**: 12+ files with duplicated data loading logic
5. **Testing Gaps**: Insufficient test coverage for 18k+ line core module

---

## Prevention Rules by Category

### ARCHITECTURAL RULES

1. **Module Size Limit**: No file shall exceed 500 lines of code (enforced via CI)
2. **Single Responsibility**: One class per file, one concern per class
3. **API Organization**: Routes must be grouped by domain in separate Flask blueprints
4. **Dependency Injection**: No direct imports between modules; use interfaces
5. **Service Layer**: Business logic must be separated from Flask routes
6. **Layered Architecture**: Strict separation of presentation, business, and data layers
7. **No Circular Dependencies**: Enforced by import linter in CI/CD
8. **Interface Segregation**: Clients should not depend on interfaces they don't use

### CODE QUALITY RULES

1. **Complexity Limit**: Cyclomatic complexity ≤ 10 per function
2. **Function Size**: Maximum 50 lines per function
3. **Class Size**: Maximum 200 lines per class
4. **Duplication**: Less than 3% code duplication (measured by tools like radon)
5. **Type Hints**: Required for all public functions and methods
6. **Naming Conventions**: PEP 8 compliant with domain-specific terminology
7. **Documentation**: Docstrings required for all public APIs
8. **Code Reviews**: Mandatory for all changes, 2 approvals for production

### ERROR HANDLING RULES

1. **No Bare Excepts**: Always catch specific exception types
2. **Error Context**: Include meaningful context in all error messages
3. **Logging Required**: All exceptions must be logged with appropriate level
4. **Graceful Degradation**: Fallback strategies required for external dependencies
5. **Validation First**: Input validation before any processing
6. **Custom Exceptions**: Domain-specific exception hierarchy required
7. **Error Response Format**: Consistent error response structure across all APIs
8. **Circuit Breakers**: Required for all external service calls

### SECURITY RULES

1. **No Hardcoded Secrets**: All credentials must come from environment variables
2. **Input Sanitization**: Required for all user inputs (SQLAlchemy for SQL, bleach for HTML)
3. **SQL Injection Prevention**: Use parameterized queries exclusively
4. **Authentication Required**: All endpoints need authentication check via decorator
5. **Rate Limiting**: Implement on all public endpoints (100 req/min default)
6. **HTTPS Only**: Enforce TLS 1.2+ for all communications
7. **Security Headers**: CSP, HSTS, X-Frame-Options required
8. **Dependency Scanning**: Automated vulnerability scanning in CI/CD

### DATA HANDLING RULES

1. **Single Data Loader**: One unified data loading system with consistent interface
2. **Column Standardization**: Enforce standard column names via mapping config
3. **Cache Strategy**: TTL-based caching with explicit invalidation triggers
4. **Batch Processing**: Required for operations involving > 1000 records
5. **Data Validation**: Schema validation on all inputs using Pydantic/Marshmallow
6. **Transaction Management**: Explicit transaction boundaries for multi-table operations
7. **Optimistic Locking**: Version fields for concurrent update handling
8. **Data Sanitization**: PII data must be masked in logs

### TESTING RULES

1. **Coverage Minimum**: 80% line coverage, 70% branch coverage required
2. **Test Before Merge**: All PRs need passing tests
3. **Performance Tests**: Required for all data-intensive operations
4. **Security Tests**: OWASP Top 10 coverage required
5. **Integration Tests**: Required for all API endpoints
6. **Unit Test Isolation**: No external dependencies in unit tests
7. **Test Data Management**: Fixtures and factories, no production data
8. **Regression Tests**: Required for all bug fixes

### DEPLOYMENT RULES

1. **Environment Configs**: Separate configurations per environment (dev/staging/prod)
2. **Health Checks**: Required endpoints: /health (basic), /ready (dependencies)
3. **Rollback Plan**: Documented and tested for each deployment
4. **Load Testing**: Required before production deployment
5. **Monitoring**: APM and error tracking required (Sentry/New Relic)
6. **Blue-Green Deployment**: Zero-downtime deployments required
7. **Feature Flags**: All new features behind toggles
8. **Deployment Checklist**: Mandatory pre-deployment verification

### PERFORMANCE & SCALABILITY RULES

1. **Database Query Optimization**: No N+1 queries, use eager loading
2. **Pagination Required**: All list endpoints must support pagination (max 100 items)
3. **Async Processing**: Operations > 5 seconds must use job queue (Celery/RQ)
4. **Memory Limits**: Functions cannot hold > 100MB in memory
5. **Connection Pooling**: Database connections must use pool (min 5, max 20)
6. **Index Requirements**: All foreign keys and frequently queried columns indexed
7. **Query Timeout**: 30-second maximum query execution time
8. **Bulk Operations**: Use bulk insert/update for > 100 records

### API DESIGN RULES

1. **RESTful Standards**: Strict REST conventions (GET=read, POST=create, etc.)
2. **API Versioning**: All endpoints versioned (/api/v1/, /api/v2/)
3. **Response Consistency**: Standard format `{data: {}, meta: {}, errors: []}`
4. **HTTP Status Codes**: Proper codes (200, 201, 400, 401, 403, 404, 500)
5. **Request ID Tracking**: X-Request-ID header for request tracing
6. **API Documentation**: OpenAPI/Swagger spec required and auto-generated
7. **Deprecation Policy**: 90-day notice with Sunset headers
8. **CORS Configuration**: Explicit allowed origins, no wildcards in production

### DATA INTEGRITY RULES

1. **Transaction Boundaries**: All multi-table operations in database transactions
2. **Idempotency**: All POST/PUT operations must be idempotent
3. **Soft Deletes**: No hard deletes, use deleted_at timestamp
4. **Audit Trail**: All data changes logged with user/timestamp
5. **Data Validation**: Pydantic/Marshmallow schemas for all inputs
6. **Referential Integrity**: Foreign key constraints enforced at DB level
7. **Unique Constraints**: Business keys must have database-level uniqueness
8. **Backup Strategy**: Daily backups with 30-day retention minimum

### MONITORING & OBSERVABILITY RULES

1. **Structured Logging**: JSON format with correlation IDs
2. **Metrics Collection**: Response time, error rate, throughput per endpoint
3. **Alert Thresholds**: Error rate > 1%, response time > 2s, CPU > 80%
4. **Distributed Tracing**: OpenTelemetry for request flow tracking
5. **Health Endpoints**: /health (liveness), /ready (readiness)
6. **Log Retention**: 30 days hot storage, 90 days cold storage
7. **Dashboard Requirements**: Key metrics visible in single dashboard
8. **SLA Monitoring**: 99.9% uptime tracking with automated alerts

### DEVELOPMENT WORKFLOW RULES

1. **Branch Protection**: No direct commits to main/master branch
2. **PR Size Limit**: Maximum 400 lines changed per pull request
3. **Code Review**: 2 approvals required for production code changes
4. **Commit Messages**: Conventional commits format (feat:, fix:, refactor:)
5. **Pre-commit Hooks**: Format, lint, type-check before commit
6. **Feature Flags**: All new features behind feature toggles
7. **Database Migrations**: All migrations must be reversible
8. **Documentation Updates**: Code and documentation in same PR

### DEPENDENCY MANAGEMENT RULES

1. **Version Pinning**: Exact versions in requirements.txt/pyproject.toml
2. **Security Scanning**: Daily vulnerability scans (Snyk/Safety)
3. **License Check**: Only approved licenses (MIT, Apache 2.0, BSD)
4. **Update Schedule**: Monthly dependency updates with testing
5. **Vendor Lock-in**: Abstract external services behind interfaces
6. **Package Audit**: Justify each new dependency in ADR
7. **Size Limits**: Bundle size < 50MB, Docker image < 500MB
8. **Compatibility Matrix**: Test against Python 3.10, 3.11, 3.12

### MACHINE LEARNING SPECIFIC RULES

1. **Model Versioning**: All models tagged with git hash + timestamp
2. **Training Data Tracking**: Data lineage and version tracked
3. **Model Metrics**: Accuracy, precision, recall logged per training
4. **A/B Testing**: New models tested against production baseline
5. **Drift Detection**: Monitor prediction distribution changes
6. **Fallback Models**: Always have n-1 version available
7. **Explainability**: Feature importance tracked for decisions
8. **Retraining Schedule**: Weekly assessment, monthly retraining

### INCIDENT RESPONSE RULES

1. **Runbook Required**: Each service has operational runbook
2. **Rollback Time**: Less than 5 minutes to previous version
3. **Circuit Breakers**: Auto-disable failing features
4. **Incident Templates**: Standardized incident report format
5. **Post-Mortem**: Required for all P1/P2 incidents (blameless)
6. **Recovery Time Objective**: 1 hour for critical services
7. **Communication Plan**: Stakeholder notification matrix defined
8. **Chaos Testing**: Monthly failure scenario testing

### CRITICAL ANTI-PATTERNS TO PREVENT

1. **No God Objects**: Classes with more than 10 responsibilities
2. **No Magic Numbers**: All constants in configuration files
3. **No Circular Dependencies**: Enforced by import linter
4. **No Shared Mutable State**: Use immutable data structures
5. **No Synchronous External Calls**: Use async/queue for external APIs
6. **No Catch-and-Ignore**: All exceptions must be handled appropriately
7. **No Copy-Paste Code**: DRY principle strictly enforced
8. **No Mixed Concerns**: Separate business logic from infrastructure

### BEVERLY KNITS ERP SPECIFIC RULES

1. **Yarn Data Consistency**: Single source of truth for yarn inventory
2. **BOM Validation**: Style-to-yarn mappings verified on every load
3. **Production Order State**: Explicit state machine for order lifecycle
4. **Work Center Patterns**: Regex validation for x.xx.xx.X format
5. **Planning Balance**: Never allow negative allocated values
6. **Column Name Mapping**: Centralized column name resolver service
7. **Cache Invalidation**: Clear cache on any data file updates
8. **eFab Integration**: Retry logic with exponential backoff (3 retries max)
9. **Knit Order Processing**: Validate machine assignments against work centers
10. **SharePoint Sync**: Daily validation of data consistency

---

## Implementation Priority

### Phase 1: Critical (Immediate)
- Fix monolithic architecture (split beverly_comprehensive_erp.py)
- Remove hardcoded credentials
- Implement proper error handling
- Add authentication to all endpoints

### Phase 2: High (Within 1 Month)
- Consolidate data loading logic
- Implement comprehensive testing
- Add monitoring and alerting
- Create API documentation

### Phase 3: Medium (Within 3 Months)
- Implement caching strategy
- Add performance optimizations
- Create operational runbooks
- Implement feature flags

### Phase 4: Long-term (Within 6 Months)
- Complete microservices migration
- Implement full CI/CD pipeline
- Add chaos engineering tests
- Achieve 90% test coverage

---

## Enforcement Mechanisms

### Automated Enforcement
1. **Pre-commit Hooks**: black, isort, flake8, mypy
2. **CI/CD Pipeline**: pytest, coverage, security scanning
3. **Code Analysis**: SonarQube, CodeClimate
4. **Dependency Scanning**: Snyk, Safety
5. **Performance Testing**: Locust, Apache Bench

### Manual Enforcement
1. **Code Reviews**: Mandatory peer review
2. **Architecture Reviews**: Monthly architecture board meetings
3. **Security Audits**: Quarterly security assessments
4. **Documentation Reviews**: Technical writer validation

---

## Metrics & KPIs

### Code Quality Metrics
- Code coverage: Target 85% (Current: ~40%)
- Cyclomatic complexity: Target < 10 (Current: >20 in many functions)
- Technical debt ratio: Target < 5% (Current: ~15%)
- Duplication: Target < 3% (Current: ~8%)

### Performance Metrics
- API response time: p95 < 200ms
- Database query time: p95 < 100ms
- Page load time: < 3 seconds
- Throughput: > 1000 req/sec

### Reliability Metrics
- Uptime: 99.9% SLA
- MTTR: < 1 hour
- Error rate: < 1%
- Rollback success rate: 100%

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-09-14 | AI Assistant | Initial comprehensive rules documentation |

---

## References

- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [12 Factor App](https://12factor.net/)
- [Clean Code by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Domain-Driven Design by Eric Evans](https://www.amazon.com/Domain-Driven-Design-Tackling-Complexity-Software/dp/0321125215)

---

## Appendix: Quick Reference Checklist

### Before Committing Code
- [ ] File < 500 lines?
- [ ] Functions < 50 lines?
- [ ] Cyclomatic complexity < 10?
- [ ] Type hints added?
- [ ] Tests written?
- [ ] Documentation updated?
- [ ] No hardcoded values?
- [ ] Error handling complete?

### Before Creating PR
- [ ] All tests passing?
- [ ] Coverage > 80%?
- [ ] No security issues?
- [ ] Performance tested?
- [ ] Documentation complete?
- [ ] PR < 400 lines?
- [ ] Commit messages follow convention?

### Before Deployment
- [ ] Load testing complete?
- [ ] Rollback plan documented?
- [ ] Monitoring configured?
- [ ] Feature flags set?
- [ ] Runbook updated?
- [ ] Stakeholders notified?