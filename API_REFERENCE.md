# AI Supply Chain Framework - API Reference

Complete API documentation for the AI Supply Chain Optimization Framework, covering all endpoints, agents, and integration capabilities.

## Base Configuration

```
Base URL: https://api.ai-supply-chain.com/v1
Authentication: Bearer token (JWT)
Content-Type: application/json
Rate Limit: 1000 requests/hour per tenant
```

---

## Core Framework APIs

### Framework Initialization

#### Initialize Framework Instance
```http
POST /framework/initialize
```

**Request Body:**
```json
{
  "industry_type": "FURNITURE|INJECTION_MOLDING|ELECTRICAL_EQUIPMENT|TEXTILE|GENERIC",
  "complexity": "SIMPLE|MODERATE|COMPLEX|ENTERPRISE", 
  "customer_config": {
    "company_size": "small|medium|large|enterprise",
    "company_name": "string",
    "industry_specific_requirements": {},
    "compliance_requirements": ["ISO_9001", "FDA", "UL"],
    "integration_requirements": ["SAP", "Oracle", "MES"]
  }
}
```

**Response:**
```json
{
  "framework_id": "fw_abc123",
  "status": "initialized",
  "industry_type": "FURNITURE", 
  "estimated_implementation_weeks": 8,
  "confidence_score": 0.94,
  "assigned_agents": [
    "data_migration_intelligence",
    "configuration_generation", 
    "furniture_manufacturing",
    "customer_success_prediction"
  ]
}
```

#### Get Framework Status
```http
GET /framework/{framework_id}/status
```

**Response:**
```json
{
  "framework_id": "fw_abc123",
  "deployment_phase": "SETUP|TESTING|PRODUCTION",
  "current_phase": "data_migration",
  "progress_percentage": 45.2,
  "health_status": {
    "overall": "HEALTHY|WARNING|CRITICAL",
    "components": {
      "inventory_manager": "HEALTHY",
      "production_planner": "HEALTHY", 
      "forecasting_engine": "WARNING"
    }
  },
  "performance_metrics": {
    "response_time_ms": 145,
    "throughput_requests_per_minute": 850,
    "error_rate": 0.02
  }
}
```

---

## Agent Communication APIs

### Data Migration Intelligence Agent

#### Discover Legacy Systems
```http
POST /agents/data_migration/discover
```

**Request Body:**
```json
{
  "customer_id": "cust_123",
  "connection_details": {
    "system_type": "SAP|ORACLE|QUICKBOOKS|EXCEL|CSV|DATABASE",
    "connection_string": "string",
    "file_path": "/path/to/data.xlsx", 
    "credentials": {
      "username": "string",
      "password": "string",
      "host": "string",
      "port": 5432
    }
  }
}
```

**Response:**
```json
{
  "discovery_id": "disc_456",
  "discovered_sources": [
    {
      "source_id": "inventory_table",
      "source_type": "DATABASE_TABLE",
      "estimated_records": 12000,
      "complexity_factor": 3.2,
      "column_count": 47,
      "data_quality": "GOOD"
    }
  ],
  "schema_analysis": {
    "total_tables": 12,
    "total_columns": 340,
    "confidence_score": 0.87,
    "column_mappings": {
      "inventory": [
        {
          "source_column": "Item_Number",
          "target_column": "item_id", 
          "confidence_score": 0.95,
          "transformation_rule": "direct_copy"
        }
      ]
    }
  },
  "complexity_assessment": {
    "overall_complexity": 6.4,
    "estimated_duration_hours": 24,
    "migration_complexity": "MEDIUM",
    "risk_level": "LOW"
  }
}
```

#### Execute Data Migration
```http
POST /agents/data_migration/execute
```

**Request Body:**
```json
{
  "migration_job": {
    "customer_id": "cust_123",
    "source_config": {
      "source_id": "legacy_erp",
      "connection_details": {},
      "table_mappings": {}
    },
    "target_schema": {},
    "column_mappings": [],
    "execution_parameters": {
      "batch_size": 1000,
      "parallel_processing": true,
      "validation_enabled": true
    }
  }
}
```

**Response:**
```json
{
  "job_id": "mig_789",
  "status": "STARTED|IN_PROGRESS|COMPLETED|FAILED",
  "records_processed": 8500,
  "quality_score": 0.96,
  "duration_hours": 2.3,
  "errors": [],
  "warnings": ["Column 'old_field' not mapped"],
  "performance_metrics": {
    "throughput_records_per_hour": 3700,
    "error_rate": 0.004,
    "success_rate": 0.996
  }
}
```

### Configuration Generation Agent

#### Generate System Configuration
```http
POST /agents/configuration/generate
```

**Request Body:**
```json
{
  "customer_id": "cust_123",
  "industry": "furniture_manufacturing",
  "complexity": "MODERATE",
  "config_types": ["COMPLETE_SYSTEM", "BUSINESS_RULES", "COMPLIANCE"],
  "business_requirements": {
    "workflows": {
      "approval_levels": 3,
      "business_rules": {
        "inventory_reorder": "min_max_planning",
        "production_scheduling": "capacity_constrained"
      }
    },
    "reporting": {
      "dashboards": ["executive", "operations", "quality"],
      "kpis": ["efficiency", "quality", "delivery"]
    }
  },
  "technical_requirements": {
    "performance": {
      "response_time_ms": 200,
      "concurrent_users": 100
    },
    "security": {
      "authentication": "multi_factor",
      "encryption": "aes_256"
    }
  },
  "integration_requirements": ["SAP", "MES_Platform"],
  "compliance_requirements": ["ISO_9001", "OSHA"]
}
```

**Response:**
```json
{
  "request_id": "cfg_101",
  "generated_configurations": [
    {
      "config_id": "cfg_101_COMPLETE_SYSTEM",
      "configuration_type": "COMPLETE_SYSTEM",
      "confidence_score": 0.93,
      "estimated_implementation_hours": 16,
      "customizations_count": 12,
      "warnings": [],
      "recommendations": [
        "Consider increasing user training budget",
        "Add comprehensive security configuration"
      ]
    }
  ],
  "implementation_plan": {
    "total_estimated_hours": 40,
    "estimated_duration_weeks": 1.0,
    "phases": {
      "Phase 1 - System Setup": {
        "duration_hours": 12,
        "tasks": ["Infrastructure setup", "Core system installation"]
      },
      "Phase 2 - Configuration": {
        "duration_hours": 16,
        "tasks": ["Apply generated configurations", "Customize business rules"]
      }
    }
  },
  "overall_confidence": 0.93
}
```

### Customer Success Prediction Agent

#### Predict Implementation Success
```http
POST /agents/success_prediction/predict
```

**Request Body:**
```json
{
  "customer_profile": {
    "customer_id": "cust_123",
    "industry": "INJECTION_MOLDING",
    "company_size": "medium",
    "previous_erp_experience": true
  },
  "project_characteristics": {
    "complexity_score": 0.6,
    "budget_adequacy": 0.8,
    "timeline_realism": 0.7,
    "data_quality_score": 0.85,
    "integration_complexity": 0.4,
    "compliance_requirements": 3,
    "user_training_budget": 0.7
  },
  "organizational_factors": {
    "executive_support": 0.9,
    "technical_readiness": 0.75,
    "organizational_readiness": 0.8,
    "stakeholder_engagement": 0.7,
    "change_management_maturity": 0.6,
    "internal_it_capability": 0.8,
    "project_management_maturity": 0.85
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_202",
  "success_probability": 0.87,
  "risk_level": "LOW",
  "estimated_timeline_weeks": 8,
  "confidence_score": 0.91,
  "key_risk_factors": [
    "Change management maturity below optimal level"
  ],
  "success_enablers": [
    "Strong executive support: 0.90", 
    "Good data quality score: 0.85",
    "High technical readiness: 0.75"
  ],
  "recommended_actions": [
    "Implement change management program with user champions",
    "Establish executive steering committee with regular reviews"
  ],
  "predicted_satisfaction_score": 8.4,
  "milestone_probabilities": {
    "requirements_gathering": 0.92,
    "system_design": 0.89,
    "development": 0.86,
    "testing": 0.91,
    "training": 0.78,
    "go_live": 0.87
  },
  "detailed_analysis": {
    "strengths": [
      "executive_support: 0.90",
      "technical_readiness: 0.75",
      "data_quality_score: 0.85"
    ],
    "improvement_areas": [
      "change_management_maturity: 0.60"
    ],
    "industry_benchmarks": {
      "industry_baseline_success_rate": 0.82,
      "customer_vs_baseline": "Above average"
    }
  }
}
```

---

## Industry-Specific APIs

### Furniture Manufacturing

#### Optimize Wood Inventory
```http
POST /industry/furniture/wood-inventory/optimize
```

**Request Body:**
```json
{
  "inventory_data": [
    {
      "wood_type": "Oak",
      "grade": "A",
      "moisture_content": 8.5,
      "board_feet": 1200,
      "cost_per_bf": 4.25,
      "supplier": "Wood Supplier Co"
    }
  ],
  "optimization_criteria": {
    "minimize_cost": true,
    "moisture_preference": "8-12%",
    "grade_preference": ["A", "B"]
  }
}
```

**Response:**
```json
{
  "optimization_id": "opt_wood_301",
  "optimized_inventory": [
    {
      "wood_type": "Oak",
      "recommended_quantity": 1000,
      "optimization_reason": "Optimal moisture content and grade balance",
      "cost_savings": 127.50,
      "yield_improvement": 0.12
    }
  ],
  "total_cost_savings": 127.50,
  "recommendations": [
    "Source additional Grade A Oak to improve yield",
    "Monitor moisture content to maintain 8-12% range"
  ]
}
```

### Injection Molding

#### Optimize Process Parameters
```http
POST /industry/injection_molding/process/optimize
```

**Request Body:**
```json
{
  "material_specs": {
    "resin_type": "ABS",
    "melt_flow_index": 22,
    "density": 1.05
  },
  "part_specifications": {
    "weight_grams": 45.2,
    "wall_thickness": 2.5,
    "complexity_factor": 0.7
  },
  "current_parameters": {
    "injection_pressure": 1200,
    "hold_pressure": 800,
    "mold_temperature": 60,
    "melt_temperature": 240,
    "cycle_time": 35
  }
}
```

**Response:**
```json
{
  "optimization_id": "opt_process_401", 
  "optimized_parameters": {
    "injection_pressure": 1150,
    "hold_pressure": 850,
    "mold_temperature": 65,
    "melt_temperature": 235,
    "cycle_time": 32
  },
  "predicted_improvements": {
    "cycle_time_reduction": 8.6,
    "quality_improvement": 0.15,
    "material_savings": 0.08
  },
  "recommendations": [
    "Increase hold pressure for better part density",
    "Reduce cycle time while maintaining quality",
    "Monitor part dimensions during optimization"
  ]
}
```

---

## Integration APIs

### Legacy System Connectors

#### SAP Integration
```http
POST /integrations/sap/connect
```

**Request Body:**
```json
{
  "connection_config": {
    "client": "100",
    "username": "sap_user",
    "password": "encrypted_password",
    "application_server": "sap.company.com",
    "system_number": "00",
    "language": "EN"
  },
  "sync_config": {
    "modules": ["MM", "PP", "QM"],
    "sync_frequency": "real_time",
    "batch_size": 1000
  }
}
```

**Response:**
```json
{
  "connection_id": "sap_conn_501",
  "status": "connected",
  "available_modules": ["MM", "PP", "QM", "FI", "SD"],
  "sync_status": {
    "MM": "active",
    "PP": "active", 
    "QM": "pending"
  },
  "last_sync": "2024-01-15T14:30:00Z",
  "records_synced": 15420
}
```

#### Oracle Integration
```http
POST /integrations/oracle/connect
```

**Request Body:**
```json
{
  "connection_config": {
    "host": "oracle.company.com",
    "port": 1521,
    "service_name": "ORCL",
    "username": "oracle_user",
    "password": "encrypted_password"
  },
  "mapping_config": {
    "tables": ["INVENTORY", "WORK_ORDERS", "BOM"],
    "custom_mappings": {}
  }
}
```

---

## Monitoring & Analytics APIs

### Framework Performance
```http
GET /monitoring/performance
```

**Response:**
```json
{
  "system_health": {
    "overall_status": "HEALTHY",
    "uptime_percentage": 99.94,
    "active_implementations": 23,
    "total_customers": 156
  },
  "performance_metrics": {
    "avg_response_time_ms": 145,
    "requests_per_minute": 2340,
    "error_rate_percentage": 0.08,
    "cache_hit_rate": 0.87
  },
  "agent_metrics": {
    "data_migration_intelligence": {
      "active_jobs": 5,
      "success_rate": 0.96,
      "avg_duration_hours": 3.2
    },
    "configuration_generation": {
      "configurations_generated": 45,
      "avg_confidence": 0.91,
      "customization_rate": 0.73
    },
    "success_prediction": {
      "predictions_made": 67,
      "accuracy_rate": 0.94,
      "early_warnings": 12
    }
  }
}
```

### Implementation Analytics
```http
GET /analytics/implementations
```

**Query Parameters:**
- `industry`: Filter by industry type
- `date_range`: Date range for analysis
- `customer_size`: Filter by company size

**Response:**
```json
{
  "summary": {
    "total_implementations": 156,
    "success_rate": 0.953,
    "avg_implementation_weeks": 7.2,
    "customer_satisfaction": 4.6
  },
  "by_industry": {
    "FURNITURE": {
      "count": 45,
      "success_rate": 0.96,
      "avg_weeks": 6.8
    },
    "INJECTION_MOLDING": {
      "count": 38,
      "success_rate": 0.95,
      "avg_weeks": 7.5
    }
  },
  "trends": {
    "implementation_time": {
      "trend": "decreasing",
      "improvement_rate": 0.08
    },
    "success_rate": {
      "trend": "stable", 
      "current_streak": 23
    }
  }
}
```

---

## Webhook Events

### Implementation Status Updates
```json
{
  "event": "implementation.status_changed",
  "framework_id": "fw_abc123",
  "customer_id": "cust_123",
  "timestamp": "2024-01-15T14:30:00Z",
  "data": {
    "old_status": "data_migration",
    "new_status": "configuration",
    "progress_percentage": 45.2,
    "estimated_completion": "2024-01-29T17:00:00Z"
  }
}
```

### Agent Completion Events
```json
{
  "event": "agent.task_completed",
  "agent_id": "data_migration_intelligence", 
  "task_id": "mig_789",
  "customer_id": "cust_123",
  "timestamp": "2024-01-15T16:45:00Z",
  "data": {
    "task_type": "data_migration",
    "status": "completed",
    "duration_hours": 2.3,
    "records_processed": 8500,
    "quality_score": 0.96
  }
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "INVALID_INDUSTRY_TYPE",
    "message": "The specified industry type is not supported",
    "details": {
      "provided": "AUTOMOTIVE",
      "supported": ["FURNITURE", "INJECTION_MOLDING", "ELECTRICAL_EQUIPMENT", "TEXTILE", "GENERIC"]
    },
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes
- `FRAMEWORK_NOT_FOUND`: Framework instance not found
- `INVALID_INDUSTRY_TYPE`: Unsupported industry type
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `AGENT_UNAVAILABLE`: Required agent is not available
- `DATA_VALIDATION_FAILED`: Request data validation failed
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `INTERNAL_SERVER_ERROR`: Internal server error

---

## Authentication & Security

### Bearer Token Authentication
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### API Key Authentication (Alternative)
```http
X-API-Key: your-api-key-here
```

### Rate Limiting
- **Standard Tier**: 1,000 requests/hour
- **Professional Tier**: 5,000 requests/hour  
- **Enterprise Tier**: 25,000 requests/hour

### Security Headers
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
```

---

*API Reference v1.0.0 - AI Supply Chain Optimization Framework*