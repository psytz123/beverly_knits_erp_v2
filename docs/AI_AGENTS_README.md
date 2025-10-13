# Beverly Knits ERP v2 - AI Agent System

This document describes how to initialize and use the AI Agent System in Beverly Knits ERP v2.

## Quick Start

### Initialize AI Agents

```bash
# Method 1: Using the initialization script
python init_agents.py

# Method 2: Using the startup script (keeps running)
python start_ai_agents.py

# Method 3: Check status only
python init_agents.py --status
```

### Initialization Modes

- **Full Mode** (default): Complete system with all agents
- **Minimal Mode**: Core infrastructure only for development
- **Training Only**: Training system without deployment

```bash
python init_agents.py --mode minimal
python init_agents.py --mode training_only
```

## AI Agent Roles

The system includes 5 specialized AI agents:

### 1. Inventory Intelligence Agent
- **Role**: Real-time inventory analysis and balance calculations
- **Capabilities**: Planning Balance calculation, shortage detection, multi-level netting
- **Authority**: Supervised execution

### 2. Forecast Intelligence Agent  
- **Role**: ML-based demand forecasting and predictive analytics
- **Capabilities**: ARIMA/Prophet/LSTM models, accuracy monitoring, ensemble optimization
- **Authority**: Autonomous execution

### 3. Production Planning Agent
- **Role**: Production scheduling and capacity optimization
- **Capabilities**: Schedule optimization, machine assignment, 6-phase planning
- **Authority**: Supervised execution

### 4. Yarn Substitution Agent
- **Role**: Intelligent yarn substitution and procurement optimization
- **Capabilities**: Compatibility analysis, quality impact assessment, supplier optimization
- **Authority**: Supervised execution

### 5. Quality Assurance Agent
- **Role**: Data quality monitoring and system performance
- **Capabilities**: Data validation, anomaly detection, compliance checking
- **Authority**: Recommendation only

## System Architecture

### Core Components

1. **Agent Base Classes** (`src/ai_agents/core/agent_base.py`)
   - Abstract base class for all agents
   - Message handling and communication protocols
   - Performance metrics and status tracking

2. **Central Orchestrator** (`src/ai_agents/core/orchestrator.py`)
   - Master coordinator for all agents
   - Task routing and load balancing
   - Implementation workflow management

3. **Agent Factory** (`src/ai_agents/deployment/agent_factory.py`)
   - Dynamic agent creation and lifecycle management
   - Resource-aware deployment strategies
   - Customer-specific agent provisioning

4. **Message Router** (`src/ai_agents/communication/message_router.py`)
   - Inter-agent communication infrastructure
   - Message queuing and delivery
   - Broadcast and capability-based routing

5. **Training Framework** (`src/agents/training_framework.py`)
   - Agent training orchestration
   - Performance evaluation and certification
   - Continuous learning and improvement

### Configuration

The system is configured via `config/agent_training_config.json`:

- **Global Settings**: Parallel training, resource allocation
- **Agent Configurations**: Specific settings for each agent type
- **Training Phases**: Knowledge acquisition through full autonomy
- **Deployment Strategy**: Shadow mode → Advisory → Supervised → Full autonomy
- **Monitoring & Alerting**: Performance thresholds and notifications

## API Endpoints

Once initialized, agents are accessible through these API endpoints:

- `/api/inventory-intelligence-enhanced` - Inventory operations
- `/api/ml-forecast-detailed` - Demand forecasting
- `/api/production-planning` - Production scheduling
- `/api/yarn-substitution-intelligent` - Yarn substitution
- `/api/data-validation` - Quality assurance

## Training & Deployment Phases

### Phase 1: Knowledge Acquisition (3 days)
- Load domain knowledge and business rules
- Parse data structures and API endpoints
- Success criteria: 100 knowledge items, 50 business rules

### Phase 2: Skill Development (7 days)
- Train on historical data
- Practice decision scenarios
- Success criteria: 85% accuracy, 200ms response time

### Phase 3: Integration Testing (3 days)
- Test API integration and data flow
- Validate error handling
- Success criteria: 95% API success rate

### Phase 4: Certification (1 day)
- Pass certification tests
- Meet performance benchmarks
- Success criteria: 90% test pass rate

### Deployment Phases
1. **Shadow Mode** (2 weeks): Run parallel to existing systems
2. **Advisory Mode** (2 weeks): Provide recommendations
3. **Supervised Autonomy** (2 weeks): Execute low-risk decisions
4. **Full Autonomy** (Ongoing): Complete autonomous operation

## Monitoring & Performance

### Key Performance Indicators

**Inventory Intelligence**:
- Planning Balance accuracy: 99.5%
- Shortage detection rate: 95%
- False positive rate: <5%

**Forecast Intelligence**:
- 30-day MAPE: <15%
- 90-day MAPE: <20%
- Forecast bias: <5%

**Production Planning**:
- On-time delivery: 95%
- Machine utilization: 85%
- Schedule adherence: 90%

**Yarn Substitution**:
- Substitution success: 90%
- Quality maintenance: 95%
- Cost optimization: 15%

**Quality Assurance**:
- Data accuracy: 99%
- Anomaly detection: 85%
- False alarm rate: <10%

### Health Monitoring

```bash
# Check system status
python init_agents.py --status

# View detailed metrics (when system is running)
curl http://localhost:5000/api/comprehensive-kpis
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed
   - Verify Python path includes `src/` directory

2. **Initialization Failures**
   - Check configuration file syntax
   - Verify database connections
   - Ensure sufficient system resources

3. **Agent Communication Issues**
   - Check message router status
   - Verify agent registration
   - Review system logs for errors

### Logs and Debugging

- System logs: Check console output during initialization
- Agent logs: Individual agent performance and errors
- Router logs: Message delivery and routing issues

### Recovery Procedures

1. **Soft Restart**: Stop and restart the agent system
2. **Agent Reset**: Destroy and recreate specific agents
3. **Full Reinitialization**: Complete system reset

```bash
# Soft restart
python init_agents.py --mode minimal
python init_agents.py --mode full

# Check status before/after
python init_agents.py --status
```

## Development & Testing

### Local Development

```bash
# Start in minimal mode for development
python init_agents.py --mode minimal

# Run with custom configuration
python init_agents.py --config dev_config.json

# Training only mode
python init_agents.py --mode training_only
```

### Testing

- Unit tests: `tests/unit/test_*.py`
- Integration tests: `tests/integration/test_*.py`
- Agent-specific tests: `tests/manual_tests/test_*.py`

## Support

For issues or questions:
1. Check this documentation
2. Review system logs
3. Check the project's issue tracker
4. Contact the development team

---

**Last Updated**: January 2025
**Version**: Beverly Knits ERP v2.0
