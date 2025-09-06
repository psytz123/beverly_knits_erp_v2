# AI Supply Chain Framework - Deployment Guide

Complete guide for deploying the AI Supply Chain Optimization Framework in production environments, covering single-tenant and multi-tenant deployments across various infrastructure platforms.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements (Single Tenant)
- **CPU**: 4 cores, 2.4 GHz
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps connection
- **OS**: Ubuntu 20.04 LTS, CentOS 8, or RHEL 8

#### Recommended Requirements (Multi-Tenant)
- **CPU**: 16 cores, 3.0 GHz
- **Memory**: 64 GB RAM
- **Storage**: 500 GB NVMe SSD
- **Network**: 10 Gbps connection
- **OS**: Ubuntu 22.04 LTS

#### Software Dependencies
- **Python**: 3.9+ (3.11 recommended)
- **PostgreSQL**: 14+ (15 recommended)
- **Redis**: 7.0+
- **Docker**: 20.10+ (optional)
- **Kubernetes**: 1.24+ (for orchestrated deployments)
- **Nginx**: 1.20+ (reverse proxy)

---

## 🚀 Quick Deployment (Development)

### Local Development Setup

```bash
# Clone the framework
git clone https://github.com/your-org/ai-supply-chain-framework.git
cd ai-supply-chain-framework

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
nano .env  # Configure your settings

# Initialize database
python -m framework db init
python -m framework db migrate

# Start development server
python -m framework run --dev
```

### Docker Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-supply-chain-framework.git
cd ai-supply-chain-framework

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f framework
```

---

## 🏗️ Production Deployment

### Single-Tenant Production Deployment

#### 1. Infrastructure Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql-14 redis-server nginx git

# Create framework user
sudo useradd -m -s /bin/bash framework
sudo usermod -aG sudo framework
```

#### 2. Database Configuration

```sql
-- Connect to PostgreSQL as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE ai_supply_chain;
CREATE USER framework WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_supply_chain TO framework;

-- Enable required extensions
\c ai_supply_chain
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

\q
```

#### 3. Redis Configuration

```bash
# Configure Redis
sudo nano /etc/redis/redis.conf

# Key settings:
# maxmemory 4gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

#### 4. Framework Installation

```bash
# Switch to framework user
sudo su - framework

# Clone and setup
git clone https://github.com/your-org/ai-supply-chain-framework.git
cd ai-supply-chain-framework

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-prod.txt

# Configure environment
cp .env.production .env
nano .env
```

#### 5. Environment Configuration

```bash
# .env file
DATABASE_URL=postgresql://framework:your_secure_password@localhost/ai_supply_chain
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your_very_secure_secret_key_here
FLASK_ENV=production
LOG_LEVEL=INFO
WORKERS=4

# Security settings
SSL_REQUIRED=True
CSRF_PROTECTION=True
RATE_LIMITING=True

# Framework settings
DEFAULT_INDUSTRY=GENERIC_MANUFACTURING
MAX_CONCURRENT_IMPLEMENTATIONS=10
CACHE_TTL=3600

# External integrations
SAP_CONNECTOR_ENABLED=True
ORACLE_CONNECTOR_ENABLED=True
QUICKBOOKS_CONNECTOR_ENABLED=True

# Monitoring
PROMETHEUS_ENABLED=True
GRAFANA_ENABLED=True
LOG_TO_FILE=True
```

#### 6. Database Migration

```bash
# Initialize database
python -m framework db init

# Run migrations
python -m framework db migrate

# Create initial data
python -m framework db seed --production

# Verify setup
python -m framework db verify
```

#### 7. Systemd Service Configuration

```bash
# Create systemd service
sudo nano /etc/systemd/system/ai-supply-chain.service
```

```ini
[Unit]
Description=AI Supply Chain Optimization Framework
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=framework
Group=framework
WorkingDirectory=/home/framework/ai-supply-chain-framework
Environment=PATH=/home/framework/ai-supply-chain-framework/venv/bin
ExecStart=/home/framework/ai-supply-chain-framework/venv/bin/python -m framework run --production
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=5

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/home/framework/ai-supply-chain-framework/logs
ReadWritePaths=/home/framework/ai-supply-chain-framework/uploads

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-supply-chain
sudo systemctl start ai-supply-chain

# Check status
sudo systemctl status ai-supply-chain
```

#### 8. Nginx Configuration

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/ai-supply-chain
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/ssl/certificate.crt;
    ssl_certificate_key /path/to/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /static/ {
        alias /home/framework/ai-supply-chain-framework/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    location /uploads/ {
        alias /home/framework/ai-supply-chain-framework/uploads/;
        expires 1d;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ai-supply-chain /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## ☸️ Kubernetes Deployment

### Multi-Tenant Production with Kubernetes

#### 1. Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-supply-chain
  labels:
    name: ai-supply-chain
```

#### 2. Database Configuration

```yaml
# postgres-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: ai-supply-chain
type: Opaque
data:
  username: ZnJhbWV3b3Jr  # framework (base64)
  password: eW91cl9zZWN1cmVfcGFzc3dvcmQ=  # your_secure_password (base64)

---
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ai-supply-chain
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: "ai_supply_chain"
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: ai-supply-chain
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
```

#### 3. Redis Configuration

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ai-supply-chain
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: ai-supply-chain
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
```

#### 4. Framework Application

```yaml
# framework-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: framework-config
  namespace: ai-supply-chain
data:
  FLASK_ENV: "production"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
  MAX_CONCURRENT_IMPLEMENTATIONS: "50"
  CACHE_TTL: "3600"
  PROMETHEUS_ENABLED: "true"

---
# framework-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: framework-secret
  namespace: ai-supply-chain
type: Opaque
data:
  secret_key: eW91cl92ZXJ5X3NlY3VyZV9zZWNyZXRfa2V5X2hlcmU=
  database_url: cG9zdGdyZXNxbDovL2ZyYW1ld29yazp5b3VyX3NlY3VyZV9wYXNzd29yZEBwb3N0Z3Jlcy1zZXJ2aWNlL2FpX3N1cHBseV9jaGFpbg==

---
# framework-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: framework
  namespace: ai-supply-chain
spec:
  replicas: 3
  selector:
    matchLabels:
      app: framework
  template:
    metadata:
      labels:
        app: framework
    spec:
      containers:
      - name: framework
        image: ai-supply-chain-framework:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: framework-secret
              key: database_url
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: framework-secret
              key: secret_key
        envFrom:
        - configMapRef:
            name: framework-config
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

---
apiVersion: v1
kind: Service
metadata:
  name: framework-service
  namespace: ai-supply-chain
spec:
  selector:
    app: framework
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

#### 5. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: framework-ingress
  namespace: ai-supply-chain
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: framework-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: framework-service
            port:
              number: 80
```

#### 6. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: framework-hpa
  namespace: ai-supply-chain
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: framework
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 7. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f postgres-secret.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f framework-configmap.yaml
kubectl apply -f framework-secret.yaml
kubectl apply -f framework-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get all -n ai-supply-chain

# Check logs
kubectl logs -f deployment/framework -n ai-supply-chain
```

---

## 🏢 Multi-Tenant Configuration

### Tenant Isolation Strategy

#### Database-per-Tenant (Recommended)

```python
# tenant_config.py
TENANT_CONFIG = {
    "isolation_level": "database",  # database, schema, row
    "auto_provisioning": True,
    "max_tenants_per_cluster": 100,
    "tenant_resources": {
        "small": {
            "max_users": 25,
            "max_implementations": 3,
            "storage_gb": 10
        },
        "medium": {
            "max_users": 100, 
            "max_implementations": 10,
            "storage_gb": 50
        },
        "large": {
            "max_users": 500,
            "max_implementations": 25,
            "storage_gb": 200
        }
    }
}
```

#### Tenant Provisioning

```bash
# Create new tenant
python -m framework tenant create \
  --tenant-id "customer_123" \
  --plan "medium" \
  --industry "FURNITURE" \
  --admin-email "admin@customer.com"

# Verify tenant
python -m framework tenant verify --tenant-id "customer_123"

# List tenants
python -m framework tenant list --active-only
```

---

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: ai-supply-chain
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'framework'
      static_configs:
      - targets: ['framework-service:80']
      metrics_path: '/metrics'
      scrape_interval: 10s
    - job_name: 'postgres'
      static_configs:
      - targets: ['postgres-exporter:9187']
    - job_name: 'redis'
      static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AI Supply Chain Framework",
    "panels": [
      {
        "title": "Active Implementations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "framework_active_implementations_total",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Success Rate",
        "type": "singlestat", 
        "targets": [
          {
            "expr": "rate(framework_implementations_success_total[1h]) / rate(framework_implementations_total[1h])",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, framework_request_duration_seconds_bucket)",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

---

## 🔒 Security Hardening

### Application Security

```python
# security_config.py
SECURITY_CONFIG = {
    # Authentication
    "jwt_secret_rotation": True,
    "jwt_expiry_hours": 24,
    "mfa_required": True,
    
    # Authorization
    "rbac_enabled": True,
    "api_key_rotation_days": 90,
    
    # Data Protection
    "encryption_at_rest": True,
    "encryption_algorithm": "AES-256-GCM",
    "field_level_encryption": ["passwords", "api_keys", "pii"],
    
    # Network Security
    "tls_version": "1.3",
    "hsts_enabled": True,
    "csrf_protection": True,
    
    # Rate Limiting
    "rate_limiting": {
        "per_ip": "1000/hour",
        "per_user": "5000/hour",
        "per_tenant": "50000/hour"
    },
    
    # Audit Logging
    "audit_all_requests": True,
    "log_retention_days": 90,
    "log_encryption": True
}
```

### Network Security

```bash
# Firewall rules (Ubuntu UFW)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL (internal)
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis (internal)
sudo ufw enable
```

---

## 🔧 Performance Tuning

### PostgreSQL Optimization

```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Application Performance

```python
# performance_config.py
PERFORMANCE_CONFIG = {
    # Caching
    "redis_max_connections": 100,
    "cache_default_ttl": 3600,
    "query_cache_enabled": True,
    
    # Database
    "db_pool_size": 20,
    "db_max_overflow": 30,
    "db_pool_timeout": 30,
    
    # Async Processing
    "celery_workers": 8,
    "celery_max_tasks_per_child": 1000,
    
    # Resource Limits
    "max_file_upload_mb": 100,
    "request_timeout_seconds": 300,
    "max_concurrent_implementations": 50
}
```

---

## 🔄 Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# backup_script.sh

BACKUP_DIR="/backup/ai-supply-chain"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="ai_supply_chain"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U framework -d $DB_NAME -f $BACKUP_DIR/db_backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/db_backup_$DATE.sql

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://your-backup-bucket/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### Automated Backup with Cron

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /home/framework/backup_script.sh

# Weekly full backup at 1 AM Sunday
0 1 * * 0 /home/framework/full_backup_script.sh
```

---

## 📚 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# PostgreSQL memory tuning
sudo nano /etc/postgresql/14/main/postgresql.conf
# Adjust shared_buffers, work_mem, maintenance_work_mem
```

#### Slow API Responses
```bash
# Check database queries
sudo -u postgres psql ai_supply_chain -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Check Redis performance
redis-cli --latency -i 1

# Application profiling
python -m framework profile --duration 60
```

#### Agent Communication Issues
```bash
# Check Redis connectivity
redis-cli ping

# Check message queue
python -m framework queue status

# Restart agents
python -m framework agents restart
```

---

## 📈 Scaling Guidelines

### Vertical Scaling Triggers
- CPU utilization > 70% for 15+ minutes
- Memory utilization > 80% for 10+ minutes
- Database connection pool > 80% utilized
- Response time > 500ms for 95th percentile

### Horizontal Scaling Triggers
- Active implementations > 40 per instance
- Queue depth > 100 messages
- Request rate > 1000 requests/minute per instance
- Success prediction accuracy < 90%

### Auto-scaling Configuration

```yaml
# kubernetes-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: framework-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: framework
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: active_implementations
      target:
        type: AverageValue
        averageValue: "30"
```

---

## ✅ Production Checklist

### Pre-Deployment
- [ ] Infrastructure provisioned and tested
- [ ] SSL certificates installed and validated
- [ ] Database migrations applied successfully
- [ ] Environment variables configured
- [ ] Security hardening applied
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security scanning performed
- [ ] Documentation updated

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Alerts configured and tested
- [ ] Performance baselines established
- [ ] Backup schedules active
- [ ] Security monitoring enabled
- [ ] Team access provisioned
- [ ] Runbook created
- [ ] Disaster recovery plan tested

---

*Deployment Guide v1.0.0 - AI Supply Chain Optimization Framework*