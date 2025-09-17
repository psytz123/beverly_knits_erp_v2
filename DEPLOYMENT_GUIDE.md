# Beverly Knits ERP v2 - Deployment Guide

Complete guide for deploying Beverly Knits ERP v2 in production environments, covering Docker deployment, API integration setup, and monitoring configuration.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements (Single Instance)
- **CPU**: 4 cores, 2.4 GHz
- **Memory**: 8 GB RAM
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps connection
- **OS**: Ubuntu 20.04 LTS, CentOS 8, or RHEL 8

#### Recommended Requirements (Production)
- **CPU**: 8 cores, 3.0 GHz
- **Memory**: 16 GB RAM
- **Storage**: 100 GB NVMe SSD
- **Network**: 10 Gbps connection
- **OS**: Ubuntu 22.04 LTS

#### Software Dependencies
- **Python**: 3.10+ (3.11 recommended)
- **Docker**: 20.10+ (recommended for deployment)
- **Redis**: 7.0+ (optional for caching)
- **Nginx**: 1.20+ (reverse proxy for production)

---

## 🚀 Quick Deployment (Development)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/beverly_knits_erp_v2.git
cd beverly_knits_erp_v2

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
nano .env  # Configure your eFab/QuadS credentials

# Start development server
python3 src/core/beverly_comprehensive_erp.py
```

### Docker Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/beverly_knits_erp_v2.git
cd beverly_knits_erp_v2

# Start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f beverly-erp
```

---

## 🏗️ Production Deployment

### Docker Production Deployment (Recommended)

#### 1. Infrastructure Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
sudo apt install -y docker.io docker-compose nginx git

# Create beverly user
sudo useradd -m -s /bin/bash beverly
sudo usermod -aG docker beverly
```

#### 2. Environment Configuration

Create production environment file:

```bash
# Create .env.production
cat > .env.production << 'EOF'
# eFab ERP Configuration
ERP_BASE_URL=https://efab.bkiapps.com
ERP_LOGIN_URL=https://efab.bkiapps.com/login
ERP_API_PREFIX=/api
ERP_USERNAME=psytz
ERP_PASSWORD=big$cat
EFAB_SESSION=aMdcwNLa0ov0pcbWcQ_zb5wyPLSkYF_B

# QuadS Configuration
QUADS_BASE_URL=https://quads.bkiapps.com
QUADS_LOGIN_URL=https://quads.bkiapps.com/LOGIN

# Session Management
SESSION_COOKIE_NAME=dancer.session
SESSION_STATE_PATH=/tmp/erp_session.json

# Beverly ERP Settings
FLASK_ENV=production
DEBUG=False
PORT=5006
HOST=0.0.0.0

# Yarn Demand Scheduler
ENABLE_YARN_SCHEDULER=true
FILTER_NONPRODUCTION_YARNS=true
SCHEDULER_INTERVAL_HOURS=2

# Performance Settings
WORKERS=4
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=100

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE_PATH=/app/logs/beverly_erp.log

# Security
SECRET_KEY=your_very_secure_secret_key_here
CORS_ORIGINS=*
RATE_LIMITING=true
EOF
```

#### 3. Docker Compose Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  beverly-erp:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "5006:5006"
    environment:
      - FLASK_ENV=production
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - /tmp:/tmp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5006/api/comprehensive-kpis"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      - redis
    networks:
      - beverly-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    networks:
      - beverly-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - beverly-erp
    restart: unless-stopped
    networks:
      - beverly-network

volumes:
  redis_data:

networks:
  beverly-network:
    driver: bridge
```

#### 4. Production Dockerfile

Create `Dockerfile.production`:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY web/ ./web/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user
RUN useradd -m -u 1000 beverly && chown -R beverly:beverly /app
USER beverly

# Expose port
EXPOSE 5006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5006/api/comprehensive-kpis || exit 1

# Start application
CMD ["python3", "src/core/beverly_comprehensive_erp.py"]
```

#### 5. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream beverly_erp {
        server beverly-erp:5006;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/certificate.crt;
        ssl_certificate_key /etc/nginx/ssl/private.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        client_max_body_size 100M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;

        location / {
            proxy_pass http://beverly_erp;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support for real-time updates
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /api/ {
            proxy_pass http://beverly_erp;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # API-specific timeouts
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }

        location /static/ {
            alias /app/web/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

#### 6. Deploy and Start Services

```bash
# Switch to beverly user
sudo su - beverly

# Clone and setup
git clone https://github.com/your-org/beverly_knits_erp_v2.git
cd beverly_knits_erp_v2

# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f beverly-erp
```

---

## 🔧 Configuration Management

### eFab API Integration

#### Session Management
The system requires valid eFab session cookies. Update these regularly:

```bash
# Check current session status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.efab_connection_status'

# Update session cookie
docker-compose -f docker-compose.prod.yml exec beverly-erp \
  python3 -c "
import os
os.environ['EFAB_SESSION'] = 'new_session_cookie_value'
print('Session updated')
"

# Restart service to apply new session
docker-compose -f docker-compose.prod.yml restart beverly-erp
```

#### API Endpoint Verification
Verify all wrapper endpoints are working:

```bash
# Test primary endpoints
curl -s http://localhost:5006/api/yarn/active | jq '.status'
curl -s http://localhost:5006/api/knitorder/list | jq '.status'
curl -s http://localhost:5006/api/sales-order/plan/list | jq '.status'
curl -s http://localhost:5006/api/styles | jq '.status'

# Test QuadS endpoints
curl -s http://localhost:5006/api/styles/greige/active | jq '.status'
curl -s http://localhost:5006/api/styles/finished/active | jq '.status'

# Test reporting endpoints
curl -s http://localhost:5006/api/report/yarn_demand | jq '.status'
curl -s http://localhost:5006/api/yarn-po | jq '.status'
```

### Yarn Demand Scheduler Configuration

The system includes automated yarn demand report downloading:

```bash
# Enable scheduler
export ENABLE_YARN_SCHEDULER=true
export FILTER_NONPRODUCTION_YARNS=true

# Manual refresh trigger
curl -X POST http://localhost:5006/api/manual-yarn-refresh

# Check scheduler status in logs
docker-compose -f docker-compose.prod.yml logs beverly-erp | grep SCHEDULER
```

---

## 📊 Monitoring & Observability

### Health Monitoring

#### System Health Endpoint
```bash
# Check overall system health
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.'
```

#### Docker Container Monitoring
```bash
# Monitor container resources
docker stats

# Check container health
docker-compose -f docker-compose.prod.yml ps

# View container logs
docker-compose -f docker-compose.prod.yml logs --tail=100 beverly-erp
```

### Performance Metrics

#### Key Performance Indicators
- **API Response Time**: <200ms for most endpoints
- **Data Load Time**: <2 seconds for inventory data
- **Cache Hit Rate**: >80% for optimal performance
- **Memory Usage**: <2GB per container
- **Yarn Shortage Detection**: Real-time updates

#### Monitoring Commands
```bash
# Check API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5006/api/inventory-intelligence-enhanced

# Monitor cache performance
curl -s http://localhost:5006/api/consolidation-metrics | jq '.cache_metrics'

# Check memory usage
docker exec beverly-erp ps aux --sort=-%mem | head -10
```

---

## 🔒 Security Configuration

### SSL/TLS Setup

#### Let's Encrypt Certificate (Recommended)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### API Security
- **Session-based authentication** via eFab cookies
- **CORS protection** configured for specific origins
- **Rate limiting** enabled for API endpoints
- **Secure headers** included in all responses

---

## 🔄 Backup & Recovery

### Data Backup Strategy

#### Automated Backup Script
```bash
#!/bin/bash
# backup_script.sh

BACKUP_DIR="/backup/beverly-erp"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application data
docker run --rm \
  -v beverly_knits_erp_v2_data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/data_backup_$DATE.tar.gz -C /data .

# Backup Redis data
docker exec beverly_knits_erp_v2_redis_1 redis-cli BGSAVE
docker cp beverly_knits_erp_v2_redis_1:/data/dump.rdb $BACKUP_DIR/redis_backup_$DATE.rdb

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete

echo "Backup completed: $DATE"
```

#### Automated Backup with Cron
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /home/beverly/backup_script.sh

# Weekly full system backup
0 1 * * 0 /home/beverly/full_backup_script.sh
```

### Disaster Recovery

#### Recovery Procedure
```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data
tar xzf /backup/data_backup_YYYYMMDD_HHMMSS.tar.gz -C ./data/

# Restore Redis data
docker-compose -f docker-compose.prod.yml up -d redis
docker cp /backup/redis_backup_YYYYMMDD_HHMMSS.rdb beverly_knits_erp_v2_redis_1:/data/dump.rdb
docker-compose -f docker-compose.prod.yml restart redis

# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Verify recovery
curl -s http://localhost:5006/api/comprehensive-kpis
```

---

## 📚 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats beverly-erp

# Restart service to clear memory
docker-compose -f docker-compose.prod.yml restart beverly-erp

# Optimize memory settings
# Edit docker-compose.prod.yml:
# mem_limit: 2g
# memswap_limit: 2g
```

#### Slow API Responses
```bash
# Check API performance
curl -w "%{time_total}" -o /dev/null -s http://localhost:5006/api/yarn-intelligence

# Enable Redis caching
# Verify Redis is running:
docker-compose -f docker-compose.prod.yml ps redis

# Check cache hit rates
curl -s http://localhost:5006/api/consolidation-metrics | jq '.cache_hit_rate'
```

#### eFab Session Expiration
```bash
# Check session status
curl -s http://localhost:5006/api/comprehensive-kpis | jq '.efab_session_status'

# Update session cookie
# 1. Login to eFab in browser
# 2. Copy session cookie from browser dev tools
# 3. Update environment variable:
docker-compose -f docker-compose.prod.yml exec beverly-erp \
  bash -c 'export EFAB_SESSION="new_session_value"'

# Restart service
docker-compose -f docker-compose.prod.yml restart beverly-erp
```

#### Yarn Scheduler Issues
```bash
# Check scheduler logs
docker-compose -f docker-compose.prod.yml logs beverly-erp | grep SCHEDULER

# Manual refresh
curl -X POST http://localhost:5006/api/manual-yarn-refresh

# Verify downloaded files
docker-compose -f docker-compose.prod.yml exec beverly-erp \
  ls -la /app/data/production/5/ERP\ Data/Expected_Yarn_Report.xlsx
```

---

## 📈 Scaling Guidelines

### Horizontal Scaling

#### Load Balancer Configuration
```bash
# Add multiple ERP instances
# Update docker-compose.prod.yml:
services:
  beverly-erp-1:
    # ... same config ...
    ports:
      - "5006:5006"

  beverly-erp-2:
    # ... same config ...
    ports:
      - "5007:5006"

  nginx:
    # Update upstream in nginx.conf:
    upstream beverly_erp {
        server beverly-erp-1:5006;
        server beverly-erp-2:5006;
    }
```

### Performance Optimization

#### Redis Cluster Setup
```yaml
# For high-availability Redis
services:
  redis-master:
    image: redis:7-alpine
    command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru

  redis-replica:
    image: redis:7-alpine
    command: redis-server --slaveof redis-master 6379
    depends_on:
      - redis-master
```

---

## ✅ Production Checklist

### Pre-Deployment
- [ ] eFab and QuadS credentials configured
- [ ] SSL certificates installed and validated
- [ ] Environment variables properly set
- [ ] Firewall rules configured
- [ ] Docker containers build successfully
- [ ] Health checks passing
- [ ] Backup procedures tested
- [ ] Monitoring configured

### Post-Deployment
- [ ] All API endpoints responding correctly
- [ ] eFab session authentication working
- [ ] Yarn demand scheduler active
- [ ] Cache performance optimized
- [ ] Logs being collected
- [ ] Backup schedules active
- [ ] Security monitoring enabled
- [ ] Performance baselines established

### API Endpoint Validation
```bash
# Validate all primary wrapper endpoints
endpoints=(
  "/api/yarn/active"
  "/api/knitorder/list"
  "/api/sales-order/plan/list"
  "/api/styles"
  "/api/styles/greige/active"
  "/api/styles/finished/active"
  "/api/report/yarn_demand"
  "/api/yarn-po"
)

for endpoint in "${endpoints[@]}"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5006$endpoint")
  echo "$endpoint: $status"
done
```

---

*Deployment Guide v2.0.0 - Beverly Knits ERP System - Updated September 2025*