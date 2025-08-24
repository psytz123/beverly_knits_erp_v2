# Beverly Knits ERP - Production Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Internet Access (ngrok)](#internet-access-ngrok)
3. [Cloud Deployment](#cloud-deployment)
4. [Security Configuration](#security-configuration)
5. [Monitoring & Maintenance](#monitoring--maintenance)

## Quick Start

### Current Setup Status
Your Beverly Knits ERP is running in Docker on your local machine:
- **Local Access**: http://localhost:5005/consolidated
- **Network Access**: http://[YOUR-IP]:5005/consolidated
- **Container**: `bki-erp-minimal`

### Basic Docker Commands
```bash
# View logs
docker logs -f bki-erp-minimal

# Stop application
docker-compose -f docker-compose.minimal.yml down

# Restart application
docker-compose -f docker-compose.minimal.yml restart

# Rebuild after changes
docker-compose -f docker-compose.minimal.yml up -d --build
```

## Internet Access (ngrok)

### Option 1: Quick Setup (Temporary URL)
```bash
# Install ngrok (Windows)
winget install ngrok.ngrok

# Or download from
https://ngrok.com/download

# Sign up for free account
https://ngrok.com/signup

# Configure auth token
ngrok authtoken YOUR_TOKEN

# Start tunnel
ngrok http 5005

# Share the generated URL (e.g., https://abc123.ngrok.io)
```

### Option 2: Custom Domain (Paid)
```bash
# Reserve custom domain
ngrok http 5005 --domain=erp.yourcompany.com
```

## Cloud Deployment

### Option 1: DigitalOcean (Recommended for Simplicity)

1. **Create Droplet**
```bash
# Specs: 4GB RAM, 2 vCPUs, 80GB SSD (~$24/month)
# OS: Ubuntu 22.04 LTS
# Enable backups
```

2. **Initial Setup**
```bash
# SSH into droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin
```

3. **Deploy Application**
```bash
# Clone your repository
git clone https://github.com/yourusername/beverly-knits-erp.git
cd beverly-knits-erp/BKI_comp

# Copy environment file
cp .env.docker .env

# Edit environment variables
nano .env

# Start application
docker-compose -f docker-compose.prod.yml up -d
```

4. **Configure Domain & SSL**
```bash
# Install Nginx & Certbot
apt install nginx certbot python3-certbot-nginx

# Configure Nginx
nano /etc/nginx/sites-available/erp

# Add configuration:
server {
    server_name erp.yourdomain.com;
    
    location / {
        proxy_pass http://localhost:5005;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Enable site
ln -s /etc/nginx/sites-available/erp /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Get SSL certificate
certbot --nginx -d erp.yourdomain.com
```

### Option 2: AWS EC2

1. **Launch EC2 Instance**
```bash
# Instance type: t3.medium (2 vCPU, 4GB RAM)
# AMI: Amazon Linux 2023
# Storage: 30GB gp3
# Security Group: Allow 80, 443, 22, 5005
```

2. **Install Dependencies**
```bash
# Connect via SSH
ssh -i your-key.pem ec2-user@your-instance-ip

# Install Docker
sudo yum update -y
sudo yum install docker git -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **Deploy & Configure**
```bash
# Clone and deploy (same as DigitalOcean)
# Configure Application Load Balancer for HTTPS
# Use Route 53 for domain management
```

### Option 3: Azure Container Instances

1. **Create Resource Group**
```bash
az group create --name BeverlyKnitsRG --location eastus
```

2. **Create Container Registry**
```bash
az acr create --resource-group BeverlyKnitsRG \
  --name beverlyknitsacr --sku Basic
```

3. **Push Image to Registry**
```bash
# Build and tag image
docker build -t beverlyknitsacr.azurecr.io/bki-erp:latest .

# Login to ACR
az acr login --name beverlyknitsacr

# Push image
docker push beverlyknitsacr.azurecr.io/bki-erp:latest
```

4. **Deploy Container Instance**
```bash
az container create \
  --resource-group BeverlyKnitsRG \
  --name bki-erp \
  --image beverlyknitsacr.azurecr.io/bki-erp:latest \
  --cpu 2 --memory 4 \
  --ports 5005 \
  --dns-name-label beverly-knits-erp
```

## Security Configuration

### Essential Security Steps

1. **Environment Variables**
```bash
# Generate secure keys
openssl rand -hex 32  # For SECRET_KEY
openssl rand -hex 32  # For JWT_SECRET_KEY

# Update .env file
SECRET_KEY=your-generated-key
JWT_SECRET_KEY=your-generated-jwt-key
ADMIN_TOKEN=your-secure-admin-token
```

2. **Firewall Configuration**
```bash
# UFW (Ubuntu/Debian)
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP
ufw allow 443/tcp    # HTTPS
ufw enable
```

3. **Database Security**
```sql
-- Create application user (don't use root)
CREATE USER 'bki_app'@'localhost' IDENTIFIED BY 'secure_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON beverly_knits.* TO 'bki_app'@'localhost';
```

4. **API Rate Limiting**
```python
# Already configured in .env
API_RATE_LIMIT=100/minute
```

## Monitoring & Maintenance

### 1. Health Monitoring

```bash
# Create health check script
cat > /opt/bki/health_check.sh << 'EOF'
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5005/api/comprehensive-kpis)
if [ $response -eq 200 ]; then
    echo "Health check passed"
else
    echo "Health check failed"
    # Restart container
    docker-compose restart erp-app
fi
EOF

# Add to crontab (every 5 minutes)
*/5 * * * * /opt/bki/health_check.sh
```

### 2. Automated Backups

```bash
# Create backup script
cat > /opt/bki/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup database
docker exec bki-postgres pg_dump -U bki_admin beverly_knits_erp | gzip > $BACKUP_DIR/database.sql.gz

# Backup data files
tar -czf $BACKUP_DIR/data_files.tar.gz /mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP\ Data/

# Keep only last 7 days
find /backups -type d -mtime +7 -exec rm -rf {} \;
EOF

# Schedule daily backup at 2 AM
0 2 * * * /opt/bki/backup.sh
```

### 3. Log Management

```bash
# View logs
docker logs bki-erp-minimal --tail 100 --follow

# Export logs
docker logs bki-erp-minimal > /var/log/bki-erp/app_$(date +%Y%m%d).log

# Log rotation (add to /etc/logrotate.d/bki-erp)
/var/log/bki-erp/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 root adm
}
```

### 4. Performance Monitoring

```bash
# Monitor resource usage
docker stats bki-erp-minimal

# Check disk usage
df -h
du -sh /var/lib/docker/

# Monitor network
netstat -tuln | grep 5005
```

## Scaling Considerations

### Horizontal Scaling (Multiple Instances)
```yaml
# docker-compose.scale.yml
services:
  erp-app:
    scale: 3  # Run 3 instances
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
```

### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_yarn_inventory_desc ON yarn_inventory(Desc#);
CREATE INDEX idx_sales_activity_style ON sales_activity(Style#);
CREATE INDEX idx_bom_style ON style_bom(Style#);
```

### Caching Strategy
- Redis cache already configured
- Cache TTL: 5-15 minutes for inventory data
- Use CDN for static assets

## Troubleshooting

### Common Issues

1. **Container won't start**
```bash
# Check logs
docker logs bki-erp-minimal
# Check disk space
df -h
# Verify ports
netstat -tuln | grep 5005
```

2. **Data not loading**
```bash
# Clear cache
docker exec bki-erp-minimal rm -rf /app/cache/*
# Restart container
docker-compose restart
```

3. **High memory usage**
```bash
# Limit container memory
docker update --memory="2g" --memory-swap="2g" bki-erp-minimal
```

## Support & Documentation

- **Application Logs**: `/app/logs/`
- **Data Path**: `/mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/ERP Data/5/`
- **Cache Directory**: `/app/cache/`
- **Configuration**: `.env` file

## Quick Commands Reference

```bash
# Start application
docker-compose -f docker-compose.minimal.yml up -d

# Stop application
docker-compose -f docker-compose.minimal.yml down

# View logs
docker logs -f bki-erp-minimal

# Enter container shell
docker exec -it bki-erp-minimal bash

# Check API health
curl http://localhost:5005/api/comprehensive-kpis

# Restart application
docker-compose -f docker-compose.minimal.yml restart

# Update application
git pull
docker-compose -f docker-compose.minimal.yml up -d --build

# Backup database
docker exec bki-postgres pg_dump -U bki_admin beverly_knits_erp > backup.sql

# Restore database
docker exec -i bki-postgres psql -U bki_admin beverly_knits_erp < backup.sql
```

## Next Steps

1. **Immediate**: Test application thoroughly
2. **Short-term**: Set up monitoring and backups
3. **Medium-term**: Configure SSL and domain
4. **Long-term**: Implement auto-scaling and load balancing

---

For additional support or questions, refer to the main documentation or contact the development team.