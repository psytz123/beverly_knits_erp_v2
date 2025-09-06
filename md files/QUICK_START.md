# Beverly Knits ERP - Quick Start Guide

## 🚀 Your Application is Running!

Your Beverly Knits ERP is successfully deployed in Docker and ready for production use.

### 📍 Current Access Points

- **Local Access**: http://localhost:5005/consolidated
- **Network Access**: http://172.30.1.251:5005/consolidated
- **Container Status**: Running (`bki-erp-minimal`)

## 🌐 Next Steps for Production

### 1. Internet Access (5 minutes)
```bash
# Quick setup with ngrok
./setup_internet_access.sh

# Or manually:
ngrok http 5005
# Share the generated URL (e.g., https://abc123.ngrok.io)
```

### 2. Cloud Deployment (30 minutes)
```bash
# Interactive deployment wizard
./deploy-to-cloud.sh

# Recommended: Railway (simplest)
railway login
railway init
railway up
```

### 3. Monitoring & Backup Setup (10 minutes)
```bash
# Interactive monitoring setup
./monitoring-and-backup.sh

# Setup automated monitoring
./monitoring-and-backup.sh setup_cron

# Manual backup
./monitoring-and-backup.sh perform_backup
```

## 📊 Quick Management Commands

### Container Management
```bash
# View logs
docker logs -f bki-erp-minimal

# Restart application
docker-compose -f docker-compose.minimal.yml restart

# Stop application
docker-compose -f docker-compose.minimal.yml down

# Update and rebuild
git pull
docker-compose -f docker-compose.minimal.yml up -d --build
```

### Health Monitoring
```bash
# Check status
curl http://localhost:5005/api/comprehensive-kpis

# Monitor resources
docker stats bki-erp-minimal

# View recent activity
docker logs bki-erp-minimal --tail 100
```

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `.env.docker` | Environment variables |
| `docker-compose.minimal.yml` | Minimal deployment (current) |
| `docker-compose.yml` | Full deployment with database |
| `docker-compose.prod.yml` | Production with all services |
| `Dockerfile.minimal` | Lightweight image (current) |
| `Dockerfile.optimized` | Full ML features |

## 🚨 Troubleshooting

### Container won't start
```bash
# Check logs
docker logs bki-erp-minimal

# Check ports
netstat -tuln | grep 5005

# Restart Docker
docker-compose -f docker-compose.minimal.yml down
docker-compose -f docker-compose.minimal.yml up -d
```

### Data not loading
```bash
# Clear cache
docker exec bki-erp-minimal rm -rf /app/cache/*

# Force reload
curl http://localhost:5005/api/reload-data
```

### High memory usage
```bash
# Limit memory
docker update --memory="2g" bki-erp-minimal

# Or use production config with limits
docker-compose -f docker-compose.prod.yml up -d
```

## 📈 Performance Optimization

### For Better Performance
1. **Enable Redis caching**:
   ```bash
   docker-compose up -d  # Uses full stack with Redis
   ```

2. **Use production build**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Scale horizontally**:
   ```bash
   docker-compose up -d --scale erp-app=3
   ```

## 🔐 Security Checklist

- [ ] Change default passwords in `.env.docker`
- [ ] Generate new SECRET_KEY: `openssl rand -hex 32`
- [ ] Enable HTTPS (use nginx proxy or cloud provider)
- [ ] Configure firewall rules
- [ ] Set up regular backups
- [ ] Enable monitoring alerts

## 📝 Deployment Options Summary

| Platform | Difficulty | Cost | Best For |
|----------|------------|------|----------|
| **ngrok** | Easy | Free/$8 | Testing & demos |
| **Railway** | Very Easy | $5+ | Quick deployment |
| **Render** | Easy | Free/$7+ | Small teams |
| **DigitalOcean** | Medium | $24+ | Growing business |
| **AWS EC2** | Hard | $30+ | Enterprise |
| **Azure** | Hard | $40+ | Microsoft stack |

## 🎯 Recommended Production Path

1. **Now**: Test thoroughly with ngrok
2. **This Week**: Deploy to Railway/Render
3. **This Month**: Set up monitoring & backups
4. **Next Quarter**: Scale to DigitalOcean/AWS

## 📞 Getting Help

- **Documentation**: See `PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Logs**: Check `docker logs bki-erp-minimal`
- **API Health**: http://localhost:5005/api/comprehensive-kpis
- **Dashboard**: http://localhost:5005/consolidated

## ✅ Status Check

Run this command to verify everything is working:
```bash
echo "=== Beverly Knits ERP Status ==="
echo "Container: $(docker ps | grep bki-erp-minimal | awk '{print $7}')"
echo "API: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:5005/api/comprehensive-kpis)"
echo "Dashboard: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:5005/consolidated)"
echo "Logs: $(docker logs bki-erp-minimal --tail 1 2>&1 | cut -c1-50)..."
```

---

**Your Beverly Knits ERP is ready for production!** 🎉

Choose your deployment path above and follow the corresponding guide.