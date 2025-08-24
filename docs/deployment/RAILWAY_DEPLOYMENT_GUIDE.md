# 🚂 Beverly Knits ERP - Railway Deployment Guide

## 📋 Prerequisites

- [ ] Railway account (sign up at https://railway.app)
- [ ] Railway CLI installed (already installed ✅)
- [ ] GitHub account (optional but recommended)

## 🚀 Quick Deployment (5 Minutes)

### Option 1: Automated Script (Recommended)

```bash
# Run the deployment script
./deploy-to-railway.sh
```

The script will:

1. ✅ Check Railway CLI installation
2. ✅ Login to Railway
3. ✅ Initialize project
4. ✅ Configure environment variables
5. ✅ Deploy application
6. ✅ Provide deployment URL

### Option 2: Manual Steps

#### Step 1: Login to Railway

```bash
railway login
```

#### Step 2: Initialize Project

```bash


# In the BKI_comp directory
cd /mnt/d/Agent-MCP-1-ddd/Agent-MCP-1-dd/BKI_comp

# Initialize new Railway project
railway init
```

#### Step 3: Configure Environment Variables

```bash
# Set required environment variables
railway variables set PORT=5005
railway variables set FLASK_ENV=production
railway variables set SECRET_KEY="$(openssl rand -hex 32)"
railway variables set ADMIN_TOKEN="$(openssl rand -hex 16)"
railway variables set ENABLE_UNIFIED_MCP=true
railway variables set MOCK_ERP_DATA=false
railway variables set ML_CONFIDENCE_THRESHOLD=0.85
railway variables set LOG_LEVEL=INFO
```

#### Step 4: Deploy

```bash
# Deploy to Railway
railway up

# This will:
# 1. Build Docker image using Dockerfile.railway
# 2. Push to Railway's registry
# 3. Deploy and start the application
```

#### Step 5: Get Your URL

```bash
# Open Railway dashboard
railway open

# Or get status
railway status
```

## 🔧 Configuration Files

### Files Already Created for You:

- ✅ `railway.json` - Railway configuration
- ✅ `railway.toml` - Additional settings
- ✅ `Dockerfile.railway` - Optimized Docker image
- ✅ `.gitignore` - Excludes unnecessary files

## 🌐 Post-Deployment

### 1. Access Your Application

Your app will be available at:

```
https://[your-app-name].railway.app
```

Access points:

- Dashboard: `https://[your-app-name].railway.app/consolidated`
- API: `https://[your-app-name].railway.app/api/comprehensive-kpis`

### 2. Monitor Your Application

```bash
# View logs
railway logs

# View logs with follow
railway logs -f

# Check deployment status
railway status
```

### 3. Configure Custom Domain (Optional)

1. Go to Railway dashboard
2. Click on your project
3. Go to Settings → Domains
4. Add custom domain
5. Update DNS records as instructed

## 📊 Resource Monitoring

### Check Usage

```bash
# Open Railway dashboard
railway open
```

In the dashboard you can see:

- Memory usage
- CPU usage
- Network traffic
- Build logs
- Application logs

### Railway Pricing

- **Hobby Plan**: $5/month (includes $5 of usage)
- **Pro Plan**: $20/month (includes $20 of usage)
- **Usage**: ~$0.01/hour for this application

Estimated monthly cost: **$7-10**

## 🔄 Updates and Redeployment

### Update Code and Redeploy

```bash
# Make your changes
# Then redeploy
railway up
```

### Rollback to Previous Version

```bash
# View deployment history in dashboard
railway open

# Or use CLI to redeploy specific version
railway redeploy [deployment-id]
```

## 🛠️ Troubleshooting

### Application Not Starting

```bash
# Check logs
railway logs

# Common issues:
# 1. Port binding - ensure PORT env var is set
# 2. Missing dependencies - check requirements.txt
# 3. Environment variables - verify all are set
```

### High Memory Usage

```bash
# Set resource limits in railway.json
{
  "deploy": {
    "maxMemory": "2048"
  }
}
```

### Slow Performance

1. Enable caching (Redis addon)
2. Optimize Docker image
3. Scale to multiple instances (Pro plan)

## 🔐 Security

### Environment Variables Set:

- ✅ SECRET_KEY (auto-generated)
- ✅ ADMIN_TOKEN (auto-generated)
- ✅ Database credentials (if using Railway PostgreSQL)

### Additional Security Steps:

1. Enable HTTPS (automatic on Railway)
2. Set up monitoring alerts
3. Regular backups
4. Use Railway's built-in DDoS protection

## 📈 Scaling

### Horizontal Scaling (Pro Plan)

```json
// In railway.json
{
  "deploy": {
    "numReplicas": 3
  }
}
```

### Add Database (Optional)

```bash
# Add PostgreSQL
railway add postgresql

# Get database URL
railway variables get DATABASE_URL
```

### Add Redis Cache (Optional)

```bash
# Add Redis
railway add redis

# Get Redis URL
railway variables get REDIS_URL
```

## 📝 Useful Railway Commands

```bash
# Basic Commands
railway login          # Login to Railway
railway logout         # Logout
railway whoami         # Check logged in user

# Project Management
railway init           # Initialize new project
railway open           # Open project in browser
railway status         # Check deployment status
railway list           # List all projects

# Deployment
railway up             # Deploy current directory
railway down           # Remove deployment
railway redeploy       # Redeploy latest version

# Environment Variables
railway variables      # List all variables
railway variables set KEY=value  # Set variable
railway variables get KEY        # Get variable value

# Logs and Debugging
railway logs           # View logs
railway logs -f        # Follow logs
railway run [command]  # Run command in Railway environment

# Database
railway connect        # Connect to database
railway add postgresql # Add PostgreSQL service
railway add redis      # Add Redis service
```

## ✅ Deployment Checklist

- [X] Railway CLI installed
- [X] Configuration files created
- [X] Dockerfile optimized for Railway
- [ ] Login to Railway (`railway login`)
- [ ] Initialize project (`railway init`)
- [ ] Set environment variables
- [ ] Deploy application (`railway up`)
- [ ] Verify deployment
- [ ] Test endpoints
- [ ] Set up monitoring

## 🎉 Success Indicators

Your deployment is successful when:

1. ✅ Railway shows "Deployment live"
2. ✅ Logs show "Running on http://0.0.0.0:5005"
3. ✅ Health check returns 200
4. ✅ Dashboard loads correctly
5. ✅ API endpoints respond with data

## 🆘 Getting Help

- **Railway Documentation**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Status Page**: https://status.railway.app
- **Support**: support@railway.app

---

**Ready to deploy?** Run `./deploy-to-railway.sh` and your application will be live in minutes!
