# Fix Railway "Cannot create code snapshot" Error

## Solutions (Try in Order):

### Solution 1: Clear Railway Cache
```bash
railway unlink
railway link -p believable-rejoicing
railway up
```

### Solution 2: Use GitHub Integration
1. Push your code to GitHub:
   ```bash
   git push origin main
   ```
2. In Railway Dashboard:
   - Delete the current service
   - Create new service → "Deploy from GitHub Repo"
   - Select your repository
   - Railway will auto-detect Dockerfile.railway

### Solution 3: Deploy Without Config Files
1. Temporarily rename config files:
   ```bash
   mv railway.json railway.json.bak
   mv railway.toml railway.toml.bak
   ```
2. Deploy with Nixpacks:
   ```bash
   railway up
   ```

### Solution 4: Use Dockerfile Directly
```bash
# Build locally first
docker build -f Dockerfile.railway -t bki-erp .

# Then deploy
railway up --docker
```

### Solution 5: Create New Railway Project
```bash
railway logout
railway login
railway init
railway up
```

## Common Causes:
- Large files in directory (we've already fixed this)
- Git issues (try `git gc` to clean up)
- Railway platform temporary issues
- Conflicting configuration files

## Files are Correct:
✅ railway.json - Using DOCKERFILE builder
✅ railway.toml - Proper TOML format
✅ Dockerfile.railway - Exists and valid
✅ ERP Data - Only 892KB included
