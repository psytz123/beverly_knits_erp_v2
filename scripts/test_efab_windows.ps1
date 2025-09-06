# PowerShell script to test eFab API from Windows
# Run this in PowerShell (not WSL) to test the connection

Write-Host "Testing eFab API Connection from Windows" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Test DNS resolution
Write-Host "`nTesting DNS resolution..." -ForegroundColor Yellow
try {
    $dns = Resolve-DnsName efab.bklapps.com -ErrorAction Stop
    Write-Host "✅ DNS resolved: $($dns.IPAddress)" -ForegroundColor Green
} catch {
    Write-Host "❌ DNS resolution failed: $_" -ForegroundColor Red
    Write-Host "Make sure you're on the corporate network/VPN" -ForegroundColor Yellow
    exit 1
}

# Test HTTPS connection
Write-Host "`nTesting HTTPS connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "https://efab.bklapps.com" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ HTTPS connection successful (Status: $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "⚠️ HTTPS connection issue: $_" -ForegroundColor Yellow
}

# Test API endpoint with session cookie
Write-Host "`nTesting API endpoint with session cookie..." -ForegroundColor Yellow
$headers = @{
    "Cookie" = "dancer.session=aLfHTrRrtWWy4FPgLnxdEPC7ohA37dlR"
}

try {
    $apiResponse = Invoke-RestMethod -Uri "https://efab.bklapps.com/api/sales-order/plan/list" -Headers $headers -TimeoutSec 10
    Write-Host "✅ API call successful!" -ForegroundColor Green
    Write-Host "Response contains $($apiResponse.Count) records" -ForegroundColor Cyan
    
    # Show first record as sample
    if ($apiResponse.Count -gt 0) {
        Write-Host "`nSample record:" -ForegroundColor Yellow
        $apiResponse[0] | ConvertTo-Json -Depth 2 | Write-Host
    }
} catch {
    Write-Host "❌ API call failed: $_" -ForegroundColor Red
    Write-Host "Session cookie may have expired. Get a fresh one from browser." -ForegroundColor Yellow
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "Test complete!" -ForegroundColor Green