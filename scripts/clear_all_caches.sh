#!/bin/bash
# Beverly Knits ERP - Cache Clearing Script
# Phase 1 Implementation - Comprehensive System Fix
# Created: 2025-09-02

echo "=================================================="
echo "Beverly Knits ERP - Cache Clearing Script"
echo "=================================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Define cache directories
CACHE_DIRS=(
    "/tmp/bki_cache"
    "/tmp/beverly_cache"
    "/tmp/yarn_cache"
    "$HOME/.cache/beverly_knits"
)

# Track statistics
TOTAL_FILES_CLEARED=0
TOTAL_SIZE_CLEARED=0

# Function to get directory size in bytes
get_dir_size() {
    if [ -d "$1" ]; then
        du -sb "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Clear each cache directory
for dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # Get size before clearing
        SIZE_BEFORE=$(get_dir_size "$dir")
        FILES_BEFORE=$(find "$dir" -type f 2>/dev/null | wc -l)
        
        echo "Clearing cache: $dir"
        echo "  Files: $FILES_BEFORE"
        echo "  Size: $(numfmt --to=iec-i --suffix=B $SIZE_BEFORE 2>/dev/null || echo "${SIZE_BEFORE} bytes")"
        
        # Clear the cache
        rm -rf "$dir"/*
        
        # Update statistics
        TOTAL_FILES_CLEARED=$((TOTAL_FILES_CLEARED + FILES_BEFORE))
        TOTAL_SIZE_CLEARED=$((TOTAL_SIZE_CLEARED + SIZE_BEFORE))
        
        echo "  ✓ Cleared"
    else
        echo "  ⚠ Directory not found: $dir"
    fi
    echo ""
done

# Clear Python cache
echo "Clearing Python cache..."
PYTHON_CACHE_COUNT=$(find /mnt/c/finalee/beverly_knits_erp_v2 -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ $PYTHON_CACHE_COUNT -gt 0 ]; then
    find /mnt/c/finalee/beverly_knits_erp_v2 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "  ✓ Cleared $PYTHON_CACHE_COUNT __pycache__ directories"
else
    echo "  ℹ No Python cache directories found"
fi
echo ""

# Clear Redis if available
if command -v redis-cli &> /dev/null; then
    echo "Clearing Redis cache..."
    
    # Check if Redis is running
    if redis-cli ping &> /dev/null; then
        # Get info before clearing
        REDIS_KEYS=$(redis-cli DBSIZE | awk '{print $2}')
        
        # Clear all databases
        redis-cli FLUSHALL &> /dev/null
        
        echo "  ✓ Redis cache cleared ($REDIS_KEYS keys removed)"
    else
        echo "  ⚠ Redis is not running"
    fi
else
    echo "ℹ Redis not installed - skipping"
fi
echo ""

# Clear application-specific caches
APP_CACHE_DIR="/mnt/c/finalee/beverly_knits_erp_v2/cache"
if [ -d "$APP_CACHE_DIR" ]; then
    echo "Clearing application cache..."
    SIZE_BEFORE=$(get_dir_size "$APP_CACHE_DIR")
    FILES_BEFORE=$(find "$APP_CACHE_DIR" -type f 2>/dev/null | wc -l)
    
    rm -rf "$APP_CACHE_DIR"/*
    
    TOTAL_FILES_CLEARED=$((TOTAL_FILES_CLEARED + FILES_BEFORE))
    TOTAL_SIZE_CLEARED=$((TOTAL_SIZE_CLEARED + SIZE_BEFORE))
    
    echo "  ✓ Cleared $FILES_BEFORE files"
fi

# Summary
echo "=================================================="
echo "Cache Clearing Complete!"
echo "=================================================="
echo "Total files cleared: $TOTAL_FILES_CLEARED"
if [ $TOTAL_SIZE_CLEARED -gt 0 ]; then
    echo "Total space freed: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE_CLEARED 2>/dev/null || echo "${TOTAL_SIZE_CLEARED} bytes")"
fi
echo ""

# Optional: Restart suggestion
echo "Recommendation:"
echo "  Restart the Beverly Knits ERP server to ensure clean state:"
echo "  1. Stop server: pkill -f 'python3.*beverly'"
echo "  2. Clear cache: ./scripts/clear_all_caches.sh"
echo "  3. Start server: python3 src/core/beverly_comprehensive_erp.py"
echo ""
echo "Script completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# Exit with success
exit 0