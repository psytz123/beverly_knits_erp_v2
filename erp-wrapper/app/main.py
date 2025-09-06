from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .erp_client import (
    get_sales_orders,
    get_knit_orders,
    get_inventory,
    get_styles,
    test_connection,
    session_mgr
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="eFab ERP Wrapper API",
    description="Proxy service for eFab ERP with automatic session management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)

# Cache for recent data
data_cache = {
    "sales_orders": {"data": None, "timestamp": None},
    "knit_orders": {"data": None, "timestamp": None},
    "inventory": {"data": None, "timestamp": None},
    "styles": {"data": None, "timestamp": None}
}
CACHE_TTL_SECONDS = 300  # 5 minutes

def is_cache_valid(cache_entry: Dict) -> bool:
    """Check if cache entry is still valid."""
    if cache_entry["data"] is None or cache_entry["timestamp"] is None:
        return False
    age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
    return age < CACHE_TTL_SECONDS

@app.on_event("startup")
async def startup_event():
    """Initialize the session on startup."""
    logger.info("Starting eFab ERP Wrapper...")
    logger.info(f"ERP Base URL: {settings.ERP_BASE_URL}")
    logger.info(f"API Prefix: {settings.ERP_API_PREFIX}")
    
    # Test connection in background
    loop = asyncio.get_event_loop()
    try:
        connected = await loop.run_in_executor(executor, test_connection)
        if connected:
            logger.info("✅ Successfully connected to eFab ERP")
        else:
            logger.warning("⚠️ Could not connect to eFab ERP on startup")
    except Exception as e:
        logger.error(f"❌ Startup connection test failed: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "eFab ERP Wrapper",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/health",
            "/api/status",
            "/api/sales-orders",
            "/api/knit-orders",
            "/api/inventory/{warehouse}",
            "/api/styles",
            "/api/sync-all",
            "/api/cache/clear"
        ],
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    loop = asyncio.get_event_loop()
    try:
        connected = await loop.run_in_executor(executor, test_connection)
        return {
            "status": "healthy" if connected else "degraded",
            "erp_connected": connected,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/status")
async def get_status():
    """Get detailed status of the wrapper service."""
    session_info = session_mgr.get_session_info()
    
    # Check cache status
    cache_status = {}
    for key, entry in data_cache.items():
        cache_status[key] = {
            "has_data": entry["data"] is not None,
            "valid": is_cache_valid(entry),
            "age_seconds": (
                (datetime.now() - entry["timestamp"]).total_seconds()
                if entry["timestamp"] else None
            )
        }
    
    return {
        "service": "running",
        "session": session_info,
        "cache": cache_status,
        "config": {
            "base_url": str(settings.ERP_BASE_URL),
            "username": settings.ERP_USERNAME,
            "ssl_verify": settings.VERIFY_SSL
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/sales-orders")
async def get_sales_orders_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get sales orders from eFab."""
    cache_entry = data_cache["sales_orders"]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info("Returning cached sales orders")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_sales_orders)
        
        # Update cache
        data_cache["sales_orders"] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "data": data,
            "count": len(data),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching sales orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knit-orders")
async def get_knit_orders_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get knit orders from eFab."""
    cache_entry = data_cache["knit_orders"]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info("Returning cached knit orders")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_knit_orders)
        
        # Update cache
        data_cache["knit_orders"] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "data": data,
            "count": len(data),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching knit orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inventory/{warehouse}")
async def get_inventory_endpoint(
    warehouse: str,
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get inventory for specific warehouse."""
    valid_warehouses = ["F01", "G00", "G02", "I01", "all"]
    if warehouse not in valid_warehouses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid warehouse. Must be one of: {valid_warehouses}"
        )
    
    cache_key = f"inventory_{warehouse}"
    if cache_key not in data_cache:
        data_cache[cache_key] = {"data": None, "timestamp": None}
    
    cache_entry = data_cache[cache_key]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info(f"Returning cached inventory for {warehouse}")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "warehouse": warehouse,
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_inventory, warehouse)
        
        # Update cache
        data_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "data": data,
            "count": len(data),
            "warehouse": warehouse,
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    for key in data_cache:
        data_cache[key] = {"data": None, "timestamp": None}
    
    return {
        "success": True,
        "message": "Cache cleared",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )