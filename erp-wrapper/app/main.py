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
    get_greige_styles,
    get_finished_styles,
    test_connection,
    test_quads_connection,
    session_mgr,
    quads_session_mgr,
    erp_get_json
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
    "styles": {"data": None, "timestamp": None},
    "greige_styles": {"data": None, "timestamp": None},
    "finished_styles": {"data": None, "timestamp": None}
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
            "/api/styles/greige/active",
            "/api/styles/finished/active",
            "/api/yarn/active",
            "/api/report/yarn_demand",
            "/api/report/yarn_demand_ko",
            "/api/yarn-po",
            "/api/report/yarn_expected",
            "/api/greige/g00",
            "/api/greige/g02",
            "/api/finished/i01",
            "/api/finished/f01",
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
        # Test both eFab and QuadS connections
        erp_connected, quads_connected = await asyncio.gather(
            loop.run_in_executor(executor, test_connection),
            loop.run_in_executor(executor, test_quads_connection),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(erp_connected, Exception):
            erp_connected = False
        if isinstance(quads_connected, Exception):
            quads_connected = False
            
        overall_status = "healthy" if (erp_connected and quads_connected) else (
            "degraded" if (erp_connected or quads_connected) else "unhealthy"
        )
        
        return {
            "status": overall_status,
            "erp_connected": erp_connected,
            "quads_connected": quads_connected,
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
    
    # Get QuadS session info
    try:
        quads_session_info = {
            "has_cookie": quads_session_mgr.cookie_cache is not None,
            "cookie_name": settings.SESSION_COOKIE_NAME,
            "base_url": str(settings.QUADS_BASE_URL),
            "client_active": quads_session_mgr.client is not None
        }
    except Exception as e:
        quads_session_info = {"error": str(e)}
    
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
        "sessions": {
            "erp": session_info,
            "quads": quads_session_info
        },
        "cache": cache_status,
        "config": {
            "erp_base_url": str(settings.ERP_BASE_URL),
            "quads_base_url": str(settings.QUADS_BASE_URL),
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

@app.get("/api/yarn/active")
async def get_yarn_active():
    """Get active yarn data from eFab."""
    try:
        data = erp_get_json("/api/yarn/active")
        return {
            "success": True,
            "data": data if isinstance(data, list) else [data] if data else [],
            "count": len(data) if isinstance(data, list) else (1 if data else 0),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching yarn active: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/yarn_demand")
async def get_yarn_demand():
    """Get yarn demand report from eFab."""
    try:
        data = erp_get_json("/api/report/yarn_demand")
        return {
            "success": True,
            "data": data if isinstance(data, list) else [data] if data else [],
            "count": len(data) if isinstance(data, list) else (1 if data else 0),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching yarn demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/yarn_demand_ko")
async def get_yarn_demand_ko():
    """Get yarn demand KO format report from eFab."""
    try:
        data = erp_get_json("/api/report/yarn_demand_ko")
        return {
            "success": True,
            "data": data if isinstance(data, list) else [data] if data else [],
            "count": len(data) if isinstance(data, list) else (1 if data else 0),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching yarn demand KO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/yarn-po")
async def get_yarn_po():
    """Get yarn purchase orders from eFab."""
    try:
        data = erp_get_json("/api/yarn-po")
        return {
            "success": True,
            "data": data if isinstance(data, list) else [data] if data else [],
            "count": len(data) if isinstance(data, list) else (1 if data else 0),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching yarn PO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/yarn_expected")
async def get_yarn_expected():
    """Get expected yarn deliveries from eFab."""
    try:
        data = erp_get_json("/api/report/yarn_expected")
        return {
            "success": True,
            "data": data if isinstance(data, list) else [data] if data else [],
            "count": len(data) if isinstance(data, list) else (1 if data else 0),
            "source": "erp",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching yarn expected: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/styles")
async def get_styles_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get styles from eFab."""
    cache_entry = data_cache["styles"]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info("Returning cached styles")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_styles)
        
        # Update cache
        data_cache["styles"] = {
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
        logger.error(f"Error fetching styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/styles/greige/active")
async def get_greige_styles_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from QuadS")
):
    """Get greige styles from QuadS."""
    cache_entry = data_cache["greige_styles"]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info("Returning cached greige styles")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_greige_styles)
        
        # Update cache
        data_cache["greige_styles"] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "data": data,
            "count": len(data),
            "source": "quads",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching greige styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/styles/finished/active")
async def get_finished_styles_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from QuadS")
):
    """Get finished styles from QuadS."""
    cache_entry = data_cache["finished_styles"]
    
    if not force_refresh and is_cache_valid(cache_entry):
        logger.info("Returning cached finished styles")
        return {
            "success": True,
            "data": cache_entry["data"],
            "count": len(cache_entry["data"]),
            "source": "cache",
            "cached_at": cache_entry["timestamp"].isoformat()
        }
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, get_finished_styles)
        
        # Update cache
        data_cache["finished_styles"] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "data": data,
            "count": len(data),
            "source": "quads",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching finished styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Individual inventory endpoints for direct access
@app.get("/api/greige/g00")
async def get_greige_g00_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get Greige G00 inventory directly."""
    return await get_inventory_endpoint("G00", force_refresh)

@app.get("/api/greige/g02")
async def get_greige_g02_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get Greige G02 inventory directly."""
    return await get_inventory_endpoint("G02", force_refresh)

@app.get("/api/finished/i01")
async def get_inspection_i01_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get Inspection I01 inventory directly."""
    return await get_inventory_endpoint("I01", force_refresh)

@app.get("/api/finished/f01")
async def get_finished_f01_endpoint(
    force_refresh: bool = Query(False, description="Force refresh from ERP")
):
    """Get Finished F01 inventory directly."""
    return await get_inventory_endpoint("F01", force_refresh)

@app.post("/api/sync-all")
async def sync_all_data(background_tasks: BackgroundTasks):
    """Sync all data in background to populate cache."""
    
    async def sync_task():
        """Background task to sync all endpoints."""
        try:
            loop = asyncio.get_event_loop()
            
            # Sync all main endpoints
            tasks = [
                loop.run_in_executor(executor, get_sales_orders),
                loop.run_in_executor(executor, get_knit_orders),
                loop.run_in_executor(executor, get_styles),
                loop.run_in_executor(executor, get_inventory, "all")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update cache with results
            endpoints = ["sales_orders", "knit_orders", "styles", "inventory"]
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    data_cache[endpoints[i]] = {
                        "data": result,
                        "timestamp": datetime.now()
                    }
                    logger.info(f"Synced {endpoints[i]}: {len(result)} records")
                else:
                    logger.error(f"Failed to sync {endpoints[i]}: {result}")
            
            logger.info("Background sync completed")
            
        except Exception as e:
            logger.error(f"Background sync failed: {e}")
    
    background_tasks.add_task(sync_task)
    
    return {
        "success": True,
        "message": "Background sync started",
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