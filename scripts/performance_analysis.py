#!/usr/bin/env python3
"""
Beverly Knits ERP Performance Analysis Script
Phase 1 Day 1-2: Profile all API endpoints and identify bottlenecks
"""

import time
import json
import tracemalloc
import psutil
import requests
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Profile Beverly Knits ERP endpoints and identify performance bottlenecks"""
    
    def __init__(self, base_url: str = "http://localhost:5006"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "endpoints": {},
            "memory_analysis": {},
            "data_loader_comparison": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
    def profile_endpoint(self, endpoint: str, method: str = "GET", 
                         data: Dict = None, iterations: int = 5) -> Dict:
        """Profile a single endpoint with multiple iterations"""
        
        url = f"{self.base_url}{endpoint}"
        timings = []
        memory_usage = []
        errors = []
        
        logger.info(f"Profiling endpoint: {endpoint}")
        
        for i in range(iterations):
            try:
                # Memory before request
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time request
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(url, timeout=30)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=30)
                else:
                    response = requests.request(method, url, json=data, timeout=30)
                    
                end_time = time.time()
                duration = end_time - start_time
                
                # Memory after request
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = mem_after - mem_before
                
                timings.append(duration)
                memory_usage.append(memory_delta)
                
                if response.status_code != 200:
                    errors.append(f"Status {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                errors.append("Request timeout (>30s)")
                timings.append(30.0)
                memory_usage.append(0)
            except Exception as e:
                errors.append(str(e))
                timings.append(0)
                memory_usage.append(0)
                
        return {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "avg_time": np.mean(timings) if timings else 0,
            "min_time": np.min(timings) if timings else 0,
            "max_time": np.max(timings) if timings else 0,
            "std_time": np.std(timings) if timings else 0,
            "avg_memory_delta_mb": np.mean(memory_usage) if memory_usage else 0,
            "errors": errors,
            "error_rate": len(errors) / iterations
        }
    
    def profile_critical_endpoints(self):
        """Profile all critical endpoints identified in the handoff document"""
        
        critical_endpoints = [
            # Core Dashboard APIs
            ("/api/production-planning", "GET"),
            ("/api/inventory-intelligence-enhanced", "GET"),
            ("/api/ml-forecast-detailed", "GET"),
            ("/api/inventory-netting", "GET"),
            ("/api/comprehensive-kpis", "GET"),
            ("/api/yarn-intelligence", "GET"),
            ("/api/production-suggestions", "GET"),
            ("/api/po-risk-analysis", "GET"),
            ("/api/production-pipeline", "GET"),
            ("/api/yarn-substitution-intelligent", "GET"),
            ("/api/production-recommendations-ml", "GET"),
            ("/api/knit-orders", "GET"),
            ("/api/machine-assignment-suggestions", "GET"),
            ("/api/factory-floor-ai-dashboard", "GET"),
            
            # Heavy computation endpoints
            ("/api/execute-planning", "POST"),
            ("/api/six-phase-supply-chain", "GET"),
            
            # Data loading endpoints
            ("/api/reload-data", "GET"),
            ("/api/debug-data", "GET"),
            
            # Health check
            ("/api/health", "GET"),
        ]
        
        logger.info(f"Starting profiling of {len(critical_endpoints)} endpoints...")
        
        for endpoint, method in critical_endpoints:
            result = self.profile_endpoint(endpoint, method)
            self.results["endpoints"][endpoint] = result
            
            # Identify bottlenecks (>2s response time or >50MB memory increase)
            if result["avg_time"] > 2.0:
                self.results["bottlenecks"].append({
                    "endpoint": endpoint,
                    "issue": "Slow response",
                    "avg_time": result["avg_time"],
                    "impact": "HIGH" if result["avg_time"] > 5.0 else "MEDIUM"
                })
                
            if result["avg_memory_delta_mb"] > 50:
                self.results["bottlenecks"].append({
                    "endpoint": endpoint,
                    "issue": "Memory leak",
                    "memory_increase_mb": result["avg_memory_delta_mb"],
                    "impact": "HIGH"
                })
                
            if result["error_rate"] > 0.2:
                self.results["bottlenecks"].append({
                    "endpoint": endpoint,
                    "issue": "High error rate",
                    "error_rate": result["error_rate"],
                    "errors": result["errors"],
                    "impact": "CRITICAL"
                })
    
    def analyze_data_loaders(self):
        """Compare performance of different data loader implementations"""
        
        logger.info("Analyzing data loader performance...")
        
        try:
            from src.data_loaders.optimized_data_loader import OptimizedDataLoader
            from src.data_loaders.parallel_data_loader import ParallelDataLoader
            from src.utils.cache_manager import UnifiedCacheManager
            
            # Test data path
            data_path = Path("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/8-28-2025")
            
            loaders = {
                "OptimizedDataLoader": OptimizedDataLoader,
                "ParallelDataLoader": ParallelDataLoader
            }
            
            for loader_name, LoaderClass in loaders.items():
                logger.info(f"Testing {loader_name}...")
                
                try:
                    # Clear cache first
                    cache_manager = UnifiedCacheManager()
                    cache_manager.clear_all()
                    
                    # Profile data loading
                    loader = LoaderClass()
                    
                    # Time yarn inventory loading
                    start = time.time()
                    tracemalloc.start()
                    
                    yarn_data = loader.load_yarn_inventory()
                    
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    load_time = time.time() - start
                    
                    self.results["data_loader_comparison"][loader_name] = {
                        "yarn_inventory_load_time": load_time,
                        "memory_usage_mb": peak / 1024 / 1024,
                        "records_loaded": len(yarn_data) if yarn_data is not None else 0,
                        "status": "SUCCESS" if yarn_data is not None else "FAILED"
                    }
                    
                except Exception as e:
                    self.results["data_loader_comparison"][loader_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    
        except ImportError as e:
            logger.error(f"Failed to import data loaders: {e}")
            self.results["data_loader_comparison"]["error"] = str(e)
    
    def analyze_memory_usage(self):
        """Analyze overall memory usage patterns"""
        
        logger.info("Analyzing memory usage...")
        
        try:
            # Start the app if not running
            process = psutil.Process()
            
            self.results["memory_analysis"] = {
                "current_memory_mb": process.memory_info().rss / 1024 / 1024,
                "virtual_memory_mb": process.memory_info().vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=1),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.net_connections()) if hasattr(process, 'net_connections') else 0
            }
            
            # System-wide memory
            vm = psutil.virtual_memory()
            self.results["memory_analysis"]["system"] = {
                "total_mb": vm.total / 1024 / 1024,
                "available_mb": vm.available / 1024 / 1024,
                "percent_used": vm.percent
            }
            
        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            self.results["memory_analysis"]["error"] = str(e)
    
    def generate_recommendations(self):
        """Generate specific recommendations based on findings"""
        
        logger.info("Generating recommendations...")
        
        # Analyze results and generate recommendations
        slow_endpoints = [b for b in self.results["bottlenecks"] 
                         if b.get("issue") == "Slow response"]
        memory_leaks = [b for b in self.results["bottlenecks"] 
                       if b.get("issue") == "Memory leak"]
        high_errors = [b for b in self.results["bottlenecks"] 
                      if b.get("issue") == "High error rate"]
        
        if slow_endpoints:
            self.results["recommendations"].append({
                "priority": "HIGH",
                "category": "Performance",
                "recommendation": "Implement caching for slow endpoints",
                "details": f"Found {len(slow_endpoints)} slow endpoints. Consider Redis caching.",
                "affected_endpoints": [s["endpoint"] for s in slow_endpoints]
            })
            
        if memory_leaks:
            self.results["recommendations"].append({
                "priority": "CRITICAL",
                "category": "Memory",
                "recommendation": "Fix memory leaks in data processing",
                "details": f"Found {len(memory_leaks)} endpoints with memory leaks.",
                "affected_endpoints": [m["endpoint"] for m in memory_leaks]
            })
            
        if high_errors:
            self.results["recommendations"].append({
                "priority": "CRITICAL",
                "category": "Stability",
                "recommendation": "Fix error-prone endpoints immediately",
                "details": f"Found {len(high_errors)} endpoints with high error rates.",
                "affected_endpoints": [e["endpoint"] for e in high_errors]
            })
            
        # Data loader recommendations
        if self.results.get("data_loader_comparison") and isinstance(self.results["data_loader_comparison"], dict):
            if "error" not in self.results["data_loader_comparison"]:
                valid_loaders = [(k, v) for k, v in self.results["data_loader_comparison"].items() 
                                if isinstance(v, dict) and "yarn_inventory_load_time" in v]
                if valid_loaders:
                    best_loader = min(
                        valid_loaders,
                        key=lambda x: x[1].get("yarn_inventory_load_time", float('inf'))
                    )
                    self.results["recommendations"].append({
                        "priority": "MEDIUM",
                        "category": "Data Loading",
                        "recommendation": f"Use {best_loader[0]} as primary data loader",
                        "details": f"Load time: {best_loader[1].get('yarn_inventory_load_time', 'N/A')}s"
                    })
            
        # Sort bottlenecks by impact
        self.results["bottlenecks"].sort(
            key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x.get("impact", "MEDIUM"))
        )
    
    def save_report(self):
        """Save performance analysis report"""
        
        report_path = Path("docs/reports/performance_analysis_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        logger.info(f"Report saved to {report_path}")
        
        # Also create a summary
        summary_path = Path("docs/reports/performance_summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# Beverly Knits ERP Performance Analysis Summary\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            f.write("## Critical Bottlenecks\n\n")
            for bottleneck in self.results["bottlenecks"][:10]:
                f.write(f"- **{bottleneck['endpoint']}**: {bottleneck['issue']} ")
                f.write(f"(Impact: {bottleneck.get('impact', 'N/A')})\n")
                
            f.write("\n## Top Recommendations\n\n")
            for rec in self.results["recommendations"]:
                f.write(f"### {rec['priority']} - {rec['category']}\n")
                f.write(f"{rec['recommendation']}\n")
                f.write(f"*{rec['details']}*\n\n")
                
            f.write("\n## Endpoint Performance\n\n")
            f.write("| Endpoint | Avg Time (s) | Error Rate | Memory Delta (MB) |\n")
            f.write("|----------|-------------|------------|------------------|\n")
            
            # Sort by average time
            sorted_endpoints = sorted(
                self.results["endpoints"].items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True
            )
            
            for endpoint, data in sorted_endpoints[:20]:
                f.write(f"| {endpoint} | {data['avg_time']:.2f} | ")
                f.write(f"{data['error_rate']:.0%} | {data['avg_memory_delta_mb']:.1f} |\n")
                
        logger.info(f"Summary saved to {summary_path}")

def main():
    """Run the complete performance analysis"""
    
    logger.info("Starting Beverly Knits ERP Performance Analysis")
    logger.info("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5006/api/health", timeout=5)
        if response.status_code != 200:
            logger.warning("Server health check failed, but continuing...")
    except:
        logger.error("Server not running on port 5006! Please start the server first.")
        logger.info("Run: python3 src/core/beverly_comprehensive_erp.py")
        return
    
    analyzer = PerformanceAnalyzer()
    
    # Run all analyses
    try:
        logger.info("\n1. Profiling Critical Endpoints...")
        analyzer.profile_critical_endpoints()
        
        logger.info("\n2. Analyzing Data Loaders...")
        analyzer.analyze_data_loaders()
        
        logger.info("\n3. Analyzing Memory Usage...")
        analyzer.analyze_memory_usage()
        
        logger.info("\n4. Generating Recommendations...")
        analyzer.generate_recommendations()
        
        logger.info("\n5. Saving Report...")
        analyzer.save_report()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        logger.info(f"\nFound {len(analyzer.results['bottlenecks'])} bottlenecks")
        logger.info(f"Generated {len(analyzer.results['recommendations'])} recommendations")
        
        # Print top 3 bottlenecks
        logger.info("\nTop Performance Issues:")
        for bottleneck in analyzer.results["bottlenecks"][:3]:
            logger.info(f"  - {bottleneck['endpoint']}: {bottleneck['issue']} ({bottleneck.get('impact', 'N/A')})")
            
        logger.info("\nReports saved to:")
        logger.info("  - docs/reports/performance_analysis_report.json")
        logger.info("  - docs/reports/performance_summary.md")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()