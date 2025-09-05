"""Dependency injection container for Beverly Knits ERP v2."""

import os
from dependency_injector import containers, providers
from pathlib import Path

# Import existing services
from src.services.inventory_analyzer_service import InventoryAnalyzerService
from src.services.sales_forecasting_service import SalesForecastingService
from src.services.capacity_planning_service import CapacityPlanningService
from src.services.erp_service_manager import ERPServiceManager
from src.services.business_rules import BusinessRulesService

# Import data loaders
from src.data_loaders.unified_data_loader import UnifiedDataLoader
from src.data_loaders.efab_api_loader import EfabAPILoader

# Import cache manager
from src.utils.cache_manager import UnifiedCacheManager

# Import configuration
from src.config.ml_config import MLConfig
from src.config.feature_flags import FeatureFlags

# Import production modules
from src.production.six_phase_planning_engine import SixPhasePlanningEngine
from src.production.enhanced_production_pipeline import EnhancedProductionPipeline
from src.production.enhanced_production_suggestions_v2 import EnhancedProductionSuggestionsV2

# Import forecasting modules
from src.forecasting.enhanced_forecasting_engine import EnhancedForecastingEngine
from src.forecasting.forecast_accuracy_monitor import ForecastAccuracyMonitor
from src.forecasting.forecast_auto_retrain import ForecastAutoRetrain

# Import yarn intelligence
from src.yarn_intelligence.yarn_intelligence_enhanced import YarnIntelligenceEnhanced
from src.yarn_intelligence.yarn_interchangeability_analyzer import YarnInterchangeabilityAnalyzer
from src.yarn_intelligence.intelligent_yarn_matcher import IntelligentYarnMatcher


class Container(containers.DeclarativeContainer):
    """Main dependency injection container for Beverly Knits ERP."""
    
    # Configuration provider
    config = providers.Configuration()
    
    # Initialize configuration from environment and files
    config.from_env("BEVERLY_", as_dict=True)
    
    # Base paths
    data_base_path = providers.Object(
        os.getenv('DATA_BASE_PATH', '/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/')
    )
    
    # Feature Flags - Singleton
    feature_flags = providers.Singleton(
        FeatureFlags
    )
    
    # ML Configuration - Singleton
    ml_config = providers.Singleton(
        MLConfig
    )
    
    # Cache Manager - Singleton (shared across all services)
    cache_manager = providers.Singleton(
        UnifiedCacheManager,
        cache_dir=config.cache.dir.as_(str) if config.cache.dir else '/tmp/bki_cache',
        enable_redis=config.cache.redis_enabled.as_(bool) if config.cache.redis_enabled else True,
        redis_host=config.redis.host.as_(str) if config.redis.host else 'localhost',
        redis_port=config.redis.port.as_(int) if config.redis.port else 6379,
        default_ttl=config.cache.default_ttl.as_(int) if config.cache.default_ttl else 900
    )
    
    # Data Loaders - Singleton (heavy initialization)
    unified_data_loader = providers.Singleton(
        UnifiedDataLoader,
        base_path=data_base_path,
        cache_manager=cache_manager,
        use_parallel=config.data.use_parallel.as_(bool) if config.data.use_parallel else True,
        max_workers=config.data.max_workers.as_(int) if config.data.max_workers else 4
    )
    
    efab_api_loader = providers.Singleton(
        EfabAPILoader,
        cache_manager=cache_manager,
        base_url=config.efab.api_url.as_(str) if config.efab.api_url else None,
        api_key=config.efab.api_key.as_(str) if config.efab.api_key else None
    )
    
    # Core Business Services - Factory (new instance per request)
    inventory_analyzer = providers.Factory(
        InventoryAnalyzerService,
        data_loader=unified_data_loader,
        cache_manager=cache_manager
    )
    
    sales_forecasting = providers.Factory(
        SalesForecastingService,
        data_loader=unified_data_loader,
        ml_config=ml_config,
        cache_manager=cache_manager
    )
    
    capacity_planning = providers.Factory(
        CapacityPlanningService,
        data_loader=unified_data_loader,
        cache_manager=cache_manager
    )
    
    business_rules = providers.Factory(
        BusinessRulesService,
        data_loader=unified_data_loader
    )
    
    # Production Services - Factory
    six_phase_planning = providers.Factory(
        SixPhasePlanningEngine,
        data_loader=unified_data_loader,
        cache_manager=cache_manager,
        inventory_analyzer=inventory_analyzer,
        forecasting_engine=sales_forecasting
    )
    
    production_pipeline = providers.Factory(
        EnhancedProductionPipeline,
        data_loader=unified_data_loader,
        planning_engine=six_phase_planning,
        cache_manager=cache_manager
    )
    
    production_suggestions = providers.Factory(
        EnhancedProductionSuggestionsV2,
        data_loader=unified_data_loader,
        inventory_analyzer=inventory_analyzer,
        forecasting_service=sales_forecasting,
        cache_manager=cache_manager
    )
    
    # Forecasting Services - Factory
    enhanced_forecasting = providers.Factory(
        EnhancedForecastingEngine,
        data_loader=unified_data_loader,
        ml_config=ml_config,
        cache_manager=cache_manager
    )
    
    forecast_monitor = providers.Factory(
        ForecastAccuracyMonitor,
        forecasting_engine=enhanced_forecasting,
        data_loader=unified_data_loader
    )
    
    forecast_auto_retrain = providers.Factory(
        ForecastAutoRetrain,
        forecasting_engine=enhanced_forecasting,
        accuracy_monitor=forecast_monitor,
        ml_config=ml_config
    )
    
    # Yarn Intelligence Services - Factory
    yarn_intelligence = providers.Factory(
        YarnIntelligenceEnhanced,
        data_loader=unified_data_loader,
        inventory_analyzer=inventory_analyzer,
        cache_manager=cache_manager
    )
    
    yarn_interchangeability = providers.Factory(
        YarnInterchangeabilityAnalyzer,
        data_loader=unified_data_loader,
        cache_manager=cache_manager
    )
    
    intelligent_yarn_matcher = providers.Factory(
        IntelligentYarnMatcher,
        data_loader=unified_data_loader,
        interchangeability_analyzer=yarn_interchangeability,
        cache_manager=cache_manager
    )
    
    # Service Manager - Singleton (coordinates all services)
    service_manager = providers.Singleton(
        ERPServiceManager,
        inventory_analyzer=inventory_analyzer,
        sales_forecasting=sales_forecasting,
        capacity_planning=capacity_planning,
        business_rules=business_rules,
        production_pipeline=production_pipeline,
        yarn_intelligence=yarn_intelligence,
        cache_manager=cache_manager,
        feature_flags=feature_flags
    )
    
    # Initialize resources on container startup
    def init_resources(self):
        """Initialize all singleton resources."""
        # Warm up cache if configured
        if self.config.cache.warm_on_startup():
            cache_manager = self.cache_manager()
            data_loader = self.unified_data_loader()
            # This would trigger initial data load and caching
            try:
                data_loader.load_all_data_sources()
            except Exception as e:
                print(f"Warning: Cache warming failed: {e}")
    
    # Cleanup resources on shutdown
    def shutdown_resources(self):
        """Clean up resources on shutdown."""
        try:
            cache_manager = self.cache_manager()
            cache_manager.close_connections()
        except Exception as e:
            print(f"Warning: Resource cleanup failed: {e}")


# Global container instance
container = Container()

# Configure from environment variables and config files
def configure_container():
    """Configure the container from various sources."""
    # Load from .env file if exists
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
    
    # Load from config file if exists
    config_file = Path('config/settings.yaml')
    if config_file.exists():
        container.config.from_yaml(str(config_file))
    
    # Override with environment variables
    container.config.from_env("BEVERLY_", as_dict=True)
    
    return container


# Wire container to modules (for decorator support)
def wire_container(modules: list):
    """Wire the container to specified modules for injection."""
    container.wire(modules=modules)
    return container