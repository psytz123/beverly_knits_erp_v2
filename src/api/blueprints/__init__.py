"""
API Blueprints Module
Contains modularized Flask blueprints for different API domains
Complete set of 6 blueprints covering all API endpoints
"""
from .inventory_bp import inventory_bp, init_blueprint as init_inventory_bp
from .production_bp import production_bp, init_blueprint as init_production_bp
from .system_bp import system_bp, init_blueprint as init_system_bp
from .forecasting_bp import forecasting_bp, init_blueprint as init_forecasting_bp
from .yarn_bp import yarn_bp, init_blueprint as init_yarn_bp
from .planning_bp import planning_bp, init_blueprint as init_planning_bp

__all__ = [
    'inventory_bp',
    'init_inventory_bp',
    'production_bp', 
    'init_production_bp',
    'system_bp',
    'init_system_bp',
    'forecasting_bp',
    'init_forecasting_bp',
    'yarn_bp',
    'init_yarn_bp',
    'planning_bp',
    'init_planning_bp'
]

def register_all_blueprints(app, service_manager, data_loader):
    """
    Register all blueprints with the Flask app
    This is a convenience function to register all 6 blueprints at once
    """
    # Initialize each blueprint with required services
    if service_manager and data_loader:
        # Inventory blueprint
        init_inventory_bp(
            service_manager.get_service('inventory'),
            service_manager.get_service('pipeline'),
            data_loader
        )
        
        # Production blueprint
        init_production_bp(
            service_manager.get_service('capacity'),
            service_manager.get_service('pipeline'),
            None,  # planning_engine - optional
            data_loader
        )
        
        # System blueprint
        init_system_bp(
            service_manager,
            data_loader,
            None  # cache_manager - optional
        )
        
        # Forecasting blueprint
        init_forecasting_bp(
            service_manager.get_service('forecasting'),
            data_loader,
            None  # ml_integration - optional
        )
        
        # Yarn blueprint (using None for services not in basic ServiceManager)
        init_yarn_bp(
            None,  # yarn_intelligence
            None,  # interchangeability
            data_loader,
            service_manager
        )
        
        # Planning blueprint (using None for services not in basic ServiceManager)
        init_planning_bp(
            None,  # six_phase_engine
            None,  # optimization
            service_manager,
            data_loader
        )
    
    # Register all blueprints with the app
    app.register_blueprint(inventory_bp, url_prefix='/api')
    app.register_blueprint(production_bp, url_prefix='/api')
    app.register_blueprint(system_bp, url_prefix='/api')
    app.register_blueprint(forecasting_bp, url_prefix='/api')
    app.register_blueprint(yarn_bp, url_prefix='/api')
    app.register_blueprint(planning_bp, url_prefix='/api')
    
    return True