#!/usr/bin/env python3
"""
Machine Work Center Mapping Service for Beverly Knits ERP
Maps the complete production chain: Style → Work Center → Machine ID
Data Sources: QuadS_greigeFabricList_(1).xlsx + Machine Report fin.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MachineInfo:
    """Machine information container"""
    machine_id: str
    work_center: str
    status: str = "UNKNOWN"
    capacity_lbs_per_day: float = 0.0
    utilization_percent: float = 0.0
    current_style: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class WorkCenterInfo:
    """Work center information container"""
    work_center_id: str
    machine_count: int
    total_capacity: float
    avg_utilization: float
    machines: List[str]
    active_styles: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

class MachineWorkCenterMapper:
    """
    Manages the complete mapping chain:
    Style → Work Center → Machine ID (287 machines)
    
    Data Sources:
    1. QuadS_greigeFabricList_(1).xlsx - Style to Work Center mapping
    2. Machine Report fin.xlsx - Work Center to Machine ID mapping  
    3. Production Lbs.xlsx - Style capacity (via ProductionCapacityManager)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize the machine mapper"""
        self.data_path = data_path or "/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data"
        
        # Data containers
        self.style_to_work_center: Dict[str, str] = {}
        self.work_center_to_machines: Dict[str, List[str]] = {}
        self.machine_info: Dict[str, MachineInfo] = {}
        self.work_center_info: Dict[str, WorkCenterInfo] = {}
        
        # Mapping chains
        self.style_to_machine_chain: Dict[str, Dict] = {}
        
        # Data loading status
        self.quads_loaded = False
        self.machine_report_loaded = False
        self.last_update = None
        
        # Performance caching
        self._cache_valid = False
        self._cached_mappings = {}
        
        logger.info("Initialized MachineWorkCenterMapper")
    
    def load_all_data(self) -> bool:
        """Load all mapping data from Excel files"""
        try:
            # Load QuadS greige fabric data
            if not self.load_quads_data():
                logger.error("Failed to load QuadS data")
                return False
            
            # Load machine report data  
            if not self.load_machine_report():
                logger.error("Failed to load machine report")
                return False
            
            # Build complete mapping chain
            self._build_mapping_chain()
            
            # Update work center summaries
            self._update_work_center_summaries()
            
            self.last_update = datetime.now()
            logger.info("Successfully loaded all machine mapping data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading machine mapping data: {e}")
            return False
    
    def load_quads_data(self) -> bool:
        """Load QuadS greige fabric list - Style to Work Center mapping"""
        try:
            quads_file = Path(self.data_path) / "QuadS_greigeFabricList_ (1).xlsx"
            
            if not quads_file.exists():
                # Try alternative name
                quads_file = Path(self.data_path) / "QuadS_finishedFabricList.csv"
                if quads_file.exists():
                    df = pd.read_csv(quads_file)
                else:
                    logger.error(f"QuadS file not found: {quads_file}")
                    return False
            else:
                df = pd.read_excel(quads_file)
            
            logger.info(f"Loaded QuadS data with {len(df)} records")
            
            # Extract Style to Work Center mapping
            # Look for common column variations
            style_col = None
            work_center_col = None
            
            # Find style column
            for col in df.columns:
                if any(x in col.lower() for x in ['style', 'fabric']):
                    style_col = col
                    break
            
            # Find work center column  
            for col in df.columns:
                if any(x in col.lower() for x in ['work', 'center', 'wc', 'machine']):
                    work_center_col = col
                    break
            
            if not style_col or not work_center_col:
                logger.warning("Could not identify style or work center columns")
                # Use first two columns as fallback
                if len(df.columns) >= 2:
                    style_col = df.columns[0]
                    work_center_col = df.columns[1]
                    logger.info(f"Using columns: {style_col} -> {work_center_col}")
                else:
                    return False
            
            # Build mapping
            for _, row in df.iterrows():
                style = str(row[style_col]).strip()
                work_center = str(row[work_center_col]).strip()
                
                if style and work_center and style != 'nan' and work_center != 'nan':
                    self.style_to_work_center[style] = work_center
            
            self.quads_loaded = True
            logger.info(f"Loaded {len(self.style_to_work_center)} style→work center mappings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading QuadS data: {e}")
            return False
    
    def load_machine_report(self) -> bool:
        """Load Machine Report - Work Center to Machine ID mapping (supports multiple machines per work center)"""
        try:
            # Try different file variations
            possible_files = [
                Path(self.data_path) / "Machine Report fin1.xlsx",
                Path(self.data_path) / "Machine Report fin1.csv", 
                Path(self.data_path) / "Machine Report fin.xlsx",
                Path(self.data_path) / "Machine Report.xlsx"
            ]
            
            machine_file = None
            for file_path in possible_files:
                if file_path.exists():
                    machine_file = file_path
                    break
            
            if not machine_file:
                logger.error(f"Machine report file not found. Tried: {[str(f) for f in possible_files]}")
                return False
            
            # Load file based on extension
            if machine_file.suffix.lower() == '.csv':
                df = pd.read_csv(machine_file)
            else:
                df = pd.read_excel(machine_file)
            logger.info(f"Loaded Machine Report with {len(df)} records")
            
            # Find machine ID and work center columns
            machine_id_col = None
            work_center_col = None
            
            # Find machine ID column
            for col in df.columns:
                if any(x in col.lower() for x in ['machine', 'id', 'number']):
                    machine_id_col = col
                    break
            
            # Find work center column
            for col in df.columns:
                if any(x in col.lower() for x in ['work', 'center', 'wc']):
                    work_center_col = col
                    break
            
            if not machine_id_col or not work_center_col:
                logger.warning("Could not identify machine ID or work center columns")
                # Use first two columns as fallback
                if len(df.columns) >= 2:
                    machine_id_col = df.columns[0]
                    work_center_col = df.columns[1] 
                    logger.info(f"Using columns: {machine_id_col} -> {work_center_col}")
                else:
                    return False
            
            # Build work center to machines mapping (supports multiple machines per work center)
            for _, row in df.iterrows():
                machine_id = str(row[machine_id_col]).strip()
                work_center = str(row[work_center_col]).strip()
                
                if machine_id and work_center and machine_id != 'nan' and work_center != 'nan':
                    # Add to work center mapping (multiple machines per work center supported)
                    if work_center not in self.work_center_to_machines:
                        self.work_center_to_machines[work_center] = []
                    
                    # Avoid duplicates
                    if machine_id not in self.work_center_to_machines[work_center]:
                        self.work_center_to_machines[work_center].append(machine_id)
                    
                    # Create machine info (overwrite if duplicate - keep latest)
                    self.machine_info[machine_id] = MachineInfo(
                        machine_id=machine_id,
                        work_center=work_center,
                        status="IDLE"  # Default status
                    )
            
            self.machine_report_loaded = True
            total_machines = sum(len(machines) for machines in self.work_center_to_machines.values())
            logger.info(f"Loaded {len(self.work_center_to_machines)} work centers with {total_machines} machines")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading machine report: {e}")
            return False
    
    def _build_mapping_chain(self):
        """Build the complete mapping chain: Style → Work Center → Machine IDs"""
        self.style_to_machine_chain = {}
        
        for style, work_center in self.style_to_work_center.items():
            machine_ids = self.work_center_to_machines.get(work_center, [])
            
            self.style_to_machine_chain[style] = {
                'style': style,
                'work_center': work_center,
                'machine_ids': machine_ids,
                'machine_count': len(machine_ids)
            }
        
        logger.info(f"Built mapping chain for {len(self.style_to_machine_chain)} styles")
    
    def _update_work_center_summaries(self):
        """Update work center summary information"""
        self.work_center_info = {}
        
        for work_center, machine_ids in self.work_center_to_machines.items():
            # Get styles assigned to this work center
            assigned_styles = [
                style for style, wc in self.style_to_work_center.items() 
                if wc == work_center
            ]
            
            # Calculate total capacity (placeholder - would need real data)
            total_capacity = len(machine_ids) * 500  # Assume 500 lbs/day per machine
            
            self.work_center_info[work_center] = WorkCenterInfo(
                work_center_id=work_center,
                machine_count=len(machine_ids),
                total_capacity=total_capacity,
                avg_utilization=0.0,  # Would calculate from real utilization data
                machines=machine_ids.copy(),
                active_styles=assigned_styles
            )
    
    # === Core Mapping Methods ===
    
    def get_work_center_for_style(self, style: str) -> Optional[str]:
        """Get work center assigned to a style"""
        return self.style_to_work_center.get(style)
    
    def get_machine_ids_for_style(self, style: str) -> List[str]:
        """Get all machine IDs that can process a style"""
        work_center = self.get_work_center_for_style(style)
        if work_center:
            return self.work_center_to_machines.get(work_center, [])
        return []
    
    def get_machine_ids_for_work_center(self, work_center: str) -> List[str]:
        """Get all machine IDs in a work center"""
        return self.work_center_to_machines.get(work_center, [])
    
    def get_work_center_for_machine(self, machine_id: str) -> Optional[str]:
        """Get work center that contains a machine"""
        machine = self.machine_info.get(machine_id)
        return machine.work_center if machine else None
    
    def get_styles_for_work_center(self, work_center: str) -> List[str]:
        """Get all styles assigned to a work center"""
        return [
            style for style, wc in self.style_to_work_center.items() 
            if wc == work_center
        ]
    
    def get_complete_mapping_for_style(self, style: str) -> Optional[Dict]:
        """Get complete mapping chain for a style"""
        return self.style_to_machine_chain.get(style)
    
    # === Summary and Statistics Methods ===
    
    def get_work_center_summary(self) -> Dict[str, WorkCenterInfo]:
        """Get summary of all work centers"""
        return self.work_center_info.copy()
    
    def get_machine_summary(self) -> Dict[str, MachineInfo]:
        """Get summary of all machines"""
        return self.machine_info.copy()
    
    def get_mapping_statistics(self) -> Dict:
        """Get overall mapping statistics"""
        total_machines = sum(len(machines) for machines in self.work_center_to_machines.values())
        
        return {
            'total_styles': len(self.style_to_work_center),
            'total_work_centers': len(self.work_center_to_machines),
            'total_machines': total_machines,
            'avg_machines_per_work_center': total_machines / max(1, len(self.work_center_to_machines)),
            'styles_with_mappings': len(self.style_to_machine_chain),
            'data_loaded': self.quads_loaded and self.machine_report_loaded,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def validate_mappings(self) -> Dict[str, List[str]]:
        """Validate mapping consistency and return issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Check for styles without work centers
        styles_without_wc = []
        for style in self.style_to_work_center.keys():
            work_center = self.style_to_work_center[style]
            if work_center not in self.work_center_to_machines:
                styles_without_wc.append(f"Style {style} → Work Center {work_center} (no machines)")
        
        if styles_without_wc:
            issues['errors'].extend(styles_without_wc)
        
        # Check for work centers without machines
        empty_work_centers = [
            wc for wc, machines in self.work_center_to_machines.items() 
            if not machines
        ]
        
        if empty_work_centers:
            issues['warnings'].extend([f"Empty work center: {wc}" for wc in empty_work_centers])
        
        # Check for duplicate machine assignments
        all_machines = []
        for machines in self.work_center_to_machines.values():
            all_machines.extend(machines)
        
        duplicate_machines = [m for m in set(all_machines) if all_machines.count(m) > 1]
        if duplicate_machines:
            issues['errors'].extend([f"Duplicate machine: {m}" for m in duplicate_machines])
        
        return issues
    
    def export_mappings_to_json(self, file_path: Optional[str] = None) -> str:
        """Export all mappings to JSON file for debugging"""
        export_data = {
            'statistics': self.get_mapping_statistics(),
            'style_to_work_center': self.style_to_work_center,
            'work_center_to_machines': self.work_center_to_machines,
            'style_to_machine_chain': self.style_to_machine_chain,
            'work_center_info': {k: v.to_dict() for k, v in self.work_center_info.items()},
            'machine_info': {k: v.to_dict() for k, v in self.machine_info.items()},
            'validation_issues': self.validate_mappings()
        }
        
        if not file_path:
            file_path = f"/tmp/machine_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported mappings to {file_path}")
        return file_path


# Global instance
_machine_mapper = None

def get_machine_mapper(data_path: Optional[str] = None) -> MachineWorkCenterMapper:
    """Get or create the global machine mapper instance"""
    global _machine_mapper
    
    if _machine_mapper is None:
        _machine_mapper = MachineWorkCenterMapper(data_path)
        # Auto-load data
        _machine_mapper.load_all_data()
    
    return _machine_mapper

def reset_machine_mapper():
    """Reset the global machine mapper instance"""
    global _machine_mapper
    _machine_mapper = None


if __name__ == "__main__":
    # Test the machine mapper
    print("Testing MachineWorkCenterMapper")
    print("=" * 50)
    
    mapper = MachineWorkCenterMapper()
    
    # Load data
    if mapper.load_all_data():
        print("Data loaded successfully!")
        
        # Show statistics
        stats = mapper.get_mapping_statistics()
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show work center summary
        print(f"\nWork Center Summary:")
        for wc_id, wc_info in mapper.get_work_center_summary().items():
            print(f"  {wc_id}: {wc_info.machine_count} machines, {len(wc_info.active_styles)} styles")
        
        # Validate mappings
        issues = mapper.validate_mappings()
        if issues['errors']:
            print(f"\nErrors found:")
            for error in issues['errors'][:5]:  # Show first 5
                print(f"  - {error}")
        
        if issues['warnings']:
            print(f"\nWarnings:")
            for warning in issues['warnings'][:5]:  # Show first 5
                print(f"  - {warning}")
        
        # Export for debugging
        export_file = mapper.export_mappings_to_json()
        print(f"\nMappings exported to: {export_file}")
        
    else:
        print("Failed to load data")