#!/usr/bin/env python3
"""
Industry specialization agents for eFab AI Agent System
Manufacturing domain expertise
"""

from .furniture_agent import FurnitureManufacturingAgent, FurnitureProductType, WoodType, FinishType
from .injection_molding_agent import InjectionMoldingAgent, PlasticType, MoldType, DefectType
from .electrical_equipment_agent import ElectricalEquipmentAgent, ElectricalProductType, CertificationType, TestType

__all__ = [
    "FurnitureManufacturingAgent",
    "FurnitureProductType",
    "WoodType", 
    "FinishType",
    "InjectionMoldingAgent",
    "PlasticType",
    "MoldType",
    "DefectType", 
    "ElectricalEquipmentAgent",
    "ElectricalProductType",
    "CertificationType",
    "TestType"
]