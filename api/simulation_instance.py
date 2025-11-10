"""
Singleton instance of SimulationAPI.
This module provides a centralized access point for the simulation instance
to avoid circular import issues.
"""
from backend.api.SimulationApi import SimulationAPI

# Global simulation instance (Singleton)
sim_api = SimulationAPI.get_instance()