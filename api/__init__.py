"""
API package for controlling simulation via REST endpoints
"""

from .server import create_app, SimulationAPI

__all__ = ['create_app', 'SimulationAPI']
