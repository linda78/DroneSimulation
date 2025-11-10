import threading
from typing import Optional

from backend import Simulation, SimulationConfig, ConfigLoader


class SimulationAPI:
    """
    API wrapper for simulation control (Singleton)
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.simulation: Optional[Simulation] = None
        self.config: Optional[SimulationConfig] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.is_running = False

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SimulationAPI"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_config(self, config_path: str):
        """Load simulation configuration"""
        self.config = ConfigLoader.load(config_path)
        self.simulation = Simulation(self.config)

    def create_simulation(self, config_dict: dict):
        """Create simulation from configuration dictionary"""
        # Convert dict to config (simplified version)
        self.config = SimulationConfig()
        # TODO: Parse config_dict into SimulationConfig
        self.simulation = Simulation(self.config)

    def start_simulation(self):
        """Start simulation in background thread"""
        if self.simulation is None:
            raise ValueError("No simulation loaded")

        if self.is_running:
            return False

        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.start()
        return True

    def _run_simulation(self):
        """Run simulation (internal)"""
        try:
            while self.is_running and self.simulation.current_time < self.simulation.config.duration:
                self.simulation.step()
        finally:
            self.is_running = False

    def stop_simulation(self):
        """Stop simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)

    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.stop_simulation()
        if self.simulation:
            self.simulation.reset()

    def get_state(self):
        """Get current simulation state"""
        if self.simulation is None:
            return None
        return self.simulation.get_state()

    def get_history(self):
        """Get simulation history"""
        if self.simulation is None:
            return []
        return self.simulation.state_history
