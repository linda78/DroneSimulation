"""
Flask REST API for drone simulation control
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import threading
from typing import Optional

from backend import Simulation, SimulationConfig, ConfigLoader


class SimulationAPI:
    """
    API wrapper for simulation control
    """

    def __init__(self):
        self.simulation: Optional[Simulation] = None
        self.config: Optional[SimulationConfig] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.is_running = False

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


# Global simulation instance
sim_api = SimulationAPI()


class SimulationResource(Resource):
    """REST resource for simulation control"""

    def get(self):
        """Get current simulation state"""
        state = sim_api.get_state()
        if state is None:
            return {'error': 'No simulation loaded'}, 404
        return jsonify(state)

    def post(self):
        """Create new simulation"""
        data = request.get_json()

        if 'config_path' in data:
            try:
                sim_api.load_config(data['config_path'])
                return {'message': 'Simulation loaded successfully'}, 201
            except Exception as e:
                return {'error': str(e)}, 400
        elif 'config' in data:
            try:
                sim_api.create_simulation(data['config'])
                return {'message': 'Simulation created successfully'}, 201
            except Exception as e:
                return {'error': str(e)}, 400
        else:
            return {'error': 'Missing config_path or config in request'}, 400

    def delete(self):
        """Stop and clear simulation"""
        sim_api.stop_simulation()
        return {'message': 'Simulation stopped'}, 200


class SimulationControlResource(Resource):
    """REST resource for simulation playback control"""

    def post(self, action):
        """Execute control action"""
        if action == 'start':
            success = sim_api.start_simulation()
            if success:
                return {'message': 'Simulation started'}, 200
            else:
                return {'error': 'Simulation already running'}, 400

        elif action == 'stop':
            sim_api.stop_simulation()
            return {'message': 'Simulation stopped'}, 200

        elif action == 'reset':
            sim_api.reset_simulation()
            return {'message': 'Simulation reset'}, 200

        elif action == 'step':
            if sim_api.simulation:
                success = sim_api.simulation.step()
                return {
                    'message': 'Step executed',
                    'completed': not success
                }, 200
            else:
                return {'error': 'No simulation loaded'}, 404

        else:
            return {'error': f'Unknown action: {action}'}, 400


class DronesResource(Resource):
    """REST resource for drone information"""

    def get(self, drone_id=None):
        """Get drone(s) information"""
        state = sim_api.get_state()
        if state is None:
            return {'error': 'No simulation loaded'}, 404

        if drone_id is None:
            # Return all drones
            return jsonify({'drones': state['drones']})
        else:
            # Return specific drone
            drones = state['drones']
            for drone in drones:
                if drone['id'] == drone_id:
                    return jsonify(drone)
            return {'error': f'Drone {drone_id} not found'}, 404


class HistoryResource(Resource):
    """REST resource for simulation history"""

    def get(self):
        """Get simulation history"""
        history = sim_api.get_history()
        return jsonify({
            'history': history,
            'length': len(history)
        })


class StatusResource(Resource):
    """REST resource for simulation status"""

    def get(self):
        """Get simulation status"""
        if sim_api.simulation is None:
            return jsonify({
                'loaded': False,
                'running': False
            })

        return jsonify({
            'loaded': True,
            'running': sim_api.is_running,
            'current_time': sim_api.simulation.current_time,
            'duration': sim_api.simulation.config.duration,
            'num_drones': len(sim_api.simulation.drones)
        })


def create_app():
    """
    Create and configure Flask application

    Returns:
        Flask app instance
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for web frontend

    api = Api(app)

    # Register endpoints
    api.add_resource(SimulationResource, '/api/simulation')
    api.add_resource(SimulationControlResource, '/api/simulation/control/<string:action>')
    api.add_resource(DronesResource, '/api/drones', '/api/drones/<int:drone_id>')
    api.add_resource(HistoryResource, '/api/history')
    api.add_resource(StatusResource, '/api/status')

    @app.route('/')
    def index():
        """API documentation"""
        return jsonify({
            'name': 'Drone Simulation API',
            'version': '1.0',
            'endpoints': {
                'GET /api/status': 'Get simulation status',
                'GET /api/simulation': 'Get current simulation state',
                'POST /api/simulation': 'Create/load simulation (body: {config_path: "path"} or {config: {...}})',
                'DELETE /api/simulation': 'Stop simulation',
                'POST /api/simulation/control/start': 'Start simulation',
                'POST /api/simulation/control/stop': 'Stop simulation',
                'POST /api/simulation/control/reset': 'Reset simulation',
                'POST /api/simulation/control/step': 'Execute one simulation step',
                'GET /api/drones': 'Get all drones',
                'GET /api/drones/<id>': 'Get specific drone',
                'GET /api/history': 'Get simulation history'
            }
        })

    return app


def run_server(host='0.0.0.0', port=5000, debug=False):
    """
    Run the Flask server

    Args:
        host: Host address
        port: Port number
        debug: Debug mode
    """
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
