"""
Flask REST API for drone simulation control
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from flasgger import Swagger
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
        """Get current simulation state
        ---
        tags:
          - Simulation
        summary: Get current simulation state
        description: Returns the current state of the loaded simulation including drone positions and configuration
        responses:
          200:
            description: Current simulation state
            schema:
              type: object
              properties:
                drones:
                  type: array
                  description: Array of drone objects
                time:
                  type: number
                  description: Current simulation time
          404:
            description: No simulation loaded
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: No simulation loaded
        """
        state = sim_api.get_state()
        if state is None:
            return {'error': 'No simulation loaded'}, 404
        return jsonify(state)

    def post(self):
        """Create new simulation
        ---
        tags:
          - Simulation
        summary: Create or load a new simulation
        description: Creates a new simulation from a config file path or configuration dictionary
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              oneOf:
                - properties:
                    config_path:
                      type: string
                      description: Path to YAML configuration file
                      example: configs/mpc_8Drones_WorkingOnEdge.yaml
                - properties:
                    config:
                      type: object
                      description: Configuration dictionary
        responses:
          201:
            description: Simulation created successfully
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: Simulation loaded successfully
          400:
            description: Invalid configuration or missing parameters
            schema:
              type: object
              properties:
                error:
                  type: string
        """
        data = request.get_json()
        print(f"DEBUG: Received data: {data}")
        print(f"DEBUG: Data type: {type(data)}")
        print(f"DEBUG: Keys in data: {data.keys() if data else 'None'}")

        if 'config_path' in data:
            try:
                print(f"DEBUG: Attempting to load config from: {data['config_path']}")
                sim_api.load_config(data['config_path'])
                print(f"DEBUG: Config loaded successfully!")
                response = {'message': 'Simulation loaded successfully'}
                print(f"DEBUG: Returning response: {response} with status 201")
                return response, 201
            except Exception as e:
                print(f"DEBUG: Exception caught: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}, 400
        elif 'config' in data:
            try:
                print(f"DEBUG: Creating simulation from config dict")
                sim_api.create_simulation(data['config'])
                print(f"DEBUG: Simulation created successfully!")
                return {'message': 'Simulation created successfully'}, 201
            except Exception as e:
                print(f"DEBUG: Exception in config creation: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}, 400
        else:
            print(f"DEBUG: No config_path or config found in data")
            return {'error': 'Missing config_path or config in request'}, 400

    def delete(self):
        """Stop and clear simulation
        ---
        tags:
          - Simulation
        summary: Stop the running simulation
        description: Stops the currently running simulation and clears its state
        responses:
          200:
            description: Simulation stopped successfully
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: Simulation stopped
        """
        sim_api.stop_simulation()
        return {'message': 'Simulation stopped'}, 200


class SimulationControlResource(Resource):
    """REST resource for simulation playback control"""

    def post(self, action):
        """Execute control action
        ---
        tags:
          - Simulation Control
        summary: Execute simulation control action
        description: |
          Control simulation playback with various actions:
          - start: Start the simulation
          - stop: Stop the simulation
          - reset: Reset simulation to initial state
          - step: Execute one simulation step
        parameters:
          - in: path
            name: action
            required: true
            schema:
              type: string
              enum: [start, stop, reset, step]
            description: Control action to execute
        responses:
          200:
            description: Action executed successfully
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: Simulation started
                completed:
                  type: boolean
                  description: Only for step action - indicates if simulation is completed
          400:
            description: Invalid action or simulation already in requested state
            schema:
              type: object
              properties:
                error:
                  type: string
          404:
            description: No simulation loaded
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: No simulation loaded
        """
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
        """Get drone(s) information
        ---
        tags:
          - Drones
        summary: Get information about drone(s)
        description: Returns information about all drones or a specific drone by ID
        parameters:
          - in: path
            name: drone_id
            required: false
            schema:
              type: integer
            description: ID of specific drone to retrieve (optional)
        responses:
          200:
            description: Drone information retrieved successfully
            schema:
              type: object
              properties:
                drones:
                  type: array
                  description: Array of drone objects (when no drone_id specified)
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                        description: Unique drone identifier
                      position:
                        type: array
                        items:
                          type: number
                        description: 3D position [x, y, z]
                      velocity:
                        type: array
                        items:
                          type: number
                        description: 3D velocity vector
                id:
                  type: integer
                  description: Drone ID (when specific drone_id requested)
                position:
                  type: array
                  items:
                    type: number
                  description: 3D position [x, y, z]
          404:
            description: Simulation not loaded or drone not found
            schema:
              type: object
              properties:
                error:
                  type: string
        """
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
        """Get simulation history
        ---
        tags:
          - History
        summary: Get complete simulation history
        description: Returns the complete history of simulation states over time
        responses:
          200:
            description: Simulation history retrieved successfully
            schema:
              type: object
              properties:
                history:
                  type: array
                  description: Array of historical simulation states
                  items:
                    type: object
                    properties:
                      time:
                        type: number
                        description: Timestamp of the state
                      drones:
                        type: array
                        description: Drone states at this time
                length:
                  type: integer
                  description: Number of history entries
                  example: 150
        """
        history = sim_api.get_history()
        return jsonify({
            'history': history,
            'length': len(history)
        })


class StatusResource(Resource):
    """REST resource for simulation status"""

    def get(self):
        """Get simulation status
        ---
        tags:
          - Status
        summary: Get current simulation status
        description: Returns the current status of the simulation including runtime information
        responses:
          200:
            description: Simulation status retrieved successfully
            schema:
              type: object
              properties:
                loaded:
                  type: boolean
                  description: Whether a simulation is currently loaded
                  example: true
                running:
                  type: boolean
                  description: Whether the simulation is currently running
                  example: false
                current_time:
                  type: number
                  description: Current simulation time (only if loaded)
                  example: 12.5
                duration:
                  type: number
                  description: Total simulation duration (only if loaded)
                  example: 100.0
                num_drones:
                  type: integer
                  description: Number of drones in simulation (only if loaded)
                  example: 8
        """
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

    # Configure Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/apidocs/"
    }

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Drone Simulation API",
            "description": "REST API for controlling and monitoring drone simulation",
            "version": "1.0.0",
            "contact": {
                "name": "API Support"
            }
        },
        "basePath": "/",
        "schemes": ["http", "https"],
    }

    Swagger(app, config=swagger_config, template=swagger_template)

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


def run_server(host='0.0.0.0', port=5001, debug=False):
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
