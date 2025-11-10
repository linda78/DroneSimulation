from flask import request, jsonify
from flask_restful import Resource

from api.simulation_instance import sim_api


class SimulationResource(Resource):
    """REST resource for simulation control"""

    def get(self):
        state = sim_api.get_state()
        if state is None:
            return {'error': 'No simulation loaded'}, 404
        return jsonify(state)

    def post(self):
        data = request.get_json()

        if 'config_path' in data:
            sim_api.load_config(data['config_path'])
            response = {'message': 'Simulation loaded successfully'}
            return response, 201
        elif 'config' in data:
            sim_api.create_simulation(data['config'])
            return {'message': 'Simulation created successfully'}, 201
        else:
            return {'error': 'Missing config_path or config in request'}, 400

    def delete(self):
        sim_api.stop_simulation()
        return {'message': 'Simulation stopped'}, 200
