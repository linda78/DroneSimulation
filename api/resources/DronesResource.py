from flask import jsonify
from flask_restful import Resource

from api.simulation_instance import sim_api


class DronesResource(Resource):
    """REST resource for drone information"""

    def get(self, drone_id=None):

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
