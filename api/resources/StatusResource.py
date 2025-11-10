from flask import jsonify
from flask_restful import Resource

from api.simulation_instance import sim_api


class StatusResource(Resource):
    """REST resource for simulation status"""

    def get(self):

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