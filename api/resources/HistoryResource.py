from flask import jsonify
from flask_restful import Resource

from api.simulation_instance import sim_api


class HistoryResource(Resource):
    """REST resource for simulation history"""

    def get(self):

        history = sim_api.get_history()
        return jsonify({
            'history': history,
            'length': len(history)
        })
