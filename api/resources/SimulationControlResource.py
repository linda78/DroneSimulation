from flask_restful import Resource

from api.simulation_instance import sim_api


class SimulationControlResource(Resource):
    """REST resource for simulation playback control"""

    def post(self, action):
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
