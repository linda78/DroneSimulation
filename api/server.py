from flask import Flask, jsonify, redirect
from flask_cors import CORS
from flask_restful import Api, MethodNotAllowed, NotFound
from flask_swagger_ui import get_swaggerui_blueprint

from api.resources import SimulationResource, SimulationControlResource, DronesResource, HistoryResource, StatusResource, SwaggerConfig
from api.util.common import prefix, build_swagger_config_json

import threading
from werkzeug.serving import make_server

# ============================================
# Main
# ============================================
application = Flask(__name__)
app = application
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)
api = Api(app, prefix=prefix, catch_all_404s=True)

# ============================================
# Swagger
# ============================================
build_swagger_config_json()
swaggerui_blueprint = get_swaggerui_blueprint(
    prefix,
    f'{prefix}/swagger-config',
    config={
        'app_name': "Drone Simulation API",
        "layout": "BaseLayout",
        "docExpansion": "none"
    },
)
app.register_blueprint(swaggerui_blueprint)

# ============================================
# Error Handler
# ============================================

@app.errorhandler(NotFound)
def handle_method_not_found(e):
    response = jsonify({"message": str(e)})
    response.status_code = 404
    return response


@app.errorhandler(MethodNotAllowed)
def handle_method_not_allowed_error(e):
    response = jsonify({"message": str(e)})
    response.status_code = 405
    return response


@app.route('/')
def redirect_to_prefix():
    if prefix != '':
        return redirect(prefix)

# Register endpoints
api.add_resource(SimulationResource, '/api/simulation')
api.add_resource(SimulationControlResource, '/api/simulation/control/<string:action>')
api.add_resource(DronesResource, '/api/drones', '/api/drones/<int:drone_id>')
api.add_resource(HistoryResource, '/api/history')
api.add_resource(StatusResource, '/api/status')
api.add_resource(SwaggerConfig, '/swagger-config')

class BackgroundAPIServer:
    """Helper to run the DroneSimulation API in a background thread using Werkzeug.

    This is primarily intended for tests or integrations (e.g. BoF) that need
    to spin up and shut down the API programmatically.
    """

    def __init__(self, host: str = "localhost", port: int = 5001):
        self.host = host
        self.port = port
        self._server = make_server(self.host, self.port, app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        """Start the API server in a background thread."""
        self._thread.start()

    def shutdown(self, timeout: float = 5.0):
        """Shut down the API server and wait for the thread to finish."""
        try:
            self._server.shutdown()
        except Exception:
            # Best-effort shutdown; Fehler hier sollen den Testrunner nicht killen
            pass

        if self._thread.is_alive():
            self._thread.join(timeout=timeout)


def create_background_server(host: str = "localhost", port: int = 5001) -> BackgroundAPIServer:
    """Convenience factory used by external projects (e.g. BoF) to start the API."""
    return BackgroundAPIServer(host=host, port=port)

if __name__ == '__main__':
    app.run(debug=True)
