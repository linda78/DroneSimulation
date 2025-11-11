from flask import Flask, jsonify, redirect
from flask_cors import CORS
from flask_restful import Api, MethodNotAllowed, NotFound
from flask_swagger_ui import get_swaggerui_blueprint

from api.resources import SimulationResource, SimulationControlResource, DronesResource, HistoryResource, StatusResource, SwaggerConfig
from api.util.common import prefix, build_swagger_config_json

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

if __name__ == '__main__':
    app.run(debug=True)
