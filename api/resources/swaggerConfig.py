from flask_restful import Resource
from flask import jsonify
import json
import os
from pathlib import Path


class SwaggerConfig(Resource):
    def get(self):
        # Get the project root directory (parent of api directory)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'static' / 'swagger' / 'config.json'

        with open(config_path, 'r') as config_file:
            config_data = json.load(config_file)
        return jsonify(config_data)
