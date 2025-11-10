import json
from . import environment, config_file_path


class ENVIRONMENT:
    def __init__(self):
        self.domain = environment.get('DOMAIN')
        self.port = environment.get("PORT")
        self.prefix = environment.get("PREFIX")

    def get_instance(self):
        if not hasattr(self, "_instance"):
            self._instance = ENVIRONMENT()
        return self._instance

    def getDomain(self):
        return self.domain

    def getPort(self):
        return self.port

    def getPrefix(self):
        return self.prefix


domain = ENVIRONMENT().get_instance().getDomain()
port = ENVIRONMENT().get_instance().getPort()
prefix = ENVIRONMENT().get_instance().getPrefix()


def build_swagger_config_json():
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)

    config_data['servers'] = [
        {"url": f"http://localhost:{port}{prefix}"}
        # ,{"url": f"http://{domain}:{port}{prefix}"}
    ]

    new_config_file_path = config_file_path

    with open(new_config_file_path, 'w') as new_file:
        json.dump(config_data, new_file, indent=2)
