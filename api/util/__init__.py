from pathlib import Path

environment = {
    'DOMAIN': 'localhost',
    'PORT': '5001',
    'PREFIX': ''
}

# Get absolute path to config file (project_root/static/swagger/config.json)
project_root = Path(__file__).parent.parent.parent
config_file_path = str(project_root / 'static' / 'swagger' / 'config.json')