# REST API Server Quickstart Guide

This guide covers how to use the Drone Simulation REST API both programmatically (curl, code) and through the Swagger UI.

## Table of Contents
- [Starting the Server](#starting-the-server)
- [Swagger UI Usage](#swagger-ui-usage)
- [Programmatic Usage](#programmatic-usage)
- [API Endpoints](#api-endpoints)
- [Complete Workflow Examples](#complete-workflow-examples)

---

## Starting the Server

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python api/server.py
```

The server will start on:
- **Local:** http://127.0.0.1:5001
- **Network:** http://0.0.0.0:5001
- **Swagger UI:** http://localhost:5001/apidocs/

---

## Swagger UI Usage

### Accessing Swagger UI
Navigate to: **http://localhost:5001/apidocs/**

The Swagger UI provides an interactive interface where you can:
- Browse all available endpoints
- See detailed request/response schemas
- Test endpoints directly in your browser
- View example requests and responses

### Using Swagger UI

1. **Find the endpoint** you want to test (e.g., `POST /api/simulation`)
2. **Click "Try it out"** button
3. **Enter your request body** in the JSON editor
4. **Click "Execute"** to send the request
5. **View the response** below (status code, headers, body)

**Important:** When using file paths in Swagger UI, always use **absolute paths**:
```json
{
  "config_path": "/Users/lindamumken/work/repos/tmp/DroneSimulation/configs/simple_demo.yaml"
}
```

---

## Programmatic Usage

### Using curl

All examples below use `curl`. You can adapt them to any programming language.

### Option A: Load Simulation from Config File

Use this when you have a YAML configuration file:

```bash
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/Users/lindamumken/work/repos/tmp/DroneSimulation/configs/simple_demo.yaml"
  }'
```

**Response:**
```json
{
  "message": "Simulation loaded successfully"
}
```

### Option B: Send Configuration Directly as JSON

Use this when you don't have a config file or want to define the simulation programmatically:

```bash
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "simulation": {
        "name": "API Test Simulation",
        "duration": 30.0,
        "time_step": 0.05,
        "real_time": false
      },
      "room": {
        "dimensions": [20.0, 20.0, 10.0],
        "texture_path": null,
        "is_video": false
      },
      "drones": [
        {
          "count": 2,
          "initial_positions": [
            [5, 5, 3],
            [15, 15, 3]
          ],
          "max_speed": 5.0,
          "max_acceleration": 2.0,
          "size": 0.3,
          "gif_path": null,
          "color": null,
          "route_type": "circular",
          "route_params": {
            "center": [10, 10, 0],
            "radius": 5.0,
            "height": 3.0,
            "num_points": 6
          }
        }
      ],
      "flight_model": {
        "type": "physical",
        "gif_path": null,
        "params": {
          "drag_coefficient": 0.1,
          "approach_distance": 2.0
        }
      },
      "avoidance": {
        "type": "right",
        "detection_radius": 3.0,
        "params": {
          "avoidance_strength": 1.0
        }
      },
      "visualization": {
        "show_trajectories": true,
        "trajectory_length": 50,
        "camera_follow": null
      },
      "export": {
        "video": true,
        "data": true,
        "video_fps": 30,
        "output_dir": "output"
      }
    }
  }'
```

**Response:**
```json
{
  "message": "Simulation created successfully"
}
```

**Tip:** For complex JSON, save it to a file and use `@filename`:
```bash
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d @configs/test_api_config.json
```

### Using Python

#### Option A: Load from Config File
```python
import requests

BASE_URL = "http://localhost:5001"

# Load simulation from config file
response = requests.post(
    f"{BASE_URL}/api/simulation",
    json={"config_path": "/Users/lindamumken/work/repos/tmp/DroneSimulation/configs/simple_demo.yaml"}
)
print(response.json())  # {'message': 'Simulation loaded successfully'}

# Start simulation
response = requests.post(f"{BASE_URL}/api/simulation/control/start")
print(response.json())  # {'message': 'Simulation started'}

# Check status
response = requests.get(f"{BASE_URL}/api/status")
status = response.json()
print(f"Loaded: {status['loaded']}, Running: {status['running']}")

# Stop simulation
response = requests.post(f"{BASE_URL}/api/simulation/control/stop")
print(response.json())  # {'message': 'Simulation stopped'}
```

#### Option B: Send Config Directly
```python
import requests

BASE_URL = "http://localhost:5001"

# Define configuration
config = {
    "config": {
        "simulation": {
            "name": "Python Test",
            "duration": 30.0,
            "time_step": 0.05,
            "real_time": False
        },
        "room": {
            "dimensions": [20.0, 20.0, 10.0],
            "texture_path": None,
            "is_video": False
        },
        "drones": [{
            "count": 2,
            "initial_positions": [[5, 5, 3], [15, 15, 3]],
            "max_speed": 5.0,
            "max_acceleration": 2.0,
            "size": 0.3,
            "gif_path": None,
            "color": None,
            "route_type": "circular",
            "route_params": {
                "center": [10, 10, 0],
                "radius": 5.0,
                "height": 3.0,
                "num_points": 6
            }
        }],
        "flight_model": {
            "type": "physical",
            "gif_path": None,
            "params": {"drag_coefficient": 0.1, "approach_distance": 2.0}
        },
        "avoidance": {
            "type": "right",
            "detection_radius": 3.0,
            "params": {"avoidance_strength": 1.0}
        },
        "visualization": {
            "show_trajectories": True,
            "trajectory_length": 50,
            "camera_follow": None
        },
        "export": {
            "video": True,
            "data": True,
            "video_fps": 30,
            "output_dir": "output"
        }
    }
}

# Create simulation with inline config
response = requests.post(f"{BASE_URL}/api/simulation", json=config)
print(response.json())

# Start and monitor
requests.post(f"{BASE_URL}/api/simulation/control/start")
status = requests.get(f"{BASE_URL}/api/status").json()
print(f"Drones: {status['num_drones']}, Duration: {status['duration']}s")
```

---

## API Endpoints

### Status Endpoints

#### Get Simulation Status
```bash
curl http://localhost:5001/api/status
```

**Response:**
```json
{
  "loaded": true,
  "running": false,
  "current_time": 0.0,
  "duration": 60.0,
  "num_drones": 3
}
```

### Simulation Management

#### Create/Load Simulation
```bash
# With config file path
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/absolute/path/to/config.yaml"}'

# With inline config
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{"config": {...}}'
```

#### Get Current Simulation State
```bash
curl http://localhost:5001/api/simulation
```

#### Stop and Clear Simulation
```bash
curl -X DELETE http://localhost:5001/api/simulation
```

### Simulation Control

#### Start Simulation
```bash
curl -X POST http://localhost:5001/api/simulation/control/start
```

#### Stop Simulation
```bash
curl -X POST http://localhost:5001/api/simulation/control/stop
```

#### Reset Simulation
```bash
curl -X POST http://localhost:5001/api/simulation/control/reset
```

#### Execute Single Step
```bash
curl -X POST http://localhost:5001/api/simulation/control/step
```

### Drone Information

#### Get All Drones
```bash
curl http://localhost:5001/api/drones
```

**Response:**
```json
{
  "drones": [
    {
      "id": 0,
      "position": [10.0, 10.0, 3.0],
      "velocity": [0.0, 0.0, 0.0]
    },
    {
      "id": 1,
      "position": [10.0, 10.0, 5.0],
      "velocity": [0.0, 0.0, 0.0]
    }
  ]
}
```

#### Get Specific Drone
```bash
curl http://localhost:5001/api/drones/0
```

**Response:**
```json
{
  "id": 0,
  "position": [10.0, 10.0, 3.0],
  "velocity": [0.0, 0.0, 0.0]
}
```

### History

#### Get Simulation History
```bash
curl http://localhost:5001/api/history
```

**Response:**
```json
{
  "history": [
    {
      "time": 0.0,
      "drones": [...]
    },
    {
      "time": 0.05,
      "drones": [...]
    }
  ],
  "length": 150
}
```

---

## Complete Workflow Examples

### Example 1: Basic Workflow with Config File

```bash
# 1. Load simulation from config file
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/Users/lindamumken/work/repos/tmp/DroneSimulation/configs/simple_demo.yaml"}'

# 2. Check status
curl http://localhost:5001/api/status

# 3. Start simulation
curl -X POST http://localhost:5001/api/simulation/control/start

# 4. Get current state
curl http://localhost:5001/api/simulation

# 5. Get all drones
curl http://localhost:5001/api/drones

# 6. Stop simulation
curl -X POST http://localhost:5001/api/simulation/control/stop
```

### Example 2: Step-by-Step Execution

```bash
# 1. Load simulation
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/Users/lindamumken/work/repos/tmp/DroneSimulation/configs/simple_demo.yaml"}'

# 2. Execute single steps manually
curl -X POST http://localhost:5001/api/simulation/control/step
curl -X POST http://localhost:5001/api/simulation/control/step
curl -X POST http://localhost:5001/api/simulation/control/step

# 3. Check drone positions after each step
curl http://localhost:5001/api/drones

# 4. Reset to initial state
curl -X POST http://localhost:5001/api/simulation/control/reset
```

### Example 3: Using Python

```python
import requests
import json

BASE_URL = "http://localhost:5001"

# Load simulation from file
response = requests.post(
    f"{BASE_URL}/api/simulation",
    json={"config_path": "/path/to/config.yaml"}
)
print(response.json())

# Start simulation
response = requests.post(f"{BASE_URL}/api/simulation/control/start")
print(response.json())

# Get status
response = requests.get(f"{BASE_URL}/api/status")
print(response.json())

# Get all drones
response = requests.get(f"{BASE_URL}/api/drones")
drones = response.json()["drones"]
for drone in drones:
    print(f"Drone {drone['id']}: position={drone['position']}")

# Stop simulation
response = requests.post(f"{BASE_URL}/api/simulation/control/stop")
print(response.json())
```

### Example 4: Creating Simulation with JSON Config

```python
import requests

BASE_URL = "http://localhost:5001"

config = {
    "config": {
        "simulation": {
            "name": "Python Generated Simulation",
            "duration": 20.0,
            "time_step": 0.1,
            "real_time": False
        },
        "room": {
            "dimensions": [15.0, 15.0, 8.0],
            "texture_path": None,
            "is_video": False
        },
        "drones": [
            {
                "count": 1,
                "initial_positions": [[7.5, 7.5, 4.0]],
                "max_speed": 3.0,
                "max_acceleration": 1.5,
                "size": 0.3,
                "gif_path": None,
                "color": None,
                "route_type": "circular",
                "route_params": {
                    "center": [7.5, 7.5, 0],
                    "radius": 3.0,
                    "height": 4.0,
                    "num_points": 8
                }
            }
        ],
        "flight_model": {
            "type": "physical",
            "gif_path": None,
            "params": {
                "drag_coefficient": 0.1,
                "approach_distance": 1.5
            }
        },
        "avoidance": {
            "type": "right",
            "detection_radius": 2.5,
            "params": {
                "avoidance_strength": 1.0
            }
        },
        "visualization": {
            "show_trajectories": True,
            "trajectory_length": 40,
            "camera_follow": None
        },
        "export": {
            "video": False,
            "data": True,
            "video_fps": 30,
            "output_dir": "output"
        }
    }
}

# Create simulation
response = requests.post(f"{BASE_URL}/api/simulation", json=config)
print(response.json())

# Start simulation
response = requests.post(f"{BASE_URL}/api/simulation/control/start")
print(response.json())
```

---

## Minimal JSON Configuration Example

Here's a minimal configuration for quick testing (no file path needed):

```json
{
  "config": {
    "simulation": {
      "name": "Minimal Test",
      "duration": 10.0,
      "time_step": 0.1,
      "real_time": false
    },
    "room": {
      "dimensions": [10.0, 10.0, 5.0],
      "texture_path": null,
      "is_video": false
    },
    "drones": [
      {
        "count": 1,
        "initial_positions": [[5, 5, 2.5]],
        "max_speed": 3.0,
        "max_acceleration": 1.5,
        "size": 0.3,
        "gif_path": null,
        "color": null,
        "route_type": "circular",
        "route_params": {
          "center": [5, 5, 0],
          "radius": 2.0,
          "height": 2.5,
          "num_points": 4
        }
      }
    ],
    "flight_model": {
      "type": "physical",
      "gif_path": null,
      "params": {
        "drag_coefficient": 0.1,
        "approach_distance": 1.0
      }
    },
    "avoidance": {
      "type": "right",
      "detection_radius": 2.0,
      "params": {
        "avoidance_strength": 1.0
      }
    },
    "visualization": {
      "show_trajectories": true,
      "trajectory_length": 30,
      "camera_follow": null
    },
    "export": {
      "video": false,
      "data": true,
      "video_fps": 30,
      "output_dir": "output"
    }
  }
}
```

Save this as `minimal_config.json` and use:
```bash
curl -X POST "http://localhost:5001/api/simulation" \
  -H "Content-Type: application/json" \
  -d @minimal_config.json
```

---

## Tips and Best Practices

1. **Use absolute paths** when specifying config files in the API
2. **Check status** before starting a simulation to ensure it's loaded
3. **Stop simulations** before loading a new one
4. **Use Swagger UI** for interactive testing and exploring the API
5. **Save complex configs** to JSON files and use `@filename` with curl
6. **Access Swagger spec** at http://localhost:5001/apispec.json for OpenAPI spec

---

## Troubleshooting

### "Invalid configuration or missing parameters"
- Check that your JSON is properly formatted
- Ensure you're using either `config_path` or `config`, not both
- For `config_path`, use absolute paths

### "Configuration file not found"
- Use absolute paths: `/Users/.../DroneSimulation/configs/file.yaml`
- Verify the file exists: `ls /path/to/config.yaml`

### "No simulation loaded"
- Load a simulation first with `POST /api/simulation`
- Check status with `GET /api/status`

### Swagger UI shows old/cached responses
- Hard refresh the page: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)
- Clear browser cache

---

## API Documentation

For interactive API documentation and testing, visit:
**http://localhost:5001/apidocs/**

For the OpenAPI specification (JSON):
**http://localhost:5001/apispec.json**