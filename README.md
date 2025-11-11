# 3D Drone Simulation

A state-of-the-art 3D drone simulation with physically accurate flight dynamics, interchangeable collision avoidance algorithms and real-time visualisation.

## Features

### Core Functionality
- **3D Simulation** with VisPy for high-performance real-time visualisation
- **Interchangeable Flight Models**:
  - Physical model with realistic acceleration/deceleration
  - Simple model for direct movement
- **Collision Avoidance** with various algorithms:
  - Right Avoidance (right-hand evasion)
  - Repulsive Forces (repulsive forces)
  - Velocity Obstacle (velocity-based)
- **Flexible Route Configuration**:
  - Circular routes
  - Rectangular routes
  - User-defined waypoints
- **Configurable Space**:
  - Adjustable room size
  - Support for photo/video textures (planned)
  - Simple 3D example space

### Visualisation
- Interactive 3D camera (rotation, zoom)
- Flight path display with configurable length
- Camera tracking mode for individual drones
- Real-time performance display (FPS)

### Export & Analysis
- **Data Export** in multiple formats:
  - JSON (complete state history)
  - CSV (flat data structure)
  - Parquet (efficient binary format)
  - Excel (with summaries and metadata)
- **Video Export** of the simulation
- **3D Trajectories** as interactive HTML plots (Plotly)
- **Summary Statistics** for each drone

### API
- **REST API** for remote control:
  - Load and start simulation
  - Real-time state query
  - Step-by-step execution
  - Retrieve history data

## Project Structure

```
PythonProject1/
├── model/              # Core Data Models
│   ├── drone.py           # Drone Class with Physics State
│   ├── route.py           # Routes and Waypoints
│   ├── environment.py     # Space and Environment
│   ├── flight_model.py    # Flight Physics Models
│   └── avoidance_agent.py # Collision Avoidance Algorithms
├── backend/            # Simulation Engine
│   ├── config.py          # YAML Configuration System
│   └── simulation.py      # SimPy-based Simulation
├── gui/                # GUI and Visualisation
│   └── viewer.py          # VisPy 3D Viewer
├── api/                # REST API
│   └── server.py          # Flask Server
├── export/             # Export Functionality
│   ├── data_exporter.py   # Data Export (CSV, JSON, etc.)
│   └── video_exporter.py  # Video Export
├── configs/            # Example Configurations
│   ├── server_tester_config.yaml
│   ├── multi_drone.yaml
│   ├── stress_test.yaml
│   └── camera_follow.yaml
├── output/             # Output Directory
├── assets/             # Assets (GIFs, Textures)
├── main.py             # Main Entry Point
├── requirements.txt    # Python Dependencies
└── README.md          # This File
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1: Clone or Download Repository

```bash
cd DroneSimulation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulation

Simple demo with 3D visualisation:
```bash
python main.py run configs/server_tester_config.yaml
```

Multi-drone demo:
```bash
python main.py run configs/multi_drone.yaml
```

Headless mode (without GUI):
```bash
python main.py run configs/server_tester_config.yaml --headless
```

With video export:
```bash
python main.py run configs/multi_drone.yaml --export-video
```

### 3D Visualisation Controls

When the 3D visualisation is running:
- **Drag mouse**: Rotate camera
- **Mouse wheel**: Zoom
- **Close window**: End simulation

### Creating a Configuration File

Create default configuration:
```bash
python main.py create-config my_config.yaml
```

### Starting the API Server

Start REST API:
```bash
python main.py api --port 5000
```

Access API documentation:
```
http://localhost:5000/
```

### Displaying Available Configurations

```bash
python main.py list-configs
```

## Configuration

Configuration files are in YAML format and define all aspects of the simulation.

### Example Configuration

```yaml
simulation:
  name: "My Simulation"
  duration: 60.0        # Duration in seconds
  time_step: 0.05       # Time step in seconds
  real_time: false      # Real-time mode

room:
  dimensions: [20.0, 20.0, 10.0]  # Width, depth, height in metres
  texture_path: null              # Path to image/video
  is_video: false

drones:
  - count: 3                      # Number of drones
    max_speed: 5.0                # Max. speed (m/s)
    max_acceleration: 2.0         # Max. acceleration (m/s²)
    size: 0.3                     # Drone size (radius in m)
    colour: [1.0, 0.2, 0.2]       # RGB colour (0-1)
    route_type: "circular"        # circular, rectangular, waypoints
    route_params:
      centre: [10, 10, 0]
      radius: 5.0
      height: 5.0
      num_points: 8

flight_model:
  type: "physical"                # physical, simple
  params:
    drag_coefficient: 0.1
    approach_distance: 2.0

avoidance:
  type: "right"                   # right, repulsive, velocity_obstacle
  detection_radius: 3.0
  params:
    avoidance_strength: 1.0

visualisation:
  show_trajectories: true
  trajectory_length: 50
  camera_follow: null             # Drone ID or null

export:
  video: false
  data: true
  video_fps: 30
  output_dir: "output"
```

## Available Flight Models

### Physical Flight Model
Physically realistic model with:
- Smooth acceleration and deceleration
- Air resistance (drag)
- Speed and acceleration limits
- Automatic braking when approaching waypoints

### Simple Flight Model
Simple model with constant speed for direct movement to the target.

## Collision Avoidance Algorithms

### Right Avoidance Agent
Always evades other drones to the right (like traffic rules).
- Parameter: `avoidance_strength`

### Repulsive Avoidance Agent
Uses repulsive forces (inverse square law).
- Parameter: `repulsion_strength`

### Velocity Obstacle Avoidance Agent
Considers velocities for predictive collision avoidance.
- Parameter: `time_horizon`

## REST API

### Endpoints

- `GET /api/status` - Retrieve simulation status
- `GET /api/simulation` - Retrieve current state
- `POST /api/simulation` - Load simulation
- `POST /api/simulation/control/start` - Start simulation
- `POST /api/simulation/control/stop` - Stop simulation
- `POST /api/simulation/control/reset` - Reset simulation
- `POST /api/simulation/control/step` - Execute one step
- `GET /api/drones` - Retrieve all drones
- `GET /api/drones/<id>` - Retrieve specific drone
- `GET /api/history` - Retrieve simulation history

### Example Usage

```bash
# Start server
python main.py api

# Load simulation
curl -X POST http://localhost:5000/api/simulation \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/server_tester_config.yaml"}'

# Start simulation
curl -X POST http://localhost:5000/api/simulation/control/start

# Retrieve status
curl http://localhost:5000/api/status
```

## Export Functions

### Exporting Data

The simulation automatically exports data to the configured `output_dir`:
- `simulation_data.json` - Complete state history
- `simulation_data.csv` - Flat CSV file
- `trajectories.html` - Interactive 3D plot
- `summary.txt` - Textual summary

### Exporting Video

Videos are exported in MP4 format with configurable frame rate:
```bash
python main.py run configs/multi_drone.yaml --export-video
```

## Development

### Adding a New Flight Model

1. Create a new class in `model/flight_model.py`:
```python
class MyFlightModel(FlightModel):
    def update(self, drone, target, dt, other_drones, avoidance_vector):
        # Implement flight logic
        new_state = ...
        return new_state

    def get_name(self):
        return "MyFlightModel"
```

2. Register in `backend/simulation.py` in `_create_flight_model()`

### Adding a New Avoidance Algorithm

1. Create a new class in `model/avoidance_agent.py`:
```python
class MyAvoidanceAgent(AvoidanceAgent):
    def calculate_avoidance(self, drone, other_drones):
        # Calculate evasion vector
        return avoidance_vector

    def get_name(self):
        return "MyAvoidanceAgent"
```

2. Register in `backend/simulation.py` in `_create_avoidance_agent()`

## Technical Details

### Technologies Used
- **SimPy**: Event-based simulation
- **VisPy**: High-performance 3D visualisation (OpenGL)
- **NumPy**: Numerical calculations
- **Flask**: REST API
- **PyTorch**: (available for AI-based control)
- **Pandas**: Data analysis and export
- **Plotly**: Interactive plots
- **OpenCV**: Video export

### Performance
- Supports 20+ drones in real time
- Physics update with 0.05s time step (20 Hz)
- 3D rendering with 30-60 FPS

## Example Scenarios

### 1. Simple Demo (3 Drones)
```bash
python main.py run configs/server_tester_config.yaml
```

### 2. Multi-Drone (8 Drones, Different Groups)
```bash
python main.py run configs/multi_drone.yaml
```

### 3. Stress Test (20 Drones)
```bash
python main.py run configs/stress_test.yaml --headless
```

### 4. Camera Tracking
```bash
python main.py run configs/camera_follow.yaml
```

## Troubleshooting

### ImportError: No module named 'vispy'
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### OpenGL Errors
VisPy requires OpenGL. On some systems, an OpenGL driver may need to be installed.

### Simulation Too Slow
- Reduce the number of drones
- Increase `time_step` in the configuration
- Disable `show_trajectories`
- Use headless mode

## Licence

MIT Licence

## Author

Linda Muemken


## Further Development

Possible extensions:
- [ ] Obstacles in the space
- [ ] Photo/video textures for space
- [ ] AI-based control with PyTorch
- [ ] Multi-agent reinforcement learning
- [ ] Sensor simulation (cameras, lidar)
- [ ] Wind effects and turbulence
- [ ] Battery simulation
- [ ] Swarm intelligence algorithms
- [ ] VR support