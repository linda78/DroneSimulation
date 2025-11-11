# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 5-Minute Quick Start

### 1. Start Simple Demo

```bash
python main.py run configs/server_tester_config.yaml
```

This starts a 3D visualisation with 3 drones flying in circles.

**Controls:**
- Drag mouse: Rotate camera
- Mouse wheel: Zoom
- Close window: Exit

### 2. Try Other Demos

**Multi-Drone Demo** (8 drones, different routes):
```bash
python main.py run configs/multi_drone.yaml
```

**Camera Tracking** (camera follows a drone):
```bash
python main.py run configs/camera_follow.yaml
```

**Stress Test** (20 drones):
```bash
python main.py run configs/stress_test.yaml
```

### 3. Export Data

With data export:
```bash
python main.py run configs/server_tester_config.yaml --headless
```

The results are saved in `output/`:
- `simulation_data.json` - Complete state history
- `simulation_data.csv` - CSV format
- `trajectories.html` - Interactive 3D plot
- `summary.txt` - Summary

### 4. Export Video

```bash
python main.py run configs/multi_drone.yaml --export-video
```

The video is saved as `output/simulation.mp4`.

### 5. Use REST API

**Start server:**
```bash
python main.py api
```

**Test API:**
```bash
# Retrieve status
curl http://localhost:5000/api/status

# Load simulation
curl -X POST http://localhost:5000/api/simulation \
  -H "Content-Type: application/json" \
  -d '{"config_path": "configs/server_tester_config.yaml"}'

# Start simulation
curl -X POST http://localhost:5000/api/simulation/control/start

# Retrieve current state
curl http://localhost:5000/api/simulation
```

## Create Your Own Configuration

### Create default configuration:
```bash
python main.py create-config my_config.yaml
```

### Customise configuration:

Edit `my_config.yaml`:

```yaml
simulation:
  name: "My Simulation"
  duration: 60.0

drones:
  - count: 5                    # Change number
    max_speed: 7.0              # Fly faster
    colour: [1.0, 0.0, 0.0]     # Red
    route_type: "circular"
    route_params:
      radius: 8.0               # Larger circle

avoidance:
  type: "repulsive"             # Choose different algorithm
  detection_radius: 4.0
```

### Start simulation:
```bash
python main.py run my_config.yaml
```

## Example Scripts

Run interactive examples:
```bash
python example.py
```

This shows a menu with different examples:
1. Basic simulation
2. Programmatic configuration
3. Headless export
4. User-defined routes
5. Comparison of avoidance algorithms
6. Video export

## Common Use Cases

### Test different avoidance algorithms

```yaml
avoidance:
  type: "right"              # Always evade to the right
  # or
  type: "repulsive"          # Repulsive forces
  # or
  type: "velocity_obstacle"  # Velocity-based
```

### Change drone speed

```yaml
drones:
  - count: 3
    max_speed: 10.0           # Faster
    max_acceleration: 5.0     # Higher acceleration
```

### Use different routes

**Circular:**
```yaml
route_type: "circular"
route_params:
  centre: [10, 10, 0]
  radius: 5.0
  height: 5.0
```

**Rectangular:**
```yaml
route_type: "rectangular"
route_params:
  corner1: [5, 5, 0]
  corner2: [15, 15, 0]
  height: 5.0
```

**User-defined:**
```yaml
route_type: "waypoints"
route_params:
  positions:
    - [5, 5, 3]
    - [15, 5, 6]
    - [15, 15, 9]
    - [5, 15, 6]
  loop: true
```

## Tips

1. **Improve performance:**
   - Increase `time_step` from 0.05 to 0.1
   - Reduce `trajectory_length`
   - Use `--headless` for faster execution

2. **Better videos:**
   - Increase `video_fps` to 60
   - Use higher resolution in `VideoExporter`

3. **Debugging:**
   - Use `show_trajectories: true`
   - Set `camera_follow` to a drone ID
   - Export data and analyse in Excel/Pandas

## Next Steps

- Read the full [README.md](README.md)
- Experiment with different configurations
- Create your own flight models (see README.md "Development")
- Integrate the API into your application

## Help

Problems? See [README.md](README.md) section "Troubleshooting"
