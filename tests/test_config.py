"""
Unit tests for configuration system
"""

import unittest
import tempfile
import os
from pathlib import Path
from backend.config import (
    DroneConfig, RoomConfig, FlightModelConfig,
    AvoidanceConfig, SimulationConfig, ConfigLoader
)


class TestConfigDataClasses(unittest.TestCase):
    """Test configuration dataclasses"""

    def test_drone_config_defaults(self):
        """Test DroneConfig default values"""
        config = DroneConfig()

        self.assertEqual(config.count, 1)
        self.assertIsNone(config.initial_positions)
        self.assertEqual(config.max_speed, 5.0)
        self.assertEqual(config.max_acceleration, 2.0)
        self.assertEqual(config.size, 0.3)
        self.assertEqual(config.route_type, "circular")

    def test_drone_config_custom(self):
        """Test DroneConfig with custom values"""
        config = DroneConfig(
            count=3,
            max_speed=10.0,
            route_type="rectangular"
        )

        self.assertEqual(config.count, 3)
        self.assertEqual(config.max_speed, 10.0)
        self.assertEqual(config.route_type, "rectangular")

    def test_room_config_defaults(self):
        """Test RoomConfig default values"""
        config = RoomConfig()

        self.assertEqual(config.dimensions, [20.0, 20.0, 10.0])
        self.assertIsNone(config.texture_path)
        self.assertFalse(config.is_video)

    def test_room_config_custom(self):
        """Test RoomConfig with custom values"""
        config = RoomConfig(
            dimensions=[30.0, 25.0, 15.0],
            texture_path="/path/to/texture.png"
        )

        self.assertEqual(config.dimensions, [30.0, 25.0, 15.0])
        self.assertEqual(config.texture_path, "/path/to/texture.png")

    def test_flight_model_config_defaults(self):
        """Test FlightModelConfig defaults"""
        config = FlightModelConfig()

        self.assertEqual(config.type, "physical")
        self.assertIsNone(config.gif_path)
        self.assertEqual(len(config.params), 0)

    def test_avoidance_config_defaults(self):
        """Test AvoidanceConfig defaults"""
        config = AvoidanceConfig()

        self.assertEqual(config.type, "right")
        self.assertEqual(config.detection_radius, 3.0)
        self.assertEqual(len(config.params), 0)

    def test_simulation_config_defaults(self):
        """Test SimulationConfig defaults"""
        config = SimulationConfig()

        self.assertEqual(config.simulation_name, "drone_simulation")
        self.assertEqual(config.duration, 60.0)
        self.assertEqual(config.time_step, 0.05)
        self.assertFalse(config.real_time)
        self.assertTrue(config.show_trajectories)
        self.assertEqual(config.trajectory_length, 50)
        self.assertIsNone(config.camera_follow)


class TestConfigLoader(unittest.TestCase):
    """Test ConfigLoader functionality"""

    def setUp(self):
        """Create temporary directory for test configs"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_temp_config(self, content: str) -> str:
        """Helper to create temporary config file"""
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(config_path, 'w') as f:
            f.write(content)
        return config_path

    def test_load_minimal_config(self):
        """Test loading minimal valid configuration"""
        yaml_content = """
simulation:
  name: "test_sim"
  duration: 30.0

room:
  dimensions: [10.0, 10.0, 5.0]

drones:
  - count: 1
    max_speed: 5.0
"""
        config_path = self.create_temp_config(yaml_content)
        config = ConfigLoader.load(config_path)

        self.assertEqual(config.simulation_name, "test_sim")
        self.assertEqual(config.duration, 30.0)
        self.assertEqual(config.room.dimensions, [10.0, 10.0, 5.0])
        self.assertEqual(len(config.drones), 1)
        self.assertEqual(config.drones[0].count, 1)

    def test_load_full_config(self):
        """Test loading complete configuration"""
        yaml_content = """
simulation:
  name: "full_test"
  duration: 120.0
  time_step: 0.02
  real_time: true

room:
  dimensions: [40.0, 40.0, 20.0]

drones:
  - count: 2
    max_speed: 8.0
    max_acceleration: 4.0
    size: 0.5
    route_type: "circular"
    route_params:
      radius: 10.0

flight_model:
  type: "physical"
  params:
    drag_coefficient: 0.15

avoidance:
  type: "repulsive"
  detection_radius: 5.0

visualization:
  show_trajectories: false
  trajectory_length: 100

export:
  video: true
  data: false
"""
        config_path = self.create_temp_config(yaml_content)
        config = ConfigLoader.load(config_path)

        # Check simulation settings
        self.assertEqual(config.simulation_name, "full_test")
        self.assertEqual(config.duration, 120.0)
        self.assertEqual(config.time_step, 0.02)
        self.assertTrue(config.real_time)

        # Check room
        self.assertEqual(config.room.dimensions, [40.0, 40.0, 20.0])

        # Check drones
        self.assertEqual(len(config.drones), 1)
        self.assertEqual(config.drones[0].count, 2)
        self.assertEqual(config.drones[0].max_speed, 8.0)
        self.assertEqual(config.drones[0].size, 0.5)

        # Check flight model
        self.assertEqual(config.flight_model.type, "physical")

        # Check avoidance
        self.assertEqual(config.avoidance.type, "repulsive")
        self.assertEqual(config.avoidance.detection_radius, 5.0)

        # Check visualization
        self.assertFalse(config.show_trajectories)
        self.assertEqual(config.trajectory_length, 100)

    def test_load_multiple_drone_groups(self):
        """Test loading config with multiple drone groups"""
        yaml_content = """
simulation:
  name: "multi_group"

room:
  dimensions: [20.0, 20.0, 10.0]

drones:
  - count: 2
    max_speed: 5.0
    route_type: "circular"
  - count: 3
    max_speed: 7.0
    route_type: "rectangular"
"""
        config_path = self.create_temp_config(yaml_content)
        config = ConfigLoader.load(config_path)

        self.assertEqual(len(config.drones), 2)
        self.assertEqual(config.drones[0].count, 2)
        self.assertEqual(config.drones[0].route_type, "circular")
        self.assertEqual(config.drones[1].count, 3)
        self.assertEqual(config.drones[1].route_type, "rectangular")

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error"""
        yaml_content = """
this is not: valid: yaml:::
  - [unclosed
"""
        config_path = self.create_temp_config(yaml_content)

        with self.assertRaises(Exception):  # yaml.YAMLError or similar
            ConfigLoader.load(config_path)

    def test_config_with_colors(self):
        """Test loading config with custom drone colors"""
        yaml_content = """
simulation:
  name: "color_test"

room:
  dimensions: [10.0, 10.0, 5.0]

drones:
  - count: 1
    color: [1.0, 0.0, 0.0]  # Red
"""
        config_path = self.create_temp_config(yaml_content)
        config = ConfigLoader.load(config_path)

        self.assertEqual(config.drones[0].color, [1.0, 0.0, 0.0])

    def test_config_with_initial_positions(self):
        """Test loading config with explicit initial positions"""
        yaml_content = """
simulation:
  name: "position_test"

room:
  dimensions: [10.0, 10.0, 5.0]

drones:
  - count: 2
    initial_positions:
      - [1.0, 1.0, 1.0]
      - [9.0, 9.0, 4.0]
"""
        config_path = self.create_temp_config(yaml_content)
        config = ConfigLoader.load(config_path)

        positions = config.drones[0].initial_positions
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0], [1.0, 1.0, 1.0])
        self.assertEqual(positions[1], [9.0, 9.0, 4.0])


if __name__ == '__main__':
    unittest.main()
