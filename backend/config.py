"""
Configuration system for drone simulation
Loads and validates YAML configuration files
"""

import yaml
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DroneConfig:
    """Configuration for a single drone or drone group"""
    count: int = 1
    initial_positions: Optional[List[List[float]]] = None
    max_speed: float = 5.0
    max_acceleration: float = 2.0
    size: float = 0.3
    gif_path: Optional[str] = None
    color: Optional[List[float]] = None
    route_type: str = "circular"  # circular, rectangular, waypoints
    route_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomConfig:
    """Configuration for the simulation room"""
    dimensions: List[float] = field(default_factory=lambda: [20.0, 20.0, 10.0])
    texture_path: Optional[str] = None
    is_video: bool = False


@dataclass
class FlightModelConfig:
    """Configuration for flight physics model"""
    type: str = "physical"  # physical, simple
    gif_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AvoidanceConfig:
    """Configuration for collision avoidance"""
    type: str = "right"  # right, repulsive, velocity_obstacle
    detection_radius: float = 3.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    # General settings
    simulation_name: str = "drone_simulation"
    duration: float = 60.0  # seconds
    time_step: float = 0.05  # seconds
    real_time: bool = False

    # Components
    room: RoomConfig = field(default_factory=RoomConfig)
    drones: List[DroneConfig] = field(default_factory=list)
    flight_model: FlightModelConfig = field(default_factory=FlightModelConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)

    # Visualization
    show_trajectories: bool = True
    trajectory_length: int = 50
    camera_follow: Optional[int] = None  # Drone ID to follow, None for static

    # Export
    export_video: bool = False
    export_data: bool = True
    video_fps: int = 30
    output_dir: str = "output"


class ConfigLoader:
    """
    Loads and validates simulation configuration from YAML files
    """

    @staticmethod
    def load(config_path: str) -> SimulationConfig:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            SimulationConfig object
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return ConfigLoader._parse_config(data)

    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> SimulationConfig:
        """Parse configuration dictionary into SimulationConfig"""

        config = SimulationConfig()

        # General settings
        if 'simulation' in data:
            sim = data['simulation']
            config.simulation_name = sim.get('name', config.simulation_name)
            config.duration = sim.get('duration', config.duration)
            config.time_step = sim.get('time_step', config.time_step)
            config.real_time = sim.get('real_time', config.real_time)

        # Room configuration
        if 'room' in data:
            room = data['room']
            config.room = RoomConfig(
                dimensions=room.get('dimensions', config.room.dimensions),
                texture_path=room.get('texture_path'),
                is_video=room.get('is_video', False)
            )

        # Drone configurations
        if 'drones' in data:
            for drone_data in data['drones']:
                drone_config = DroneConfig(
                    count=drone_data.get('count', 1),
                    initial_positions=drone_data.get('initial_positions'),
                    max_speed=drone_data.get('max_speed', 5.0),
                    max_acceleration=drone_data.get('max_acceleration', 2.0),
                    size=drone_data.get('size', 0.3),
                    gif_path=drone_data.get('gif_path'),
                    color=drone_data.get('color'),
                    route_type=drone_data.get('route_type', 'circular'),
                    route_params=drone_data.get('route_params', {})
                )
                config.drones.append(drone_config)

        # Flight model configuration
        if 'flight_model' in data:
            fm = data['flight_model']
            config.flight_model = FlightModelConfig(
                type=fm.get('type', 'physical'),
                gif_path=fm.get('gif_path'),
                params=fm.get('params', {})
            )

        # Avoidance configuration
        if 'avoidance' in data:
            av = data['avoidance']
            config.avoidance = AvoidanceConfig(
                type=av.get('type', 'right'),
                detection_radius=av.get('detection_radius', 3.0),
                params=av.get('params', {})
            )

        # Visualization settings
        if 'visualization' in data:
            vis = data['visualization']
            config.show_trajectories = vis.get('show_trajectories', True)
            config.trajectory_length = vis.get('trajectory_length', 50)
            config.camera_follow = vis.get('camera_follow')

        # Export settings
        if 'export' in data:
            exp = data['export']
            config.export_video = exp.get('video', False)
            config.export_data = exp.get('data', True)
            config.video_fps = exp.get('video_fps', 30)
            config.output_dir = exp.get('output_dir', 'output')

        return config

    @staticmethod
    def save(config: SimulationConfig, output_path: str):
        """
        Save configuration to YAML file

        Args:
            config: SimulationConfig to save
            output_path: Path to output YAML file
        """
        data = {
            'simulation': {
                'name': config.simulation_name,
                'duration': config.duration,
                'time_step': config.time_step,
                'real_time': config.real_time
            },
            'room': {
                'dimensions': config.room.dimensions,
                'texture_path': config.room.texture_path,
                'is_video': config.room.is_video
            },
            'drones': [
                {
                    'count': d.count,
                    'initial_positions': d.initial_positions,
                    'max_speed': d.max_speed,
                    'max_acceleration': d.max_acceleration,
                    'size': d.size,
                    'gif_path': d.gif_path,
                    'color': d.color,
                    'route_type': d.route_type,
                    'route_params': d.route_params
                }
                for d in config.drones
            ],
            'flight_model': {
                'type': config.flight_model.type,
                'gif_path': config.flight_model.gif_path,
                'params': config.flight_model.params
            },
            'avoidance': {
                'type': config.avoidance.type,
                'detection_radius': config.avoidance.detection_radius,
                'params': config.avoidance.params
            },
            'visualization': {
                'show_trajectories': config.show_trajectories,
                'trajectory_length': config.trajectory_length,
                'camera_follow': config.camera_follow
            },
            'export': {
                'video': config.export_video,
                'data': config.export_data,
                'video_fps': config.video_fps,
                'output_dir': config.output_dir
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def create_default_config(output_path: str):
        """
        Create a default configuration file

        Args:
            output_path: Path to save default configuration
        """
        config = SimulationConfig()

        # Add default drone configuration
        config.drones.append(DroneConfig(
            count=3,
            route_type="circular",
            route_params={'radius': 5.0, 'height': 5.0}
        ))

        ConfigLoader.save(config, output_path)
