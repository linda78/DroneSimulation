"""
Main simulation engine using SimPy for event-driven simulation
"""

import simpy
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from model import (
    Drone, Route, Environment, Room,
    FlightModel, PhysicalFlightModel, SimpleFlightModel, MPCFlightModel,
    AvoidanceAgent, RightAvoidanceAgent, RepulsiveAvoidanceAgent,
    VelocityObstacleAvoidanceAgent, MPCAvoidanceAgent
)
from .config import SimulationConfig, DroneConfig


class Simulation:
    """
    Main simulation controller using SimPy for discrete event simulation
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation from configuration

        Args:
            config: SimulationConfig object
        """
        self.config = config
        self.env = simpy.Environment()

        # Create room and environment
        self.room = Room(
            dimensions=tuple(config.room.dimensions),
            texture_path=config.room.texture_path,
            is_video=config.room.is_video
        )
        self.environment = Environment(self.room)

        # Initialize drones
        self.drones: List[Drone] = []
        self._create_drones()

        # Initialize flight model
        self.flight_model = self._create_flight_model()

        # Initialize avoidance agent
        self.avoidance_agent = self._create_avoidance_agent()

        # Simulation state
        self.current_time = 0.0
        self.is_running = False
        self.is_paused = False

        # Data recording
        self.state_history: List[Dict[str, Any]] = []

    def _create_drones(self):
        """Create drones based on configuration"""
        drone_id = 0
        security_sphere = self.config.avoidance.detection_radius

        for drone_config in self.config.drones:
            for i in range(drone_config.count):
                # Determine initial position
                if drone_config.initial_positions and i < len(drone_config.initial_positions):
                    initial_pos = np.array(drone_config.initial_positions[i])
                else:
                    # Random position within room
                    initial_pos = np.random.rand(3) * self.room.dimensions * 0.8 + 0.1 * self.room.dimensions

                # Create drone
                drone = Drone(
                    drone_id=drone_id,
                    initial_position=initial_pos,
                    max_speed=drone_config.max_speed,
                    max_acceleration=drone_config.max_acceleration,
                    size=drone_config.size,
                    security_sphere_size=security_sphere,
                    gif_path=drone_config.gif_path,
                    color=tuple(drone_config.color) if drone_config.color else None
                )

                # Create and assign route
                route = self._create_route(drone_config, initial_pos)
                drone.set_route(route)

                self.drones.append(drone)
                drone_id += 1

    def _create_route(self, drone_config: DroneConfig, initial_pos: np.ndarray) -> Route:
        """Create route based on configuration"""
        route_type = drone_config.route_type
        params = drone_config.route_params

        if route_type == "circular":
            center = np.array(params.get('center', self.room.dimensions / 2))
            radius = params.get('radius', 5.0)
            height = params.get('height', initial_pos[2])
            num_points = params.get('num_points', 8)
            return Route.circular_route(center, radius, height, num_points, loop=True)

        elif route_type == "rectangular":
            corner1 = np.array(params.get('corner1', [2, 2, 0]))
            corner2 = np.array(params.get('corner2', self.room.dimensions - [2, 2, 0]))
            height = params.get('height', initial_pos[2])
            return Route.rectangular_route(corner1, corner2, height, loop=True)

        elif route_type == "waypoints":
            positions = params.get('positions', [[10, 10, 5]])
            tolerance = params.get('tolerance', 0.5)
            loop = params.get('loop', True)
            return Route.from_positions(positions, tolerance, loop)

        else:
            # Default: hover at initial position
            return Route.from_positions([initial_pos.tolist()], loop=False)

    def _create_flight_model(self) -> FlightModel:
        """Create flight model from configuration"""
        model_type = self.config.flight_model.type
        params = self.config.flight_model.params
        gif_path = self.config.flight_model.gif_path

        if model_type == "physical":
            return PhysicalFlightModel(
                gif_path=gif_path,
                drag_coefficient=params.get('drag_coefficient', 0.1),
                approach_distance=params.get('approach_distance', 2.0)
            )
        elif model_type == "simple":
            return SimpleFlightModel(
                gif_path=gif_path,
                speed_factor=params.get('speed_factor', 1.0)
            )
        elif model_type == "mpc":
            return MPCFlightModel(
                gif_path=gif_path,
                dt=self.config.time_step
            )
        else:
            return PhysicalFlightModel(gif_path=gif_path)

    def _create_avoidance_agent(self) -> AvoidanceAgent:
        """Create avoidance agent from configuration"""
        agent_type = self.config.avoidance.type
        detection_radius = self.config.avoidance.detection_radius
        params = self.config.avoidance.params

        if agent_type == "right":
            return RightAvoidanceAgent(
                detection_radius=detection_radius,
                avoidance_strength=params.get('avoidance_strength', 1.0)
            )
        elif agent_type == "repulsive":
            return RepulsiveAvoidanceAgent(
                detection_radius=detection_radius,
                repulsion_strength=params.get('repulsion_strength', 2.0)
            )
        elif agent_type == "velocity_obstacle":
            return VelocityObstacleAvoidanceAgent(
                detection_radius=detection_radius,
                time_horizon=params.get('time_horizon', 2.0)
            )
        elif agent_type == "mpc":
            return MPCAvoidanceAgent(
                detection_radius=detection_radius,
                prediction_horizon=params.get('prediction_horizon', 10),
                dt=self.config.time_step,
                Q_weight=params.get('Q_weight', 1.0),
                R_weight=params.get('R_weight', 0.1),
                room_dimensions=self.room.dimensions,
                debug=params.get('debug', False)
            )
        else:
            return RightAvoidanceAgent(detection_radius=detection_radius)

    def update_drones(self):
        """Update all drones for one time step"""
        dt = self.config.time_step

        for drone in self.drones:
            # Get current target
            target = drone.get_current_target()
            if target is None:
                continue

            # Calculate avoidance vector
            avoidance_vector = self.avoidance_agent.calculate_avoidance(drone, self.drones)

            # Update drone state using flight model
            new_state = self.flight_model.update(drone, target, dt, self.drones, avoidance_vector)

            # Ensure drone stays within bounds
            new_state.position = self.environment.get_safe_position(new_state.position, drone.size)

            # Update drone
            drone.update_state(new_state)

            # Check if waypoint reached
            if drone.route:
                current_waypoint = drone.route.waypoints[drone.current_waypoint_index]
                if current_waypoint.is_reached(drone.state.position):
                    drone.advance_waypoint()

                    # Loop route if configured
                    if drone.route.loop and drone.has_reached_destination():
                        drone.current_waypoint_index = 0

    def simulation_process(self):
        """SimPy process for running simulation"""
        while self.current_time < self.config.duration:
            # Update drones
            self.update_drones()

            # Update video frame if using video texture
            if self.room.is_video and self.room.video_capture:
                self.room.update_video_frame()

            # Record state
            if self.config.export_data:
                self._record_state()

            # Advance time
            self.current_time += self.config.time_step
            yield self.env.timeout(self.config.time_step)

        self.is_running = False

    def _record_state(self):
        """Record current simulation state"""
        state = {
            'time': self.current_time,
            'drones': [drone.to_dict() for drone in self.drones]
        }
        self.state_history.append(state)

    def run(self):
        """Run the simulation"""
        self.is_running = True
        self.env.process(self.simulation_process())
        self.env.run()

    def step(self):
        """Execute one simulation step (for interactive mode)"""
        if self.current_time < self.config.duration:
            self.update_drones()

            if self.room.is_video and self.room.video_capture:
                self.room.update_video_frame()

            if self.config.export_data:
                self._record_state()

            self.current_time += self.config.time_step
            return True
        return False

    def reset(self):
        """Reset simulation to initial state"""
        self.env = simpy.Environment()
        self.drones.clear()
        self._create_drones()
        self.current_time = 0.0
        self.state_history.clear()
        self.is_running = False
        self.is_paused = False

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'time': self.current_time,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'drones': [drone.to_dict() for drone in self.drones],
            'room_dimensions': self.room.dimensions.tolist()
        }

    def cleanup(self):
        """Cleanup resources"""
        self.room.cleanup()
