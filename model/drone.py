"""
Drone model with physics state and configuration
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DroneState:
    """Physical state of a drone"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))  # roll, pitch, yaw

    def copy(self) -> 'DroneState':
        """Create a deep copy of the state"""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            orientation=self.orientation.copy()
        )


class Drone:
    """
    Drone entity with physics state and visualization
    """

    def __init__(
        self,
        drone_id: int,
        initial_position: np.ndarray,
        max_speed: float = 5.0,
        max_acceleration: float = 2.0,
        size: float = 0.3,
        security_sphere_size: Optional[float] = None,
        gif_path: Optional[str] = None,
        color: Optional[tuple] = None
    ):
        """
        Initialize a drone

        Args:
            drone_id: Unique identifier
            initial_position: Starting position [x, y, z]
            max_speed: Maximum velocity in m/s
            max_acceleration: Maximum acceleration in m/s^2
            size: Drone size for collision detection (radius in meters)
            gif_path: Path to GIF for visualization
            color: RGB color tuple (0-1 range)
        """
        self.id = drone_id
        self.state = DroneState(position=np.array(initial_position, dtype=float))
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.size = size
        self.security_sphere_size = security_sphere_size
        self.gif_path = gif_path
        self.color = color if color else self._generate_color()

        # Route and navigation
        self.route = None
        self.current_waypoint_index = 0

        # History for visualization
        self.trajectory_history = [self.state.position.copy()]
        self.max_history_length = 1000

    def _generate_color(self) -> tuple:
        """Generate a random color for the drone"""
        return tuple(np.random.rand(3))

    def update_state(self, new_state: DroneState):
        """Update the drone's physical state"""
        self.state = new_state

        # Add to trajectory history
        self.trajectory_history.append(new_state.position.copy())
        if len(self.trajectory_history) > self.max_history_length:
            self.trajectory_history.pop(0)

    def set_route(self, route):
        """Assign a route to this drone"""
        self.route = route
        self.current_waypoint_index = 0

    def get_current_target(self) -> Optional[np.ndarray]:
        """Get the current waypoint target"""
        if self.route is None or self.current_waypoint_index >= len(self.route.waypoints):
            return None
        return self.route.waypoints[self.current_waypoint_index].position

    def advance_waypoint(self):
        """Move to the next waypoint"""
        if self.route and self.current_waypoint_index < len(self.route.waypoints) - 1:
            self.current_waypoint_index += 1

    def has_reached_destination(self) -> bool:
        """Check if drone has completed its route"""
        if self.route is None:
            return True
        return self.current_waypoint_index >= len(self.route.waypoints)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize drone state to dictionary"""
        return {
            'id': self.id,
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'acceleration': self.state.acceleration.tolist(),
            'orientation': self.state.orientation.tolist(),
            'color': self.color,
            'waypoint_index': self.current_waypoint_index,
            'trajectory': [pos.tolist() for pos in self.trajectory_history[-50:]]  # Last 50 positions
        }
