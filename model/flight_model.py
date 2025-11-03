"""
Flight models for drone physics simulation
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .drone import Drone, DroneState


class FlightModel(ABC):
    """
    Abstract base class for drone flight physics models
    """

    @abstractmethod
    def update(self, drone: Drone, target: np.ndarray, dt: float,
               other_drones: List[Drone], avoidance_vector: Optional[np.ndarray] = None) -> DroneState:
        """
        Calculate new drone state based on target and physics

        Args:
            drone: The drone to update
            target: Target position to move towards
            dt: Time step in seconds
            other_drones: List of other drones for collision avoidance
            avoidance_vector: Optional avoidance vector from collision avoidance agent

        Returns:
            New drone state
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this flight model"""
        pass


class PhysicalFlightModel(FlightModel):
    """
    Physically realistic flight model with acceleration and deceleration
    Implements smooth motion with velocity and acceleration constraints
    """

    def __init__(
        self,
        gif_path: Optional[str] = None,
        drag_coefficient: float = 0.1,
        approach_distance: float = 2.0
    ):
        """
        Initialize physical flight model

        Args:
            gif_path: Path to GIF for this model's visualization
            drag_coefficient: Air drag coefficient (0-1)
            approach_distance: Distance at which to start decelerating
        """
        self.gif_path = gif_path
        self.drag_coefficient = drag_coefficient
        self.approach_distance = approach_distance

    def update(self, drone: Drone, target: np.ndarray, dt: float,
               other_drones: List[Drone], avoidance_vector: Optional[np.ndarray] = None) -> DroneState:
        """
        Update drone state with physical acceleration/deceleration

        Uses PID-like control for smooth movement:
        - Accelerates towards target
        - Decelerates when approaching waypoint
        - Applies drag for stability
        - Respects max speed and acceleration limits
        """
        new_state = drone.state.copy()

        # Calculate direction to target
        to_target = target - new_state.position
        distance = np.linalg.norm(to_target)

        if distance < 0.01:  # Already at target
            # Apply deceleration
            new_state.acceleration = -new_state.velocity * 2.0
            new_state.velocity += new_state.acceleration * dt
            new_state.velocity *= (1.0 - self.drag_coefficient * dt)
            new_state.position += new_state.velocity * dt
            return new_state

        direction = to_target / distance

        # Apply avoidance if provided
        if avoidance_vector is not None and np.linalg.norm(avoidance_vector) > 0.01:
            avoidance_norm = avoidance_vector / np.linalg.norm(avoidance_vector)
            # Blend target direction with avoidance (70% avoidance, 30% target)
            direction = 0.3 * direction + 0.7 * avoidance_norm
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Calculate desired velocity based on distance
        desired_speed = drone.max_speed

        # Slow down when approaching target
        if distance < self.approach_distance:
            desired_speed = drone.max_speed * (distance / self.approach_distance)

        desired_velocity = direction * desired_speed

        # Calculate acceleration to reach desired velocity
        velocity_error = desired_velocity - new_state.velocity
        new_state.acceleration = velocity_error / dt

        # Limit acceleration
        acc_magnitude = np.linalg.norm(new_state.acceleration)
        if acc_magnitude > drone.max_acceleration:
            new_state.acceleration = (new_state.acceleration / acc_magnitude) * drone.max_acceleration

        # Update velocity
        new_state.velocity += new_state.acceleration * dt

        # Apply drag
        new_state.velocity *= (1.0 - self.drag_coefficient * dt)

        # Limit speed
        speed = np.linalg.norm(new_state.velocity)
        if speed > drone.max_speed:
            new_state.velocity = (new_state.velocity / speed) * drone.max_speed

        # Update position
        new_state.position += new_state.velocity * dt

        # Update orientation (yaw to face direction of movement)
        if speed > 0.1:
            new_state.orientation[2] = np.arctan2(new_state.velocity[1], new_state.velocity[0])

        return new_state

    def get_name(self) -> str:
        return "PhysicalFlightModel"


class SimpleFlightModel(FlightModel):
    """
    Simple flight model that moves directly towards target at constant speed
    No acceleration/deceleration physics
    """

    def __init__(self, gif_path: Optional[str] = None, speed_factor: float = 1.0):
        """
        Initialize simple flight model

        Args:
            gif_path: Path to GIF for visualization
            speed_factor: Speed multiplier (0-1)
        """
        self.gif_path = gif_path
        self.speed_factor = speed_factor

    def update(self, drone: Drone, target: np.ndarray, dt: float,
               other_drones: List[Drone], avoidance_vector: Optional[np.ndarray] = None) -> DroneState:
        """
        Simple linear movement towards target
        """
        new_state = drone.state.copy()

        to_target = target - new_state.position
        distance = np.linalg.norm(to_target)

        if distance < 0.01:
            new_state.velocity = np.zeros(3)
            return new_state

        direction = to_target / distance

        # Apply avoidance if provided
        if avoidance_vector is not None and np.linalg.norm(avoidance_vector) > 0.01:
            avoidance_norm = avoidance_vector / np.linalg.norm(avoidance_vector)
            direction = 0.5 * direction + 0.5 * avoidance_norm
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Move at constant speed
        speed = min(drone.max_speed * self.speed_factor, distance / dt)
        new_state.velocity = direction * speed
        new_state.position += new_state.velocity * dt

        # Update orientation
        if speed > 0.1:
            new_state.orientation[2] = np.arctan2(new_state.velocity[1], new_state.velocity[0])

        return new_state

    def get_name(self) -> str:
        return "SimpleFlightModel"
