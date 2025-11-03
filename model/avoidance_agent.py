"""
Collision avoidance agents for drone navigation
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .drone import Drone


class AvoidanceAgent(ABC):
    """
    Abstract base class for collision avoidance strategies
    """

    @abstractmethod
    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance vector based on nearby drones

        Args:
            drone: The drone to calculate avoidance for
            other_drones: List of other drones in the environment

        Returns:
            Avoidance vector (direction to move), or None if no avoidance needed
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this avoidance agent"""
        pass


class RightAvoidanceAgent(AvoidanceAgent):
    """
    Simple avoidance agent: always try to avoid to the right
    Uses "right-hand rule" like traffic conventions
    """

    def __init__(self, detection_radius: float = 3.0, avoidance_strength: float = 1.0):
        """
        Initialize right avoidance agent

        Args:
            detection_radius: Distance at which to start avoiding (meters)
            avoidance_strength: Strength of avoidance response (multiplier)
        """
        self.detection_radius = detection_radius
        self.avoidance_strength = avoidance_strength

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance by moving to the right of approaching drones

        Strategy:
        1. Find drones within detection radius
        2. For each nearby drone, calculate "right" direction
        3. Combine all avoidance vectors
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Calculate distance and direction to other drone
            to_other = other.state.position - drone.state.position
            distance = np.linalg.norm(to_other)

            # Only avoid if within detection radius
            if distance < self.detection_radius and distance > 0.01:
                # Calculate right direction (perpendicular to line between drones)
                # In 3D, "right" is perpendicular in XY plane
                to_other_normalized = to_other / distance

                # Calculate right vector (rotate 90 degrees in XY plane)
                right_vector = np.array([
                    -to_other_normalized[1],  # Right in XY plane
                    to_other_normalized[0],
                    0.0  # No vertical component
                ])

                # Strength inversely proportional to distance
                strength = (1.0 - distance / self.detection_radius) * self.avoidance_strength

                # Add to avoidance vector
                avoidance_vector += right_vector * strength
                num_threats += 1

        if num_threats == 0:
            return None

        # Normalize and return
        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "RightAvoidanceAgent"


class RepulsiveAvoidanceAgent(AvoidanceAgent):
    """
    Avoidance agent using repulsive forces (move directly away from threats)
    """

    def __init__(self, detection_radius: float = 3.0, repulsion_strength: float = 2.0):
        """
        Initialize repulsive avoidance agent

        Args:
            detection_radius: Distance at which to start avoiding (meters)
            repulsion_strength: Strength of repulsion force
        """
        self.detection_radius = detection_radius
        self.repulsion_strength = repulsion_strength

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance by moving away from nearby drones

        Uses inverse-square law: closer drones have stronger repulsion
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Calculate distance and direction
            to_other = other.state.position - drone.state.position
            distance = np.linalg.norm(to_other)

            # Only avoid if within detection radius
            if distance < self.detection_radius and distance > 0.01:
                # Direction away from other drone
                away_direction = -to_other / distance

                # Inverse square law for repulsion strength
                strength = self.repulsion_strength * (self.detection_radius / distance) ** 2

                avoidance_vector += away_direction * strength
                num_threats += 1

        if num_threats == 0:
            return None

        # Normalize and return
        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "RepulsiveAvoidanceAgent"


class VelocityObstacleAvoidanceAgent(AvoidanceAgent):
    """
    Advanced avoidance using velocity obstacle concept
    Considers both position and velocity of other drones
    """

    def __init__(self, detection_radius: float = 3.0, time_horizon: float = 2.0):
        """
        Initialize velocity obstacle avoidance

        Args:
            detection_radius: Distance at which to consider drones
            time_horizon: Time to predict collisions (seconds)
        """
        self.detection_radius = detection_radius
        self.time_horizon = time_horizon

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance considering future collisions based on current velocities
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Current separation
            relative_position = other.state.position - drone.state.position
            distance = np.linalg.norm(relative_position)

            if distance > self.detection_radius:
                continue

            # Relative velocity
            relative_velocity = other.state.velocity - drone.state.velocity

            # Time to closest approach
            if distance > 0.01:
                rel_pos_norm = relative_position / distance
                closing_speed = np.dot(relative_velocity, rel_pos_norm)

                # If moving towards each other
                if closing_speed > 0:
                    time_to_collision = distance / (closing_speed + 1e-6)

                    # If collision likely within time horizon
                    if time_to_collision < self.time_horizon:
                        # Avoid perpendicular to relative velocity
                        if np.linalg.norm(relative_velocity) > 0.01:
                            # Move perpendicular to collision course
                            perp_vector = np.cross(relative_velocity, [0, 0, 1])
                            if np.linalg.norm(perp_vector) > 0.01:
                                perp_vector = perp_vector / np.linalg.norm(perp_vector)
                                strength = (self.time_horizon - time_to_collision) / self.time_horizon
                                avoidance_vector += perp_vector * strength
                                num_threats += 1

        if num_threats == 0:
            return None

        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "VelocityObstacleAvoidanceAgent"
