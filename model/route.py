"""
Route and waypoint definitions for drone navigation
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Waypoint:
    """A single waypoint in 3D space"""
    position: np.ndarray
    tolerance: float = 0.5  # Distance threshold to consider waypoint reached (meters)
    hover_time: float = 0.0  # Time to hover at this waypoint (seconds)

    def __post_init__(self):
        """Ensure position is a numpy array"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)

    def is_reached(self, current_position: np.ndarray) -> bool:
        """Check if a position is within tolerance of this waypoint"""
        distance = np.linalg.norm(current_position - self.position)
        return distance <= self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'position': self.position.tolist(),
            'tolerance': self.tolerance,
            'hover_time': self.hover_time
        }


class Route:
    """
    A sequence of waypoints defining a drone's path
    """

    def __init__(self, waypoints: List[Waypoint], loop: bool = False):
        """
        Initialize a route

        Args:
            waypoints: List of waypoints to visit
            loop: If True, return to start after completing route
        """
        self.waypoints = waypoints
        self.loop = loop

    @classmethod
    def from_positions(cls, positions: List[List[float]], tolerance: float = 0.5, loop: bool = False) -> 'Route':
        """
        Create a route from a list of positions

        Args:
            positions: List of [x, y, z] positions
            tolerance: Distance threshold for each waypoint
            loop: Whether to loop back to start
        """
        waypoints = [Waypoint(np.array(pos), tolerance) for pos in positions]
        if loop and len(waypoints) > 0:
            # Add starting position as final waypoint for looping
            waypoints.append(Waypoint(waypoints[0].position.copy(), tolerance))
        return cls(waypoints, loop)

    @classmethod
    def circular_route(cls, center: np.ndarray, radius: float, height: float,
                      num_points: int = 8, loop: bool = True) -> 'Route':
        """
        Generate a circular route

        Args:
            center: Center point [x, y, z]
            radius: Radius of circle in XY plane
            height: Z coordinate for all points
            num_points: Number of waypoints around the circle
            loop: Whether to loop continuously
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        positions = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions.append([x, y, height])
        return cls.from_positions(positions, loop=loop)

    @classmethod
    def rectangular_route(cls, corner1: np.ndarray, corner2: np.ndarray,
                         height: float, loop: bool = True) -> 'Route':
        """
        Generate a rectangular route

        Args:
            corner1: First corner [x, y, z]
            corner2: Opposite corner [x, y, z]
            height: Z coordinate for all points
            loop: Whether to loop continuously
        """
        positions = [
            [corner1[0], corner1[1], height],
            [corner2[0], corner1[1], height],
            [corner2[0], corner2[1], height],
            [corner1[0], corner2[1], height],
        ]
        return cls.from_positions(positions, loop=loop)

    def get_total_distance(self) -> float:
        """Calculate the total path length"""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            diff = self.waypoints[i + 1].position - self.waypoints[i].position
            total += np.linalg.norm(diff)
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'waypoints': [wp.to_dict() for wp in self.waypoints],
            'loop': self.loop,
            'total_distance': self.get_total_distance()
        }
