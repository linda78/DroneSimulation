"""
Unit tests for drone model
"""

import unittest
import numpy as np
from model.drone import Drone, DroneState
from model.route import Route, Waypoint


class TestDroneState(unittest.TestCase):
    """Test DroneState dataclass"""

    def test_default_initialization(self):
        """Test default state initialization"""
        state = DroneState()

        np.testing.assert_array_equal(state.position, np.zeros(3))
        np.testing.assert_array_equal(state.velocity, np.zeros(3))
        np.testing.assert_array_equal(state.acceleration, np.zeros(3))
        np.testing.assert_array_equal(state.orientation, np.array([0., 0., 0.]))

    def test_custom_initialization(self):
        """Test state with custom values"""
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.5, 0.5, 0.5])
        acc = np.array([0.1, 0.1, 0.1])
        orient = np.array([10., 20., 30.])

        state = DroneState(position=pos, velocity=vel, acceleration=acc, orientation=orient)

        np.testing.assert_array_equal(state.position, pos)
        np.testing.assert_array_equal(state.velocity, vel)
        np.testing.assert_array_equal(state.acceleration, acc)
        np.testing.assert_array_equal(state.orientation, orient)

    def test_copy(self):
        """Test state copy creates independent copy"""
        state1 = DroneState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.5, 0.5, 0.5])
        )
        state2 = state1.copy()

        # Modify copy
        state2.position[0] = 999.0
        state2.velocity[0] = 888.0

        # Original should be unchanged
        self.assertEqual(state1.position[0], 1.0)
        self.assertEqual(state1.velocity[0], 0.5)


class TestDrone(unittest.TestCase):
    """Test Drone class"""

    def test_initialization(self):
        """Test basic drone initialization"""
        pos = np.array([5.0, 5.0, 2.0])
        drone = Drone(
            drone_id=1,
            initial_position=pos,
            max_speed=10.0,
            max_acceleration=5.0,
            size=0.5
        )

        self.assertEqual(drone.id, 1)
        np.testing.assert_array_equal(drone.state.position, pos)
        self.assertEqual(drone.max_speed, 10.0)
        self.assertEqual(drone.max_acceleration, 5.0)
        self.assertEqual(drone.size, 0.5)
        self.assertIsNone(drone.route)

    def test_color_generation(self):
        """Test that drones generate valid colors"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        self.assertIsInstance(drone.color, tuple)
        self.assertEqual(len(drone.color), 3)
        for component in drone.color:
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)

    def test_custom_color(self):
        """Test setting custom color"""
        color = (0.5, 0.7, 0.9)
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]), color=color)

        self.assertEqual(drone.color, color)

    def test_update_state(self):
        """Test state update and trajectory history"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        # Initial position should be in history
        self.assertEqual(len(drone.trajectory_history), 1)

        # Update state
        new_state = DroneState(position=np.array([1.0, 1.0, 1.0]))
        drone.update_state(new_state)

        # Check state updated
        np.testing.assert_array_equal(drone.state.position, np.array([1.0, 1.0, 1.0]))

        # Check history updated
        self.assertEqual(len(drone.trajectory_history), 2)
        np.testing.assert_array_equal(drone.trajectory_history[-1], np.array([1.0, 1.0, 1.0]))

    def test_trajectory_history_limit(self):
        """Test that trajectory history respects max length"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))
        drone.max_history_length = 10

        # Add many positions
        for i in range(20):
            new_state = DroneState(position=np.array([float(i), 0, 0]))
            drone.update_state(new_state)

        # Should not exceed max length
        self.assertLessEqual(len(drone.trajectory_history), drone.max_history_length)

    def test_route_assignment(self):
        """Test route assignment and navigation"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        waypoints = [
            Waypoint(np.array([1.0, 0, 0])),
            Waypoint(np.array([1.0, 1.0, 0])),
            Waypoint(np.array([0, 1.0, 0]))
        ]
        route = Route(waypoints)

        drone.set_route(route)

        self.assertEqual(drone.route, route)
        self.assertEqual(drone.current_waypoint_index, 0)

    def test_get_current_target(self):
        """Test getting current waypoint target"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        # No route
        self.assertIsNone(drone.get_current_target())

        # With route
        waypoints = [
            Waypoint(np.array([1.0, 0, 0])),
            Waypoint(np.array([2.0, 0, 0]))
        ]
        route = Route(waypoints)
        drone.set_route(route)

        target = drone.get_current_target()
        np.testing.assert_array_equal(target, np.array([1.0, 0, 0]))

    def test_advance_waypoint(self):
        """Test advancing to next waypoint"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        waypoints = [
            Waypoint(np.array([1.0, 0, 0])),
            Waypoint(np.array([2.0, 0, 0])),
            Waypoint(np.array([3.0, 0, 0]))
        ]
        route = Route(waypoints)
        drone.set_route(route)

        # Advance through waypoints
        self.assertEqual(drone.current_waypoint_index, 0)

        drone.advance_waypoint()
        self.assertEqual(drone.current_waypoint_index, 1)

        drone.advance_waypoint()
        self.assertEqual(drone.current_waypoint_index, 2)

        # Should not advance beyond last waypoint
        drone.advance_waypoint()
        self.assertEqual(drone.current_waypoint_index, 2)

    def test_has_reached_destination(self):
        """Test destination check"""
        drone = Drone(drone_id=1, initial_position=np.array([0, 0, 0]))

        # No route means reached
        self.assertTrue(drone.has_reached_destination())

        # With route
        waypoints = [Waypoint(np.array([1.0, 0, 0]))]
        route = Route(waypoints)
        drone.set_route(route)

        self.assertFalse(drone.has_reached_destination())

        # Advance past last waypoint
        drone.current_waypoint_index = 1
        self.assertTrue(drone.has_reached_destination())

    def test_to_dict(self):
        """Test serialization to dictionary"""
        drone = Drone(
            drone_id=5,
            initial_position=np.array([1.0, 2.0, 3.0]),
            color=(0.5, 0.6, 0.7)
        )

        data = drone.to_dict()

        self.assertEqual(data['id'], 5)
        self.assertEqual(data['position'], [1.0, 2.0, 3.0])
        self.assertEqual(data['color'], (0.5, 0.6, 0.7))
        self.assertIn('velocity', data)
        self.assertIn('acceleration', data)
        self.assertIn('trajectory', data)


if __name__ == '__main__':
    unittest.main()
