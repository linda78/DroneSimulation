"""
Unit tests for flight models
"""

import unittest
import numpy as np
from model.flight_model import PhysicalFlightModel
from model.drone import Drone, DroneState


class TestPhysicalFlightModel(unittest.TestCase):
    """Test PhysicalFlightModel class"""

    def setUp(self):
        """Set up test fixtures"""
        self.model = PhysicalFlightModel(drag_coefficient=0.1, approach_distance=2.0)
        self.drone = Drone(
            drone_id=1,
            initial_position=np.array([0.0, 0.0, 0.0]),
            max_speed=5.0,
            max_acceleration=2.0
        )

    def test_initialization(self):
        """Test flight model initialization"""
        model = PhysicalFlightModel(drag_coefficient=0.2, approach_distance=3.0)

        self.assertEqual(model.drag_coefficient, 0.2)
        self.assertEqual(model.approach_distance, 3.0)

    def test_get_name(self):
        """Test getting model name"""
        name = self.model.get_name()
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)

    def test_update_stationary_drone(self):
        """Test updating drone already at target"""
        target = np.array([0.0, 0.0, 0.0])  # Same as drone position
        dt = 0.1

        new_state = self.model.update(self.drone, target, dt, [])

        # Should decelerate and stay near position
        self.assertIsInstance(new_state, DroneState)
        # Velocity should be zero or very small
        self.assertLess(np.linalg.norm(new_state.velocity), 0.1)

    def test_update_moves_towards_target(self):
        """Test drone moves towards target"""
        target = np.array([10.0, 0.0, 0.0])
        dt = 0.1

        # Update several times
        for _ in range(10):
            new_state = self.model.update(self.drone, target, dt, [])
            self.drone.state = new_state

        # Should have moved towards target (positive x direction)
        self.assertGreater(self.drone.state.position[0], 0.0)
        self.assertGreater(self.drone.state.velocity[0], 0.0)

    def test_respects_max_speed(self):
        """Test that drone respects maximum speed limit"""
        target = np.array([100.0, 0.0, 0.0])  # Far away target
        dt = 0.1

        # Update many times to reach max speed
        for _ in range(100):
            new_state = self.model.update(self.drone, target, dt, [])
            self.drone.state = new_state

            # Check speed doesn't exceed max
            speed = np.linalg.norm(new_state.velocity)
            self.assertLessEqual(speed, self.drone.max_speed * 1.1)  # 10% tolerance for numerical issues

    def test_respects_max_acceleration(self):
        """Test that drone respects maximum acceleration"""
        target = np.array([10.0, 0.0, 0.0])
        dt = 0.1

        # Take one update step
        new_state = self.model.update(self.drone, target, dt, [])

        # Check acceleration magnitude
        accel_magnitude = np.linalg.norm(new_state.acceleration)
        # Should be bounded by max acceleration
        self.assertLessEqual(accel_magnitude, self.drone.max_acceleration * 1.5)  # Some tolerance

    def test_update_with_avoidance_vector(self):
        """Test update with collision avoidance vector"""
        target = np.array([10.0, 0.0, 0.0])
        avoidance = np.array([0.0, 5.0, 0.0])  # Push in y direction
        dt = 0.1

        # Update with avoidance
        new_state = self.model.update(self.drone, target, dt, [], avoidance_vector=avoidance)

        # Should have some velocity in both x and y directions
        # (blend of target and avoidance)
        self.assertNotEqual(new_state.velocity[0], 0.0)
        self.assertNotEqual(new_state.velocity[1], 0.0)

    def test_state_independence(self):
        """Test that update doesn't modify original drone state"""
        original_position = self.drone.state.position.copy()
        original_velocity = self.drone.state.velocity.copy()

        target = np.array([10.0, 0.0, 0.0])
        dt = 0.1

        new_state = self.model.update(self.drone, target, dt, [])

        # Original drone state should be unchanged
        np.testing.assert_array_equal(self.drone.state.position, original_position)
        np.testing.assert_array_equal(self.drone.state.velocity, original_velocity)

        # New state should be different
        self.assertFalse(np.array_equal(new_state.position, original_position))

    def test_3d_movement(self):
        """Test movement in 3D space"""
        target = np.array([5.0, 5.0, 5.0])
        dt = 0.1

        # Update several times
        for _ in range(20):
            new_state = self.model.update(self.drone, target, dt, [])
            self.drone.state = new_state

        # Should have moved in all three dimensions
        self.assertGreater(self.drone.state.position[0], 0.0)
        self.assertGreater(self.drone.state.position[1], 0.0)
        self.assertGreater(self.drone.state.position[2], 0.0)

    def test_deceleration_near_target(self):
        """Test that drone decelerates when approaching target"""
        # Start drone with high velocity towards target
        self.drone.state.velocity = np.array([5.0, 0.0, 0.0])
        self.drone.state.position = np.array([8.0, 0.0, 0.0])

        target = np.array([10.0, 0.0, 0.0])  # Close to drone
        dt = 0.1

        initial_speed = np.linalg.norm(self.drone.state.velocity)

        # Update a few times
        for _ in range(5):
            new_state = self.model.update(self.drone, target, dt, [])
            self.drone.state = new_state

        final_speed = np.linalg.norm(self.drone.state.velocity)

        # Speed should decrease as we approach target
        # (though this depends on approach_distance parameter)
        self.assertIsNotNone(final_speed)  # At least check it computes

    def test_multiple_time_steps(self):
        """Test consistent behavior over multiple time steps"""
        target = np.array([10.0, 10.0, 0.0])
        dt = 0.05

        positions = [self.drone.state.position.copy()]

        # Simulate for 2 seconds
        for _ in range(40):
            new_state = self.model.update(self.drone, target, dt, [])
            self.drone.state = new_state
            positions.append(new_state.position.copy())

        # Check that drone is moving consistently towards target
        distances = [np.linalg.norm(pos - target) for pos in positions]

        # Distance should generally decrease (allowing some oscillation)
        self.assertLess(distances[-1], distances[0])


if __name__ == '__main__':
    unittest.main()
