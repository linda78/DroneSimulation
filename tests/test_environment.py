"""
Unit tests for environment and room classes
"""

import unittest
import numpy as np
from model.environment import Room, Environment


class TestRoom(unittest.TestCase):
    """Test Room class"""

    def test_default_initialization(self):
        """Test room with default dimensions"""
        room = Room()

        np.testing.assert_array_equal(room.dimensions, np.array([20.0, 20.0, 10.0]))
        self.assertIsNone(room.texture_path)
        self.assertFalse(room.is_video)

    def test_custom_dimensions(self):
        """Test room with custom dimensions"""
        dims = (30.0, 25.0, 15.0)
        room = Room(dimensions=dims)

        np.testing.assert_array_equal(room.dimensions, np.array([30.0, 25.0, 15.0]))

    def test_is_inside_true(self):
        """Test positions inside room"""
        room = Room(dimensions=(10.0, 10.0, 10.0))

        # Center of room
        self.assertTrue(room.is_inside(np.array([5.0, 5.0, 5.0])))

        # At origin
        self.assertTrue(room.is_inside(np.array([0.0, 0.0, 0.0])))

        # At max bounds
        self.assertTrue(room.is_inside(np.array([10.0, 10.0, 10.0])))

        # Just inside bounds
        self.assertTrue(room.is_inside(np.array([9.9, 9.9, 9.9])))

    def test_is_inside_false(self):
        """Test positions outside room"""
        room = Room(dimensions=(10.0, 10.0, 10.0))

        # Negative coordinates
        self.assertFalse(room.is_inside(np.array([-0.1, 5.0, 5.0])))
        self.assertFalse(room.is_inside(np.array([5.0, -1.0, 5.0])))

        # Beyond max bounds
        self.assertFalse(room.is_inside(np.array([10.1, 5.0, 5.0])))
        self.assertFalse(room.is_inside(np.array([5.0, 5.0, 15.0])))

    def test_clamp_position(self):
        """Test position clamping to room bounds"""
        room = Room(dimensions=(10.0, 10.0, 10.0))

        # Position inside - unchanged
        pos = np.array([5.0, 5.0, 5.0])
        clamped = room.clamp_position(pos)
        np.testing.assert_array_equal(clamped, pos)

        # Position outside - clamped to bounds
        pos = np.array([-1.0, 5.0, 15.0])
        clamped = room.clamp_position(pos)
        np.testing.assert_array_equal(clamped, np.array([0.0, 5.0, 10.0]))

        # All coordinates outside
        pos = np.array([-5.0, 20.0, -10.0])
        clamped = room.clamp_position(pos)
        np.testing.assert_array_equal(clamped, np.array([0.0, 10.0, 0.0]))

    def test_get_bounds(self):
        """Test getting room bounds"""
        room = Room(dimensions=(15.0, 12.0, 8.0))

        min_bound, max_bound = room.get_bounds()

        np.testing.assert_array_equal(min_bound, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(max_bound, np.array([15.0, 12.0, 8.0]))


class TestEnvironment(unittest.TestCase):
    """Test Environment class"""

    def test_initialization(self):
        """Test environment initialization"""
        room = Room(dimensions=(20.0, 20.0, 10.0))
        env = Environment(room)

        self.assertEqual(env.room, room)
        self.assertEqual(len(env.obstacles), 0)

    def test_check_collision_no_collision(self):
        """Test collision detection when no collision"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Position well inside room
        collision = env.check_collision(np.array([5.0, 5.0, 5.0]), size=0.5)
        self.assertFalse(collision)

    def test_check_collision_with_walls(self):
        """Test collision detection with walls"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Too close to left wall (x=0)
        collision = env.check_collision(np.array([0.3, 5.0, 5.0]), size=0.5)
        self.assertTrue(collision)

        # Too close to right wall (x=10)
        collision = env.check_collision(np.array([9.8, 5.0, 5.0]), size=0.5)
        self.assertTrue(collision)

        # Too close to floor (z=0)
        collision = env.check_collision(np.array([5.0, 5.0, 0.3]), size=0.5)
        self.assertTrue(collision)

        # Too close to ceiling (z=10)
        collision = env.check_collision(np.array([5.0, 5.0, 9.8]), size=0.5)
        self.assertTrue(collision)

    def test_check_collision_edge_cases(self):
        """Test collision at exact boundaries"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Exactly at boundary with size
        collision = env.check_collision(np.array([0.5, 5.0, 5.0]), size=0.5)
        self.assertFalse(collision)  # Just fits

        # Just beyond boundary
        collision = env.check_collision(np.array([0.4, 5.0, 5.0]), size=0.5)
        self.assertTrue(collision)

    def test_get_safe_position_inside(self):
        """Test getting safe position for position already inside"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Already safe position
        pos = np.array([5.0, 5.0, 5.0])
        safe_pos = env.get_safe_position(pos, size=0.5)
        np.testing.assert_array_equal(safe_pos, pos)

    def test_get_safe_position_outside(self):
        """Test getting safe position for position outside room"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Position outside - should be clamped with size offset
        pos = np.array([-1.0, 5.0, 15.0])
        safe_pos = env.get_safe_position(pos, size=0.5)

        # Should be at least 'size' away from walls
        np.testing.assert_array_equal(safe_pos, np.array([0.5, 5.0, 9.5]))

    def test_get_safe_position_corners(self):
        """Test safe position near corners"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        # Try to place at corner
        pos = np.array([0.0, 0.0, 0.0])
        safe_pos = env.get_safe_position(pos, size=1.0)

        # Should be pushed inside by size amount
        np.testing.assert_array_equal(safe_pos, np.array([1.0, 1.0, 1.0]))

        # Try opposite corner
        pos = np.array([10.0, 10.0, 10.0])
        safe_pos = env.get_safe_position(pos, size=1.0)
        np.testing.assert_array_equal(safe_pos, np.array([9.0, 9.0, 9.0]))

    def test_different_drone_sizes(self):
        """Test collision detection with different drone sizes"""
        room = Room(dimensions=(10.0, 10.0, 10.0))
        env = Environment(room)

        position = np.array([1.0, 5.0, 5.0])

        # Small drone - no collision
        self.assertFalse(env.check_collision(position, size=0.5))

        # Medium drone - collision
        self.assertTrue(env.check_collision(position, size=1.5))

        # Large drone - definitely collision
        self.assertTrue(env.check_collision(position, size=2.0))


if __name__ == '__main__':
    unittest.main()
