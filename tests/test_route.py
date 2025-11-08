"""
Unit tests for route and waypoint classes
"""

import unittest
import numpy as np
from model.route import Waypoint, Route


class TestWaypoint(unittest.TestCase):
    """Test Waypoint class"""

    def test_initialization(self):
        """Test waypoint initialization"""
        pos = np.array([1.0, 2.0, 3.0])
        wp = Waypoint(position=pos, tolerance=0.5, hover_time=2.0)

        np.testing.assert_array_equal(wp.position, pos)
        self.assertEqual(wp.tolerance, 0.5)
        self.assertEqual(wp.hover_time, 2.0)

    def test_default_values(self):
        """Test default tolerance and hover time"""
        wp = Waypoint(position=np.array([0, 0, 0]))

        self.assertEqual(wp.tolerance, 0.5)
        self.assertEqual(wp.hover_time, 0.0)

    def test_position_conversion(self):
        """Test that list positions are converted to numpy arrays"""
        wp = Waypoint(position=[1.0, 2.0, 3.0])

        self.assertIsInstance(wp.position, np.ndarray)
        np.testing.assert_array_equal(wp.position, np.array([1.0, 2.0, 3.0]))

    def test_is_reached_within_tolerance(self):
        """Test waypoint reached when within tolerance"""
        wp = Waypoint(position=np.array([5.0, 5.0, 5.0]), tolerance=1.0)

        # Exactly at waypoint
        self.assertTrue(wp.is_reached(np.array([5.0, 5.0, 5.0])))

        # Within tolerance
        self.assertTrue(wp.is_reached(np.array([5.5, 5.5, 5.0])))
        self.assertTrue(wp.is_reached(np.array([5.0, 5.0, 5.9])))

    def test_is_reached_outside_tolerance(self):
        """Test waypoint not reached when outside tolerance"""
        wp = Waypoint(position=np.array([5.0, 5.0, 5.0]), tolerance=1.0)

        # Outside tolerance
        self.assertFalse(wp.is_reached(np.array([6.5, 5.0, 5.0])))
        self.assertFalse(wp.is_reached(np.array([10.0, 10.0, 10.0])))

    def test_to_dict(self):
        """Test waypoint serialization"""
        wp = Waypoint(
            position=np.array([1.0, 2.0, 3.0]),
            tolerance=0.8,
            hover_time=1.5
        )

        data = wp.to_dict()

        self.assertEqual(data['position'], [1.0, 2.0, 3.0])
        self.assertEqual(data['tolerance'], 0.8)
        self.assertEqual(data['hover_time'], 1.5)


class TestRoute(unittest.TestCase):
    """Test Route class"""

    def test_initialization(self):
        """Test basic route initialization"""
        waypoints = [
            Waypoint(np.array([0, 0, 0])),
            Waypoint(np.array([1, 0, 0])),
            Waypoint(np.array([1, 1, 0]))
        ]
        route = Route(waypoints, loop=False)

        self.assertEqual(len(route.waypoints), 3)
        self.assertFalse(route.loop)

    def test_from_positions(self):
        """Test creating route from position list"""
        positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        route = Route.from_positions(positions, tolerance=0.8, loop=False)

        self.assertEqual(len(route.waypoints), 3)
        np.testing.assert_array_equal(route.waypoints[0].position, np.array([0, 0, 0]))
        self.assertEqual(route.waypoints[0].tolerance, 0.8)
        self.assertFalse(route.loop)

    def test_from_positions_with_loop(self):
        """Test looping route adds starting position at end"""
        positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        route = Route.from_positions(positions, loop=True)

        # Should have original 3 + 1 to return to start
        self.assertEqual(len(route.waypoints), 4)
        np.testing.assert_array_equal(route.waypoints[-1].position, route.waypoints[0].position)
        self.assertTrue(route.loop)

    def test_circular_route(self):
        """Test circular route generation"""
        center = np.array([5.0, 5.0, 2.0])
        radius = 3.0
        route = Route.circular_route(center, radius, height=2.0, num_points=8)

        self.assertEqual(len(route.waypoints), 9)  # 8 points + return to start
        self.assertTrue(route.loop)

        # Check all points are at correct height
        for wp in route.waypoints:
            self.assertAlmostEqual(wp.position[2], 2.0)

        # Check points are roughly at correct radius
        for i in range(8):
            dist = np.linalg.norm(route.waypoints[i].position[:2] - center[:2])
            self.assertAlmostEqual(dist, radius, places=5)

    def test_rectangular_route(self):
        """Test rectangular route generation"""
        corner1 = np.array([0, 0, 0])
        corner2 = np.array([10, 8, 0])
        route = Route.rectangular_route(corner1, corner2, height=5.0, loop=True)

        # Should have 4 corners + return to start
        self.assertEqual(len(route.waypoints), 5)
        self.assertTrue(route.loop)

        # Check all points are at correct height
        for wp in route.waypoints:
            self.assertEqual(wp.position[2], 5.0)

        # Check corner positions
        self.assertEqual(route.waypoints[0].position[0], 0)
        self.assertEqual(route.waypoints[0].position[1], 0)
        self.assertEqual(route.waypoints[1].position[0], 10)
        self.assertEqual(route.waypoints[2].position[1], 8)

    def test_get_total_distance(self):
        """Test total distance calculation"""
        # Simple straight line route
        positions = [[0, 0, 0], [3, 0, 0], [3, 4, 0]]
        route = Route.from_positions(positions)

        # Distance should be 3 + 4 = 7
        total = route.get_total_distance()
        self.assertAlmostEqual(total, 7.0)

    def test_get_total_distance_empty(self):
        """Test distance for routes with < 2 waypoints"""
        route = Route([])
        self.assertEqual(route.get_total_distance(), 0.0)

        route = Route([Waypoint(np.array([0, 0, 0]))])
        self.assertEqual(route.get_total_distance(), 0.0)

    def test_to_dict(self):
        """Test route serialization"""
        positions = [[0, 0, 0], [1, 0, 0]]
        route = Route.from_positions(positions, loop=True)

        data = route.to_dict()

        self.assertIn('waypoints', data)
        self.assertIn('loop', data)
        self.assertIn('total_distance', data)
        self.assertTrue(data['loop'])
        self.assertEqual(len(data['waypoints']), 3)  # 2 + return


if __name__ == '__main__':
    unittest.main()
