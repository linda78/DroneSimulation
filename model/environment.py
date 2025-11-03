"""
Environment and room models for the simulation
"""

import numpy as np
from typing import Optional, Tuple
from PIL import Image
import cv2


class Room:
    """
    3D room/environment where drones fly
    Can be a simple box or textured with images/video
    """

    def __init__(
        self,
        dimensions: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        texture_path: Optional[str] = None,
        is_video: bool = False
    ):
        """
        Initialize a room

        Args:
            dimensions: (width, depth, height) in meters
            texture_path: Path to image or video for walls/floor
            is_video: Whether texture is a video file
        """
        self.dimensions = np.array(dimensions, dtype=float)
        self.texture_path = texture_path
        self.is_video = is_video

        # Video capture for dynamic backgrounds
        self.video_capture = None
        self.current_frame = None

        if texture_path and is_video:
            self._load_video()
        elif texture_path:
            self._load_image()

    def _load_image(self):
        """Load static image texture"""
        try:
            img = Image.open(self.texture_path)
            self.current_frame = np.array(img)
        except Exception as e:
            print(f"Warning: Could not load image {self.texture_path}: {e}")
            self.current_frame = None

    def _load_video(self):
        """Initialize video capture"""
        try:
            self.video_capture = cv2.VideoCapture(self.texture_path)
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not load video {self.texture_path}: {e}")
            self.video_capture = None

    def update_video_frame(self):
        """Get next frame from video"""
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Loop video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def is_inside(self, position: np.ndarray) -> bool:
        """Check if a position is inside the room"""
        return np.all(position >= 0) and np.all(position <= self.dimensions)

    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        """Clamp a position to be within room bounds"""
        return np.clip(position, 0, self.dimensions)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get min and max bounds of the room"""
        return np.zeros(3), self.dimensions.copy()

    def cleanup(self):
        """Release video resources"""
        if self.video_capture:
            self.video_capture.release()


class Environment:
    """
    Complete simulation environment containing room and obstacles
    """

    def __init__(self, room: Room):
        """
        Initialize environment

        Args:
            room: The room/space for simulation
        """
        self.room = room
        self.obstacles = []  # Future: add obstacle support

    def check_collision(self, position: np.ndarray, size: float) -> bool:
        """
        Check if a drone at given position collides with environment

        Args:
            position: Center position of drone
            size: Radius of drone

        Returns:
            True if collision detected
        """
        # Check wall collisions
        min_bound, max_bound = self.room.get_bounds()
        if np.any(position - size < min_bound) or np.any(position + size > max_bound):
            return True

        # Future: Check obstacle collisions
        return False

    def get_safe_position(self, position: np.ndarray, size: float) -> np.ndarray:
        """
        Get nearest safe position (not colliding)

        Args:
            position: Desired position
            size: Drone radius

        Returns:
            Safe position within bounds
        """
        min_bound, max_bound = self.room.get_bounds()
        safe_position = np.clip(position, min_bound + size, max_bound - size)
        return safe_position
