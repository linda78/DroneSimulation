"""
Image capture utility for capturing simulation frames in various formats.
Supports PNG, JPEG, and animated GIF with configurable resolution.
"""

import io
import threading
from collections import deque
from typing import Tuple, Literal, List
import numpy as np
from PIL import Image
import cv2

from export.video_exporter import VideoExporter

FormatType = Literal["png", "jpeg", "gif"]
ViewType = Literal["2d", "3d", "side", "multi"]


class ImageCaptureManager:
    """
    Manages image capture from simulation with format conversion
    and circular buffer for GIF animation.

    This class provides thread-safe image capture functionality that
    can capture the current simulation state in various formats and
    resolutions without blocking the simulation.
    """

    def __init__(self, max_frames: int = 10):
        """
        Initialize the image capture manager.

        Args:
            max_frames: Maximum number of frames to keep in buffer for GIF animation
        """
        self.frame_buffer = deque(maxlen=max_frames)
        self.buffer_lock = threading.Lock()

    def capture_frame(
        self,
        simulation,
        format: FormatType = "png",
        width: int = 1920,
        height: int = 1080,
        view_type: ViewType = "3d",
        quality: int = 85
    ) -> io.BytesIO:
        """
        Capture current simulation state as image.

        Args:
            simulation: The simulation instance to capture
            format: Output format (png, jpeg, or gif)
            width: Image width in pixels
            height: Image height in pixels
            view_type: Type of view to render (2d, 3d, side, multi)
            quality: JPEG quality (1-100, only used for JPEG format)

        Returns:
            BytesIO object containing image data

        Raises:
            ValueError: If format is invalid or frame buffer is empty (for GIF)
        """
        # For GIF, use existing buffer
        if format == "gif":
            return self._create_gif()

        # Render frame using VideoExporter methods
        frame = self._render_frame(simulation, width, height, view_type)

        # Add to buffer (for future GIF requests)
        with self.buffer_lock:
            self.frame_buffer.append((frame, view_type, width, height))

        # Convert to requested format
        return self._convert_to_format(frame, format, quality)

    def _render_frame(
        self,
        simulation,
        width: int,
        height: int,
        view_type: ViewType
    ) -> np.ndarray:
        """
        Render a single frame using VideoExporter.

        Args:
            simulation: The simulation instance to render
            width: Image width in pixels
            height: Image height in pixels
            view_type: Type of view to render

        Returns:
            Numpy array containing the rendered frame (BGR format)
        """
        exporter = VideoExporter(
            simulation=simulation,
            resolution=(width, height),
            view_type=view_type
        )

        if view_type == "2d":
            frame = exporter._render_2d_frame()
        elif view_type == "3d":
            frame = exporter._render_3d_frame()
        elif view_type == "side":
            frame = exporter._render_side_frame()
        elif view_type == "multi":
            frame = exporter._render_multi_frame()
        else:
            raise ValueError(f"Invalid view_type: {view_type}")

        return frame

    def _convert_to_format(
        self,
        frame: np.ndarray,
        format: FormatType,
        quality: int
    ) -> io.BytesIO:
        """
        Convert OpenCV frame to requested image format.

        Args:
            frame: Numpy array containing the frame (BGR format)
            format: Output format (png or jpeg)
            quality: JPEG quality (1-100)

        Returns:
            BytesIO object containing the encoded image
        """
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        buffer = io.BytesIO()
        if format == "png":
            pil_image.save(buffer, format="PNG", optimize=True)
        elif format == "jpeg":
            pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        else:
            raise ValueError(f"Invalid format: {format}")

        buffer.seek(0)
        return buffer

    def _create_gif(self) -> io.BytesIO:
        """
        Create animated GIF from frame buffer.

        Uses the last N frames in the buffer to create an animated GIF
        that loops continuously. If buffer is empty, returns a single-frame
        GIF with a placeholder message.

        Returns:
            BytesIO object containing the GIF data
        """
        with self.buffer_lock:
            if len(self.frame_buffer) == 0:
                # Create single-frame GIF with placeholder
                # Create a simple 100x100 gray image as placeholder
                placeholder = Image.new('RGB', (100, 100), color=(128, 128, 128))
                buffer = io.BytesIO()
                placeholder.save(buffer, format="GIF")
                buffer.seek(0)
                return buffer

            # Convert all frames to PIL Images
            frames = []
            for frame_data in self.frame_buffer:
                frame = frame_data[0]  # Extract numpy array
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)

            # Create animated GIF
            buffer = io.BytesIO()
            if len(frames) == 1:
                # Single frame - no animation needed
                frames[0].save(buffer, format="GIF", optimize=True)
            else:
                # Multiple frames - create animation
                frames[0].save(
                    buffer,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,  # 200ms per frame (5 fps)
                    loop=0,  # Loop forever
                    optimize=True
                )
            buffer.seek(0)
            return buffer

    def clear_buffer(self):
        """Clear the frame buffer."""
        with self.buffer_lock:
            self.frame_buffer.clear()

    def get_buffer_size(self) -> int:
        """Get the current number of frames in the buffer."""
        with self.buffer_lock:
            return len(self.frame_buffer)
