"""
Video export functionality using OpenCV
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from backend import Simulation


class VideoExporter:
    """
    Export simulation as video file
    """

    def __init__(self, simulation: Simulation, resolution: Tuple[int, int] = (1920, 1080)):
        """
        Initialize video exporter

        Args:
            simulation: Simulation instance to export
            resolution: Video resolution (width, height)
        """
        self.simulation = simulation
        self.resolution = resolution

    def export(self, output_path: str, fps: int = 30, codec: str = 'mp4v'):
        """
        Export simulation as video

        Args:
            output_path: Path to output video file
            fps: Frames per second
            codec: Video codec (mp4v, x264, etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            self.resolution
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")

        try:
            # Render frames
            print(f"Rendering video: {output_path}")
            frame_count = 0
            time_per_frame = 1.0 / fps
            next_frame_time = 0.0

            # Reset simulation
            self.simulation.reset()

            while self.simulation.current_time < self.simulation.config.duration:
                # Step simulation
                self.simulation.step()

                # Check if we should render a frame
                if self.simulation.current_time >= next_frame_time:
                    frame = self._render_frame()
                    writer.write(frame)
                    frame_count += 1
                    next_frame_time += time_per_frame

                    if frame_count % 30 == 0:
                        progress = (self.simulation.current_time / self.simulation.config.duration) * 100
                        print(f"  Progress: {progress:.1f}% ({frame_count} frames)")

            print(f"Video export complete: {frame_count} frames")

        finally:
            writer.release()

    def _render_frame(self) -> np.ndarray:
        """
        Render current simulation state to image frame

        Returns:
            BGR image for OpenCV (height, width, 3)
        """
        # Create blank frame
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)  # Dark background

        # Calculate scaling for 3D to 2D projection
        room_dims = self.simulation.room.dimensions
        margin = 50
        scale_x = (self.resolution[0] - 2 * margin) / room_dims[0]
        scale_y = (self.resolution[1] - 2 * margin) / room_dims[1]
        scale = min(scale_x, scale_y)

        def project_3d_to_2d(pos: np.ndarray) -> Tuple[int, int]:
            """Simple orthographic projection (top-down view)"""
            x = int(margin + pos[0] * scale)
            y = int(margin + pos[1] * scale)
            return (x, y)

        # Draw room bounds
        corners = [
            [0, 0, 0],
            [room_dims[0], 0, 0],
            [room_dims[0], room_dims[1], 0],
            [0, room_dims[1], 0]
        ]
        for i in range(4):
            p1 = project_3d_to_2d(np.array(corners[i]))
            p2 = project_3d_to_2d(np.array(corners[(i + 1) % 4]))
            cv2.line(frame, p1, p2, (100, 100, 100), 2)

        # Draw grid
        grid_spacing = 2.0  # meters
        for x in np.arange(0, room_dims[0], grid_spacing):
            p1 = project_3d_to_2d(np.array([x, 0, 0]))
            p2 = project_3d_to_2d(np.array([x, room_dims[1], 0]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)
        for y in np.arange(0, room_dims[1], grid_spacing):
            p1 = project_3d_to_2d(np.array([0, y, 0]))
            p2 = project_3d_to_2d(np.array([room_dims[0], y, 0]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)

        # Draw trajectories
        for drone in self.simulation.drones:
            if len(drone.trajectory_history) > 1:
                points = [project_3d_to_2d(pos) for pos in drone.trajectory_history[-100:]]
                for i in range(len(points) - 1):
                    alpha = i / len(points)  # Fade out older trajectory
                    color = tuple(int(c * 255 * alpha) for c in drone.color)
                    cv2.line(frame, points[i], points[i + 1], color, 2)

        # Draw drones
        for drone in self.simulation.drones:
            pos_2d = project_3d_to_2d(drone.state.position)
            color = tuple(int(c * 255) for c in drone.color)
            radius = int(drone.size * scale)

            # Draw drone body
            cv2.circle(frame, pos_2d, max(radius, 5), color, -1)
            cv2.circle(frame, pos_2d, max(radius, 5), (255, 255, 255), 1)

            # Draw direction indicator
            direction = drone.state.velocity
            if np.linalg.norm(direction) > 0.1:
                direction = direction / np.linalg.norm(direction)
                end_point = project_3d_to_2d(drone.state.position + direction * 1.0)
                cv2.arrowedLine(frame, pos_2d, end_point, (255, 255, 255), 2)

            # Draw label
            label = f"D{drone.id}"
            cv2.putText(frame, label, (pos_2d[0] + 10, pos_2d[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw info overlay
        info_y = 30
        info_text = [
            f"Simulation: {self.simulation.config.simulation_name}",
            f"Time: {self.simulation.current_time:.2f}s / {self.simulation.config.duration:.2f}s",
            f"Drones: {len(self.simulation.drones)}",
            f"Flight Model: {self.simulation.flight_model.get_name()}",
            f"Avoidance: {self.simulation.avoidance_agent.get_name()}"
        ]

        for text in info_text:
            cv2.putText(frame, text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            info_y += 25

        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.dtype == np.uint8 else frame

    def export_frames(self, output_dir: str, frame_interval: float = 1.0):
        """
        Export simulation as individual frame images

        Args:
            output_dir: Directory to save frames
            frame_interval: Time between frames (seconds)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting frames to: {output_path}")
        frame_count = 0
        next_frame_time = 0.0

        self.simulation.reset()

        while self.simulation.current_time < self.simulation.config.duration:
            self.simulation.step()

            if self.simulation.current_time >= next_frame_time:
                frame = self._render_frame()
                filename = output_path / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(filename), frame)
                frame_count += 1
                next_frame_time += frame_interval

        print(f"Exported {frame_count} frames")
