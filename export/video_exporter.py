"""
Video export functionality with 2D and 3D rendering support
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Literal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from backend import Simulation


ViewType = Literal["2d", "3d", "side", "multi"]


class VideoExporter:
    """
    Export simulation as video file with 2D and 3D rendering options

    Supports:
    - 2D top-down view (XY plane)
    - 3D perspective view (full 3D visualization)
    - Side view (XZ plane)
    - Multi-view (all views in one frame)
    """

    def __init__(
        self,
        simulation: Simulation,
        resolution: Tuple[int, int] = (1920, 1080),
        view_type: ViewType = "3d"
    ):
        """
        Initialize video exporter

        Args:
            simulation: Simulation instance to export
            resolution: Video resolution (width, height)
            view_type: Type of view ("2d", "3d", "side", "multi")
        """
        self.simulation = simulation
        self.resolution = resolution
        self.view_type = view_type

        # Set up matplotlib figure for 3D rendering
        if view_type in ["3d", "multi"]:
            self._setup_3d_figure()

    def _setup_3d_figure(self):
        """Setup matplotlib figure for 3D rendering"""
        dpi = 100
        figsize = (self.resolution[0] / dpi, self.resolution[1] / dpi)

        if self.view_type == "multi":
            # Create figure with 4 subplots for multi-view
            self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='black')
        else:
            # Single 3D plot
            self.fig = Figure(figsize=figsize, dpi=dpi, facecolor='black')

        self.canvas = FigureCanvasAgg(self.fig)

    def export(
        self,
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v',
        camera_angle: Tuple[float, float] = (20, 45)
    ):
        """
        Export simulation as video

        Args:
            output_path: Path to output video file
            fps: Frames per second
            codec: Video codec (mp4v, avc1, etc.)
            camera_angle: (elevation, azimuth) for 3D view in degrees
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
            print(f"Rendering {self.view_type} video: {output_path}")
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
                    if self.view_type == "2d":
                        frame = self._render_2d_frame()
                    elif self.view_type == "3d":
                        frame = self._render_3d_frame(camera_angle)
                    elif self.view_type == "side":
                        frame = self._render_side_frame()
                    elif self.view_type == "multi":
                        frame = self._render_multi_frame(camera_angle)
                    else:
                        frame = self._render_2d_frame()

                    writer.write(frame)
                    frame_count += 1
                    next_frame_time += time_per_frame

                    if frame_count % 30 == 0:
                        progress = (self.simulation.current_time / self.simulation.config.duration) * 100
                        print(f"  Progress: {progress:.1f}% ({frame_count} frames)")

            print(f"Video export complete: {frame_count} frames")

        finally:
            writer.release()
            if hasattr(self, 'fig'):
                plt.close(self.fig)

    def _render_2d_frame(self) -> np.ndarray:
        """
        Render 2D top-down view (XY plane)

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

        def project_to_2d(pos: np.ndarray) -> Tuple[int, int]:
            """Top-down orthographic projection"""
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
            p1 = project_to_2d(np.array(corners[i]))
            p2 = project_to_2d(np.array(corners[(i + 1) % 4]))
            cv2.line(frame, p1, p2, (100, 100, 100), 2)

        # Draw grid
        grid_spacing = 2.0
        for x in np.arange(0, room_dims[0], grid_spacing):
            p1 = project_to_2d(np.array([x, 0, 0]))
            p2 = project_to_2d(np.array([x, room_dims[1], 0]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)
        for y in np.arange(0, room_dims[1], grid_spacing):
            p1 = project_to_2d(np.array([0, y, 0]))
            p2 = project_to_2d(np.array([room_dims[0], y, 0]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)

        # Draw trajectories
        for drone in self.simulation.drones:
            if len(drone.trajectory_history) > 1:
                points = [project_to_2d(pos) for pos in drone.trajectory_history[-100:]]
                for i in range(len(points) - 1):
                    alpha = i / len(points)
                    color = tuple(int(c * 255 * alpha) for c in drone.color)
                    cv2.line(frame, points[i], points[i + 1], color, 2)

        # Draw security spheres (transparent)
        overlay = frame.copy()
        for drone in self.simulation.drones:
            pos_2d = project_to_2d(drone.state.position)
            color = tuple(int(c * 255) for c in drone.color)
            sphere_radius = int(drone.security_sphere_size * scale)
            cv2.circle(overlay, pos_2d, sphere_radius, color, 2)
        # Blend overlay with main frame (alpha = 0.2 for transparency)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw drones
        for drone in self.simulation.drones:
            pos_2d = project_to_2d(drone.state.position)
            color = tuple(int(c * 255) for c in drone.color)
            radius = int(drone.size * scale)

            # Draw drone body
            cv2.circle(frame, pos_2d, max(radius, 5), color, -1)
            cv2.circle(frame, pos_2d, max(radius, 5), (255, 255, 255), 1)

            # Draw direction indicator
            direction = drone.state.velocity
            if np.linalg.norm(direction) > 0.1:
                direction = direction / np.linalg.norm(direction)
                end_point = project_to_2d(drone.state.position + direction * 1.0)
                cv2.arrowedLine(frame, pos_2d, end_point, (255, 255, 255), 2)

            # Draw label
            label = f"D{drone.id}"
            cv2.putText(frame, label, (pos_2d[0] + 10, pos_2d[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw info overlay
        self._draw_info_overlay(frame)

        return frame

    def _render_3d_frame(self, camera_angle: Tuple[float, float] = (20, 45)) -> np.ndarray:
        """
        Render 3D perspective view using matplotlib

        Args:
            camera_angle: (elevation, azimuth) in degrees

        Returns:
            BGR image for OpenCV
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='3d', facecolor='black')

        room_dims = self.simulation.room.dimensions

        # Set viewing angle
        ax.view_init(elev=camera_angle[0], azim=camera_angle[1])

        # Draw room wireframe
        self._draw_3d_room(ax, room_dims)

        # Draw trajectories
        for drone in self.simulation.drones:
            if len(drone.trajectory_history) > 1:
                traj = np.array(drone.trajectory_history[-100:])
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=drone.color, alpha=0.3, linewidth=1)

        # Draw security spheres (transparent wireframe)
        for drone in self.simulation.drones:
            pos = drone.state.position
            radius = drone.security_sphere_size
            self._draw_sphere_wireframe(ax, pos, radius, drone.color, alpha=0.15)

        # Draw drones
        for drone in self.simulation.drones:
            pos = drone.state.position
            ax.scatter(pos[0], pos[1], pos[2],
                      color=drone.color, s=200, marker='o',
                      edgecolors='white', linewidths=1.5)

            # Draw label
            ax.text(pos[0], pos[1], pos[2] + 0.5,
                   f'D{drone.id}', color='white', fontsize=8)

        # Set labels and limits
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        ax.set_xlim(0, room_dims[0])
        ax.set_ylim(0, room_dims[1])
        ax.set_zlim(0, room_dims[2])

        # Style the plot
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)

        # Add title with info
        title = f"{self.simulation.config.simulation_name}\nTime: {self.simulation.current_time:.2f}s"
        ax.set_title(title, color='white', pad=20)

        # Convert matplotlib figure to numpy array
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    def _render_side_frame(self) -> np.ndarray:
        """
        Render side view (XZ plane)

        Returns:
            BGR image for OpenCV
        """
        frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)

        room_dims = self.simulation.room.dimensions
        margin = 50
        scale_x = (self.resolution[0] - 2 * margin) / room_dims[0]
        scale_z = (self.resolution[1] - 2 * margin) / room_dims[2]
        scale = min(scale_x, scale_z)

        def project_to_side(pos: np.ndarray) -> Tuple[int, int]:
            """Side view projection (XZ plane)"""
            x = int(margin + pos[0] * scale)
            z = int(self.resolution[1] - margin - pos[2] * scale)  # Flip Z
            return (x, z)

        # Draw room bounds
        corners = [
            [0, 0, 0],
            [room_dims[0], 0, 0],
            [room_dims[0], 0, room_dims[2]],
            [0, 0, room_dims[2]]
        ]
        for i in range(4):
            p1 = project_to_side(np.array(corners[i]))
            p2 = project_to_side(np.array(corners[(i + 1) % 4]))
            cv2.line(frame, p1, p2, (100, 100, 100), 2)

        # Draw grid
        for x in np.arange(0, room_dims[0], 2.0):
            p1 = project_to_side(np.array([x, 0, 0]))
            p2 = project_to_side(np.array([x, 0, room_dims[2]]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)
        for z in np.arange(0, room_dims[2], 2.0):
            p1 = project_to_side(np.array([0, 0, z]))
            p2 = project_to_side(np.array([room_dims[0], 0, z]))
            cv2.line(frame, p1, p2, (50, 50, 50), 1)

        # Draw trajectories
        for drone in self.simulation.drones:
            if len(drone.trajectory_history) > 1:
                points = [project_to_side(pos) for pos in drone.trajectory_history[-100:]]
                for i in range(len(points) - 1):
                    alpha = i / len(points)
                    color = tuple(int(c * 255 * alpha) for c in drone.color)
                    cv2.line(frame, points[i], points[i + 1], color, 2)

        # Draw security spheres (transparent)
        overlay = frame.copy()
        for drone in self.simulation.drones:
            pos_2d = project_to_side(drone.state.position)
            color = tuple(int(c * 255) for c in drone.color)
            sphere_radius = int(drone.security_sphere_size * scale)
            cv2.circle(overlay, pos_2d, sphere_radius, color, 2)
        # Blend overlay with main frame (alpha = 0.2 for transparency)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw drones
        for drone in self.simulation.drones:
            pos_2d = project_to_side(drone.state.position)
            color = tuple(int(c * 255) for c in drone.color)
            radius = int(drone.size * scale)

            cv2.circle(frame, pos_2d, max(radius, 5), color, -1)
            cv2.circle(frame, pos_2d, max(radius, 5), (255, 255, 255), 1)

            label = f"D{drone.id}"
            cv2.putText(frame, label, (pos_2d[0] + 10, pos_2d[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self._draw_info_overlay(frame)
        return frame

    def _render_multi_frame(self, camera_angle: Tuple[float, float] = (20, 45)) -> np.ndarray:
        """
        Render multi-view frame with 2D, 3D, and side views

        Returns:
            BGR image for OpenCV
        """
        self.fig.clear()

        # Create 2x2 grid
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 3D view (top-left, takes 2 rows)
        ax_3d = self.fig.add_subplot(gs[:, 0], projection='3d', facecolor='black')
        self._plot_3d_view(ax_3d, camera_angle)

        # Top view (top-right)
        ax_top = self.fig.add_subplot(gs[0, 1], facecolor='black')
        self._plot_2d_view(ax_top, 'XY')

        # Side view (bottom-right)
        ax_side = self.fig.add_subplot(gs[1, 1], facecolor='black')
        self._plot_2d_view(ax_side, 'XZ')

        # Convert to image
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    def _plot_3d_view(self, ax, camera_angle: Tuple[float, float]):
        """Plot 3D view on given axes"""
        room_dims = self.simulation.room.dimensions
        ax.view_init(elev=camera_angle[0], azim=camera_angle[1])

        self._draw_3d_room(ax, room_dims)

        for drone in self.simulation.drones:
            if len(drone.trajectory_history) > 1:
                traj = np.array(drone.trajectory_history[-100:])
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=drone.color, alpha=0.3, linewidth=1)

        # Draw security spheres
        for drone in self.simulation.drones:
            pos = drone.state.position
            radius = drone.security_sphere_size
            self._draw_sphere_wireframe(ax, pos, radius, drone.color, alpha=0.3)

        # Draw drones
        for drone in self.simulation.drones:
            pos = drone.state.position
            ax.scatter(pos[0], pos[1], pos[2],
                      color=drone.color, s=100, marker='o',
                      edgecolors='white', linewidths=1)

        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('Z', color='white', fontsize=8)
        ax.set_xlim(0, room_dims[0])
        ax.set_ylim(0, room_dims[1])
        ax.set_zlim(0, room_dims[2])
        ax.tick_params(colors='white', labelsize=6)
        ax.set_title('3D View', color='white', fontsize=10)

    def _plot_2d_view(self, ax, plane: str):
        """Plot 2D view on given axes"""
        room_dims = self.simulation.room.dimensions

        if plane == 'XY':
            ax.set_xlim(0, room_dims[0])
            ax.set_ylim(0, room_dims[1])
            ax.set_xlabel('X (m)', color='white', fontsize=8)
            ax.set_ylabel('Y (m)', color='white', fontsize=8)
            ax.set_title('Top View (XY)', color='white', fontsize=10)

            # Draw room
            ax.plot([0, room_dims[0], room_dims[0], 0, 0],
                   [0, 0, room_dims[1], room_dims[1], 0],
                   'gray', linewidth=2)

            for drone in self.simulation.drones:
                if len(drone.trajectory_history) > 1:
                    traj = np.array(drone.trajectory_history[-100:])
                    ax.plot(traj[:, 0], traj[:, 1],
                           color=drone.color, alpha=0.5, linewidth=1)

            # Draw security spheres
            for drone in self.simulation.drones:
                pos = drone.state.position
                circle = plt.Circle((pos[0], pos[1]), drone.security_sphere_size,
                                   color=drone.color, fill=False, alpha=0.5, linewidth=1)
                ax.add_patch(circle)

            # Draw drones
            for drone in self.simulation.drones:
                pos = drone.state.position
                ax.scatter(pos[0], pos[1],
                          color=drone.color, s=50, marker='o',
                          edgecolors='white', linewidths=1)

        elif plane == 'XZ':
            ax.set_xlim(0, room_dims[0])
            ax.set_ylim(0, room_dims[2])
            ax.set_xlabel('X (m)', color='white', fontsize=8)
            ax.set_ylabel('Z (m)', color='white', fontsize=8)
            ax.set_title('Side View (XZ)', color='white', fontsize=10)

            # Draw room
            ax.plot([0, room_dims[0], room_dims[0], 0, 0],
                   [0, 0, room_dims[2], room_dims[2], 0],
                   'gray', linewidth=2)

            for drone in self.simulation.drones:
                if len(drone.trajectory_history) > 1:
                    traj = np.array(drone.trajectory_history[-100:])
                    ax.plot(traj[:, 0], traj[:, 2],
                           color=drone.color, alpha=0.5, linewidth=1)

            # Draw security spheres
            for drone in self.simulation.drones:
                pos = drone.state.position
                circle = plt.Circle((pos[0], pos[2]), drone.security_sphere_size,
                                   color=drone.color, fill=False, alpha=0.5, linewidth=1)
                ax.add_patch(circle)

            # Draw drones
            for drone in self.simulation.drones:
                pos = drone.state.position
                ax.scatter(pos[0], pos[2],
                          color=drone.color, s=50, marker='o',
                          edgecolors='white', linewidths=1)

        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(True, alpha=0.3, color='gray')

    def _draw_3d_room(self, ax, room_dims):
        """Draw 3D wireframe room"""
        # Define the vertices of the room
        vertices = np.array([
            [0, 0, 0], [room_dims[0], 0, 0],
            [room_dims[0], room_dims[1], 0], [0, room_dims[1], 0],
            [0, 0, room_dims[2]], [room_dims[0], 0, room_dims[2]],
            [room_dims[0], room_dims[1], room_dims[2]], [0, room_dims[1], room_dims[2]]
        ])

        # Define the edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]

        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, color='white', linewidth=1, alpha=0.5)

    def _draw_sphere_wireframe(self, ax, center: np.ndarray, radius: float, color, alpha: float = 0.15):
        """Draw a transparent wireframe sphere in 3D"""
        # Generate sphere coordinates using spherical coordinates
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot wireframe
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)

    def _draw_info_overlay(self, frame: np.ndarray):
        """Draw information overlay on frame"""
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

    def export_frames(
        self,
        output_dir: str,
        frame_interval: float = 1.0,
        camera_angle: Tuple[float, float] = (20, 45)
    ):
        """
        Export simulation as individual frame images

        Args:
            output_dir: Directory to save frames
            frame_interval: Time between frames (seconds)
            camera_angle: (elevation, azimuth) for 3D view
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {self.view_type} frames to: {output_path}")
        frame_count = 0
        next_frame_time = 0.0

        self.simulation.reset()

        while self.simulation.current_time < self.simulation.config.duration:
            self.simulation.step()

            if self.simulation.current_time >= next_frame_time:
                if self.view_type == "2d":
                    frame = self._render_2d_frame()
                elif self.view_type == "3d":
                    frame = self._render_3d_frame(camera_angle)
                elif self.view_type == "side":
                    frame = self._render_side_frame()
                elif self.view_type == "multi":
                    frame = self._render_multi_frame(camera_angle)

                filename = output_path / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(filename), frame)
                frame_count += 1
                next_frame_time += frame_interval

        print(f"Exported {frame_count} frames")

        if hasattr(self, 'fig'):
            plt.close(self.fig)
