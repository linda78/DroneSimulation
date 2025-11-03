"""
3D Visualization using VisPy
Real-time rendering of drone simulation
"""

import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Color
from typing import List, Optional
import time

from backend import Simulation


class DroneViewer:
    """
    3D viewer for drone simulation using VisPy
    """

    def __init__(self, simulation: Simulation, window_size=(1280, 720)):
        """
        Initialize the 3D viewer

        Args:
            simulation: Simulation instance to visualize
            window_size: Window dimensions (width, height)
        """
        self.simulation = simulation
        self.window_size = window_size

        # Create canvas and view
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=window_size,
            show=True,
            title=f"Drone Simulation: {simulation.config.simulation_name}"
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 60
        self.view.camera.distance = 0

        # Center camera on room
        room_center = simulation.room.dimensions / 2
        self.view.camera.center = tuple(room_center)

        # Visual elements
        self.drone_visuals = []
        self.trajectory_visuals = []
        self.room_visual = None

        # Create scene
        self._create_room()
        self._create_drones()

        # Animation timer
        self.timer = app.Timer(interval=self.simulation.config.time_step, connect=self.update, start=False)

        # Performance tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def _create_room(self):
        """Create room visualization"""
        dims = self.simulation.room.dimensions

        # Create wireframe box for room bounds
        box = scene.visuals.Box(
            width=dims[0],
            height=dims[1],
            depth=dims[2],
            color=(0.5, 0.5, 0.5, 0.2),
            edge_color='white',
            parent=self.view.scene
        )
        box.transform = scene.STTransform(translate=dims / 2)

        # Create ground plane
        ground_size = max(dims[0], dims[1])
        grid = scene.visuals.GridLines(
            scale=(ground_size / 20, ground_size / 20),
            color=(0.3, 0.3, 0.3, 0.5),
            parent=self.view.scene
        )
        grid.transform = scene.STTransform(
            translate=(dims[0] / 2, dims[1] / 2, 0),
            scale=(ground_size, ground_size, 1)
        )

        # Add axes
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = scene.STTransform(scale=(2, 2, 2))

    def _create_drones(self):
        """Create visual representations for each drone"""
        for drone in self.simulation.drones:
            # Create sphere for drone body
            drone_visual = scene.visuals.Sphere(
                radius=drone.size,
                color=drone.color + (0.8,),  # Add alpha
                parent=self.view.scene
            )
            drone_visual.transform = scene.STTransform(translate=drone.state.position)
            self.drone_visuals.append(drone_visual)

            # Create line for trajectory
            if self.simulation.config.show_trajectories:
                trajectory_line = scene.visuals.Line(
                    pos=np.array([drone.state.position]),
                    color=drone.color + (0.5,),
                    width=2,
                    parent=self.view.scene
                )
                self.trajectory_visuals.append(trajectory_line)
            else:
                self.trajectory_visuals.append(None)

            # Add text label
            text = scene.visuals.Text(
                text=f"D{drone.id}",
                color='white',
                font_size=10,
                parent=self.view.scene
            )
            text.transform = scene.STTransform(
                translate=drone.state.position + np.array([0, 0, drone.size + 0.5])
            )

    def update(self, event=None):
        """Update visualization for one frame"""
        # Step simulation
        still_running = self.simulation.step()

        if not still_running:
            self.stop()
            return

        # Update drone visuals
        for i, drone in enumerate(self.simulation.drones):
            # Update position
            self.drone_visuals[i].transform = scene.STTransform(translate=drone.state.position)

            # Update trajectory
            if self.trajectory_visuals[i] is not None:
                trajectory_length = min(
                    self.simulation.config.trajectory_length,
                    len(drone.trajectory_history)
                )
                trajectory_points = drone.trajectory_history[-trajectory_length:]
                self.trajectory_visuals[i].set_data(pos=np.array(trajectory_points))

        # Update camera if following drone
        if self.simulation.config.camera_follow is not None:
            drone_id = self.simulation.config.camera_follow
            if 0 <= drone_id < len(self.simulation.drones):
                follow_pos = self.simulation.drones[drone_id].state.position
                self.view.camera.center = tuple(follow_pos)

        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time
            self.canvas.title = (
                f"Drone Simulation: {self.simulation.config.simulation_name} "
                f"| Time: {self.simulation.current_time:.1f}s | FPS: {self.fps:.1f}"
            )

        self.canvas.update()

    def start(self):
        """Start the visualization"""
        self.timer.start()

    def stop(self):
        """Stop the visualization"""
        self.timer.stop()

    def run(self):
        """Run the visualization (blocking)"""
        self.start()
        app.run()

    def screenshot(self, filename: str):
        """Save screenshot of current view"""
        img = self.canvas.render()
        from PIL import Image
        Image.fromarray(img).save(filename)


class HeadlessRenderer:
    """
    Headless renderer for generating video without displaying window
    """

    def __init__(self, simulation: Simulation, resolution=(1920, 1080)):
        """
        Initialize headless renderer

        Args:
            simulation: Simulation to render
            resolution: Output resolution
        """
        self.simulation = simulation
        self.resolution = resolution
        self.frames = []

    def render_frame(self) -> np.ndarray:
        """
        Render current simulation state to image

        Returns:
            RGB image array
        """
        # Create temporary canvas
        canvas = scene.SceneCanvas(size=self.resolution, show=False)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 60
        view.camera.distance = 30

        room_center = self.simulation.room.dimensions / 2
        view.camera.center = tuple(room_center)

        # Create scene (simplified version)
        dims = self.simulation.room.dimensions

        # Room box
        box = scene.visuals.Box(
            width=dims[0], height=dims[1], depth=dims[2],
            color=(0.5, 0.5, 0.5, 0.2),
            edge_color='white',
            parent=view.scene
        )
        box.transform = scene.STTransform(translate=dims / 2)

        # Drones
        for drone in self.simulation.drones:
            drone_visual = scene.visuals.Sphere(
                radius=drone.size,
                color=drone.color + (0.8,),
                parent=view.scene
            )
            drone_visual.transform = scene.STTransform(translate=drone.state.position)

            # Trajectory
            if len(drone.trajectory_history) > 1:
                trajectory_line = scene.visuals.Line(
                    pos=np.array(drone.trajectory_history[-50:]),
                    color=drone.color + (0.5,),
                    width=2,
                    parent=view.scene
                )

        # Render
        img = canvas.render()
        canvas.close()

        return img

    def render_simulation(self) -> List[np.ndarray]:
        """
        Render entire simulation

        Returns:
            List of frame images
        """
        self.frames = []

        while self.simulation.step():
            frame = self.render_frame()
            self.frames.append(frame)

        return self.frames
