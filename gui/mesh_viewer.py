"""
3D Visualization using VisPy with 3D meshes for drones
Based on viewer.py but uses OBJ meshes instead of spheres
"""
import numpy as np
from vispy import scene

from gui import lighten_color
from gui.viewer import DroneViewer
from gui.drone_visual import OBJMeshVisual


class MeshDroneViewer(DroneViewer):
    """
    3D viewer for drone simulation using 3D meshes instead of spheres
    """

    def __init__(self, simulation, window_size=(1280, 720),
                 mesh_path="assets/drones/spot.obj.gz",
                 texture_path="assets/drones/spot.png"):
        """
        Initialize the mesh-based 3D viewer

        Args:
            simulation: Simulation instance to visualize
            window_size: Window dimensions (width, height)
            mesh_path: Path to drone mesh file (.obj or .obj.gz)
            texture_path: Path to texture image file
        """
        self.mesh_path = mesh_path
        self.texture_path = texture_path

        # Call parent constructor (will call our overridden _create_drones)
        super().__init__(simulation, window_size)

    def _create_drones(self):
        """Create visual representations for each drone using 3D meshes"""
        for drone in self.simulation.drones:
            # Create 3D mesh for drone body
            try:
                drone_visual = OBJMeshVisual(
                    mesh_path=self.mesh_path,
                    texture_path=self.texture_path,
                    scale=drone.size * 2.0,  # Scale mesh larger for better visibility
                    color=drone.color,
                    parent=self.view.scene
                )

                # Set initial transform (position only)
                drone_visual.transform = scene.STTransform(translate=drone.state.position)

                print(f"Drone {drone.id}: Mesh visual created at position {drone.state.position}")

            except Exception as e:
                print(f"Warning: Failed to load mesh for drone {drone.id}, falling back to sphere: {e}")
                # Fallback to sphere if mesh loading fails
                drone_visual = scene.visuals.Sphere(
                    radius=drone.size,
                    parent=self.view.scene,
                    color=drone.color + (0.8,)
                )
                drone_visual.transform = scene.STTransform(translate=drone.state.position)

            self.drone_visuals.append(drone_visual)

            drone_sphere = scene.visuals.Sphere(
                radius=drone.security_sphere_size / 2,
                method='latitude',
                parent=self.view.scene,
                edge_color=lighten_color(drone.color, 0.6) + (0.2,),
                color=lighten_color(drone.color) + (0.005,)
            )
            drone_sphere.transform = scene.STTransform(translate=drone.state.position)
            # No security sphere in mesh viewer - just show the mesh
            # Add a placeholder None to keep array indices aligned
            self.drone_sphere_visuals.append(drone_sphere)

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
        """Update visualization for one frame (override to handle mesh rotations)"""
        # Step simulation
        still_running = self.simulation.step()

        if not still_running:
            self.stop()
            return

        # Update drone visuals
        for i, drone in enumerate(self.simulation.drones):
            # Update security sphere position (if it exists)
            if self.drone_sphere_visuals[i] is not None:
                self.drone_sphere_visuals[i].transform = scene.STTransform(translate=drone.state.position)

            # Update drone visual (position only)
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
        import time
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time
            self.canvas.title = (
                f"Drone Simulation (MESH): {self.simulation.config.simulation_name} "
                f"| Time: {self.simulation.current_time:.1f}s | FPS: {self.fps:.1f}"
            )

        self.canvas.update()
