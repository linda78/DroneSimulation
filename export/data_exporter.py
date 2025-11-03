"""
Data export functionality for simulation results
Supports CSV, JSON, and Pandas DataFrame formats
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

from backend import Simulation


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"


class DataExporter:
    """
    Export simulation data to various formats
    """

    def __init__(self, simulation: Simulation):
        """
        Initialize data exporter

        Args:
            simulation: Simulation instance with recorded data
        """
        self.simulation = simulation

    def export_state_history(self, output_path: str, format: ExportFormat = ExportFormat.JSON):
        """
        Export complete state history

        Args:
            output_path: Path to output file
            format: Export format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ExportFormat.JSON:
            self._export_json(output_path)
        elif format == ExportFormat.CSV:
            self._export_csv(output_path)
        elif format == ExportFormat.PARQUET:
            self._export_parquet(output_path)
        elif format == ExportFormat.EXCEL:
            self._export_excel(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, output_path: Path):
        """Export as JSON"""
        data = {
            'metadata': {
                'simulation_name': self.simulation.config.simulation_name,
                'duration': self.simulation.config.duration,
                'time_step': self.simulation.config.time_step,
                'num_drones': len(self.simulation.drones),
                'flight_model': self.simulation.flight_model.get_name(),
                'avoidance_agent': self.simulation.avoidance_agent.get_name()
            },
            'history': self.simulation.state_history
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, output_path: Path):
        """Export as CSV (flattened data)"""
        df = self._create_dataframe()
        df.to_csv(output_path, index=False)

    def _export_parquet(self, output_path: Path):
        """Export as Parquet (efficient binary format)"""
        df = self._create_dataframe()
        df.to_parquet(output_path, index=False)

    def _export_excel(self, output_path: Path):
        """Export as Excel with multiple sheets"""
        df = self._create_dataframe()

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Simulation Data', index=False)

            # Summary statistics sheet
            summary = self._create_summary_statistics()
            summary.to_excel(writer, sheet_name='Summary', index=False)

            # Metadata sheet
            metadata = pd.DataFrame([{
                'Simulation Name': self.simulation.config.simulation_name,
                'Duration (s)': self.simulation.config.duration,
                'Time Step (s)': self.simulation.config.time_step,
                'Number of Drones': len(self.simulation.drones),
                'Flight Model': self.simulation.flight_model.get_name(),
                'Avoidance Agent': self.simulation.avoidance_agent.get_name()
            }])
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

    def _create_dataframe(self) -> pd.DataFrame:
        """Create Pandas DataFrame from state history"""
        rows = []

        for state in self.simulation.state_history:
            time = state['time']

            for drone_data in state['drones']:
                row = {
                    'time': time,
                    'drone_id': drone_data['id'],
                    'pos_x': drone_data['position'][0],
                    'pos_y': drone_data['position'][1],
                    'pos_z': drone_data['position'][2],
                    'vel_x': drone_data['velocity'][0],
                    'vel_y': drone_data['velocity'][1],
                    'vel_z': drone_data['velocity'][2],
                    'acc_x': drone_data['acceleration'][0],
                    'acc_y': drone_data['acceleration'][1],
                    'acc_z': drone_data['acceleration'][2],
                    'orientation_roll': drone_data['orientation'][0],
                    'orientation_pitch': drone_data['orientation'][1],
                    'orientation_yaw': drone_data['orientation'][2],
                    'waypoint_index': drone_data['waypoint_index']
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics for each drone"""
        df = self._create_dataframe()

        summaries = []
        for drone_id in df['drone_id'].unique():
            drone_df = df[df['drone_id'] == drone_id]

            # Calculate statistics
            total_distance = self._calculate_distance(drone_df)
            avg_speed = np.sqrt(
                drone_df['vel_x'] ** 2 +
                drone_df['vel_y'] ** 2 +
                drone_df['vel_z'] ** 2
            ).mean()
            max_speed = np.sqrt(
                drone_df['vel_x'] ** 2 +
                drone_df['vel_y'] ** 2 +
                drone_df['vel_z'] ** 2
            ).max()

            summary = {
                'Drone ID': drone_id,
                'Total Distance (m)': total_distance,
                'Average Speed (m/s)': avg_speed,
                'Max Speed (m/s)': max_speed,
                'Flight Time (s)': drone_df['time'].max(),
                'Waypoints Reached': drone_df['waypoint_index'].max() + 1
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)

    def _calculate_distance(self, drone_df: pd.DataFrame) -> float:
        """Calculate total distance traveled by a drone"""
        positions = drone_df[['pos_x', 'pos_y', 'pos_z']].values
        if len(positions) < 2:
            return 0.0

        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.sum(distances))

    def export_trajectory_plot(self, output_path: str, drone_ids: List[int] = None):
        """
        Export 3D trajectory plot using Plotly

        Args:
            output_path: Path to output HTML file
            drone_ids: List of drone IDs to plot (None = all)
        """
        import plotly.graph_objects as go

        if drone_ids is None:
            drone_ids = [d.id for d in self.simulation.drones]

        fig = go.Figure()

        # Add trajectories for each drone
        for drone_id in drone_ids:
            drone = self.simulation.drones[drone_id]
            trajectory = np.array(drone.trajectory_history)

            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                name=f'Drone {drone_id}',
                line=dict(color=f'rgb({int(drone.color[0]*255)},{int(drone.color[1]*255)},{int(drone.color[2]*255)})', width=3),
                marker=dict(size=2)
            ))

        # Add room bounds
        dims = self.simulation.room.dimensions
        fig.add_trace(go.Mesh3d(
            x=[0, dims[0], dims[0], 0, 0, dims[0], dims[0], 0],
            y=[0, 0, dims[1], dims[1], 0, 0, dims[1], dims[1]],
            z=[0, 0, 0, 0, dims[2], dims[2], dims[2], dims[2]],
            opacity=0.1,
            color='gray',
            name='Room'
        ))

        fig.update_layout(
            title=f'Drone Trajectories - {self.simulation.config.simulation_name}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            showlegend=True
        )

        fig.write_html(output_path)

    def export_analytics(self, output_dir: str):
        """
        Export comprehensive analytics package

        Args:
            output_dir: Directory to save analytics files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export main data
        self.export_state_history(output_path / 'simulation_data.json', ExportFormat.JSON)
        self.export_state_history(output_path / 'simulation_data.csv', ExportFormat.CSV)

        # Export trajectory plot
        self.export_trajectory_plot(str(output_path / 'trajectories.html'))

        # Export summary report
        self._export_summary_report(output_path / 'summary.txt')

    def _export_summary_report(self, output_path: Path):
        """Export text summary report"""
        df = self._create_dataframe()
        summary_stats = self._create_summary_statistics()

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Drone Simulation Report\n")
            f.write(f"Simulation: {self.simulation.config.simulation_name}\n")
            f.write("=" * 60 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Duration: {self.simulation.config.duration:.2f} seconds\n")
            f.write(f"  Time Step: {self.simulation.config.time_step:.3f} seconds\n")
            f.write(f"  Number of Drones: {len(self.simulation.drones)}\n")
            f.write(f"  Flight Model: {self.simulation.flight_model.get_name()}\n")
            f.write(f"  Avoidance Agent: {self.simulation.avoidance_agent.get_name()}\n")
            f.write(f"  Room Dimensions: {self.simulation.room.dimensions}\n\n")

            f.write("Summary Statistics:\n")
            f.write(summary_stats.to_string(index=False))
            f.write("\n\n")

            f.write("=" * 60 + "\n")
