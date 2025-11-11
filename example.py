#!/usr/bin/env python3
"""
Example script showing how to use the drone simulation programmatically
"""

import numpy as np
from backend import Simulation, SimulationConfig, ConfigLoader
from backend.config import DroneConfig, RoomConfig, FlightModelConfig, AvoidanceConfig
from gui import DroneViewer
from export import DataExporter, VideoExporter


def example_1_basic_simulation():
    """
    Example 1: Run a basic simulation with default settings
    """
    print("Example 1: Basic Simulation")
    print("-" * 50)

    # Load configuration from file
    config = ConfigLoader.load('configs/server_tester_config.yaml')

    # Create and run simulation
    simulation = Simulation(config)

    # Run with visualization
    viewer = DroneViewer(simulation)
    viewer.run()


def example_2_programmatic_config():
    """
    Example 2: Create simulation configuration programmatically
    """
    print("Example 2: Programmatic Configuration")
    print("-" * 50)

    # Create configuration
    config = SimulationConfig()
    config.simulation_name = "Programmatic Demo"
    config.duration = 30.0
    config.time_step = 0.05

    # Configure room
    config.room = RoomConfig(dimensions=[15.0, 15.0, 8.0])

    # Add drones
    drone_config = DroneConfig(
        count=5,
        max_speed=4.0,
        max_acceleration=2.0,
        size=0.3,
        color=[0.2, 0.8, 0.2],  # Green
        route_type="circular",
        route_params={
            'center': [7.5, 7.5, 0],
            'radius': 4.0,
            'height': 4.0,
            'num_points': 6
        }
    )
    config.drones.append(drone_config)

    # Configure flight model
    config.flight_model = FlightModelConfig(
        type="physical",
        params={'drag_coefficient': 0.1, 'approach_distance': 2.0}
    )

    # Configure avoidance
    config.avoidance = AvoidanceConfig(
        type="right",
        detection_radius=3.0,
        params={'avoidance_strength': 1.5}
    )

    # Create and run simulation
    simulation = Simulation(config)
    viewer = DroneViewer(simulation)
    viewer.run()


def example_3_headless_export():
    """
    Example 3: Run simulation headless and export data
    """
    print("Example 3: Headless Simulation with Export")
    print("-" * 50)

    # Load configuration
    config = ConfigLoader.load('configs/server_tester_config.yaml')
    config.duration = 20.0  # Shorter for demo

    # Create and run simulation
    simulation = Simulation(config)

    print("Running simulation...")
    simulation.run()

    print("Simulation complete! Exporting data...")

    # Export data
    exporter = DataExporter(simulation)
    exporter.export_state_history('output/demo_data.json')
    exporter.export_state_history('output/demo_data.csv')
    exporter.export_trajectory_plot('output/demo_trajectories.html')

    print("Export complete!")
    print("  - output/demo_data.json")
    print("  - output/demo_data.csv")
    print("  - output/demo_trajectories.html")


def example_4_custom_routes():
    """
    Example 4: Create custom routes programmatically
    """
    print("Example 4: Custom Routes")
    print("-" * 50)

    from model import Route

    # Create configuration
    config = SimulationConfig()
    config.simulation_name = "Custom Routes Demo"
    config.duration = 60.0

    # Room
    config.room = RoomConfig(dimensions=[25.0, 25.0, 10.0])

    # Drone 1: Figure-8 pattern
    waypoints_fig8 = [
        [8, 12.5, 5],
        [12.5, 15, 5],
        [17, 12.5, 5],
        [12.5, 10, 5],
    ]
    drone1 = DroneConfig(
        count=1,
        initial_positions=[[8, 12.5, 5]],
        color=[1.0, 0.2, 0.2],
        route_type="waypoints",
        route_params={'positions': waypoints_fig8, 'loop': True}
    )
    config.drones.append(drone1)

    # Drone 2: Vertical spiral
    waypoints_spiral = [
        [12.5, 12.5, 2],
        [15, 12.5, 4],
        [12.5, 15, 6],
        [10, 12.5, 8],
    ]
    drone2 = DroneConfig(
        count=1,
        initial_positions=[[12.5, 12.5, 2]],
        color=[0.2, 0.2, 1.0],
        route_type="waypoints",
        route_params={'positions': waypoints_spiral, 'loop': True}
    )
    config.drones.append(drone2)

    # Run simulation
    simulation = Simulation(config)
    viewer = DroneViewer(simulation)
    viewer.run()


def example_5_compare_avoidance():
    """
    Example 5: Compare different avoidance algorithms
    """
    print("Example 5: Compare Avoidance Algorithms")
    print("-" * 50)

    avoidance_types = ['right', 'repulsive', 'velocity_obstacle']

    for av_type in avoidance_types:
        print(f"\nTesting {av_type} avoidance...")

        config = ConfigLoader.load('configs/multi_drone.yaml')
        config.simulation_name = f"Avoidance Test - {av_type}"
        config.duration = 30.0
        config.avoidance.type = av_type
        config.export_data = True
        config.output_dir = f"output/avoidance_{av_type}"

        simulation = Simulation(config)
        simulation.run()

        # Export results
        exporter = DataExporter(simulation)
        exporter.export_analytics(config.output_dir)

        print(f"Results saved to {config.output_dir}")


def example_6_video_export():
    """
    Example 6: Export simulation as video
    """
    print("Example 6: Video Export")
    print("-" * 50)

    # Load configuration
    config = ConfigLoader.load('configs/server_tester_config.yaml')
    config.duration = 15.0  # Short video for demo

    # Create simulation
    simulation = Simulation(config)

    print("Rendering video...")
    video_exporter = VideoExporter(simulation, resolution=(1280, 720))
    video_exporter.export('output/demo_video.mp4', fps=30)

    print("Video saved to output/demo_video.mp4")


def main():
    """
    Main function - choose which example to run
    """
    examples = {
        '1': ('Basic Simulation', example_1_basic_simulation),
        '2': ('Programmatic Configuration', example_2_programmatic_config),
        '3': ('Headless Export', example_3_headless_export),
        '4': ('Custom Routes', example_4_custom_routes),
        '5': ('Compare Avoidance', example_5_compare_avoidance),
        '6': ('Video Export', example_6_video_export),
    }

    print("\n" + "=" * 60)
    print("Drone Simulation Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. Quit")

    choice = input("\nSelect example (1-6, or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        print("Goodbye!")
        return

    if choice in examples:
        _, example_func = examples[choice]
        print("\n")
        example_func()
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    main()
