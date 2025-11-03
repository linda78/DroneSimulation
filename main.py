#!/usr/bin/env python3
"""
Drone Simulation - Main Entry Point

3D drone simulation with configurable physics, collision avoidance,
and visualization.
"""

import argparse
import sys
from pathlib import Path

from backend import ConfigLoader, Simulation
from gui import DroneViewer
from export import DataExporter, VideoExporter, ExportFormat
from api.server import run_server


def run_simulation(config_path: str, headless: bool = False, export_video: bool = False):
    """
    Run simulation with visualization

    Args:
        config_path: Path to configuration YAML file
        headless: Run without GUI (for export only)
        export_video: Export video after simulation
    """
    print(f"Loading configuration: {config_path}")
    config = ConfigLoader.load(config_path)

    print(f"Creating simulation: {config.simulation_name}")
    print(f"  Drones: {sum(d.count for d in config.drones)}")
    print(f"  Duration: {config.duration}s")
    print(f"  Flight Model: {config.flight_model.type}")
    print(f"  Avoidance: {config.avoidance.type}")

    simulation = Simulation(config)

    if not headless:
        # Run with 3D visualization
        print("\nStarting 3D visualization...")
        print("Controls:")
        print("  - Mouse: Rotate camera")
        print("  - Scroll: Zoom")
        print("  - Close window to end simulation")

        viewer = DroneViewer(simulation)
        viewer.run()
    else:
        # Run headless
        print("\nRunning simulation (headless)...")
        simulation.run()
        print("Simulation complete!")

    # Export data
    if config.export_data or export_video:
        print("\nExporting results...")
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.export_data:
            exporter = DataExporter(simulation)
            print(f"  Exporting data to {output_dir}")
            exporter.export_analytics(str(output_dir))

        if config.export_video or export_video:
            print(f"  Exporting video to {output_dir / 'simulation.mp4'}")
            video_exporter = VideoExporter(simulation)
            video_exporter.export(str(output_dir / 'simulation.mp4'), fps=config.video_fps)

    print("\nDone!")


def create_default_config(output_path: str):
    """Create a default configuration file"""
    print(f"Creating default configuration: {output_path}")
    ConfigLoader.create_default_config(output_path)
    print("Configuration file created successfully!")


def start_api_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the REST API server"""
    print(f"Starting API server on {host}:{port}")
    print(f"API documentation available at http://{host}:{port}/")
    run_server(host=host, port=port, debug=debug)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='3D Drone Simulation with Physics and Collision Avoidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default demo configuration
  python main.py run configs/simple_demo.yaml

  # Run headless and export video
  python main.py run configs/multi_drone.yaml --headless --export-video

  # Create a new default configuration
  python main.py create-config my_config.yaml

  # Start API server
  python main.py api --port 5000

  # Run stress test
  python main.py run configs/stress_test.yaml --headless
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run simulation command
    run_parser = subparsers.add_parser('run', help='Run simulation')
    run_parser.add_argument('config', type=str, help='Path to configuration YAML file')
    run_parser.add_argument('--headless', action='store_true',
                           help='Run without GUI (headless mode)')
    run_parser.add_argument('--export-video', action='store_true',
                           help='Export video after simulation')

    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create default configuration file')
    config_parser.add_argument('output', type=str, help='Output path for configuration file')

    # API server command
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0',
                           help='Host address (default: 0.0.0.0)')
    api_parser.add_argument('--port', type=int, default=5000,
                           help='Port number (default: 5000)')
    api_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')

    # List configs command
    list_parser = subparsers.add_parser('list-configs', help='List available configurations')

    args = parser.parse_args()

    # Handle commands
    if args.command == 'run':
        try:
            run_simulation(args.config, args.headless, args.export_video)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error running simulation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == 'create-config':
        try:
            create_default_config(args.output)
        except Exception as e:
            print(f"Error creating configuration: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'api':
        try:
            start_api_server(args.host, args.port, args.debug)
        except Exception as e:
            print(f"Error starting API server: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'list-configs':
        configs_dir = Path('configs')
        if configs_dir.exists():
            print("Available configurations:")
            for config_file in sorted(configs_dir.glob('*.yaml')):
                print(f"  - {config_file}")
        else:
            print("No configurations directory found")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
