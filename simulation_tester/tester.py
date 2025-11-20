import requests
import json
import os
import threading
import queue
import time
from pathlib import Path

DEBUGSERVER = False

class SimulationTester:
    """Test client for the Drone Simulation API"""

    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.project_root = Path(__file__).parent.parent

    def load_simulation(self, config_name: str = "server_tester_config.yaml"):
        """Load simulation from config file"""
        config_path = self.project_root / "configs" / config_name

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"\n=== Loading simulation from {config_path} ===")
        response = requests.post(
            f"{self.base_url}/api/simulation",
            json={"config_path": str(config_path)}
        )
        if DEBUGSERVER:
            print(f"Response: {response.json()}")
        return response

    def start_simulation(self):
        """Start the simulation"""
        print("\n=== Starting simulation ===")
        response = requests.post(f"{self.base_url}/api/simulation/control/start")
        if DEBUGSERVER:
            print(f"Response: {response.json()}")
        return response

    def stop_simulation(self):
        """Stop the simulation"""
        print("\n=== Stopping simulation ===")
        response = requests.post(f"{self.base_url}/api/simulation/control/stop")
        if DEBUGSERVER:
            print(f"Response: {response.json()}")
        return response

    def reset_simulation(self):
        """Reset the simulation"""
        print("\n=== Resetting simulation ===")
        response = requests.post(f"{self.base_url}/api/simulation/control/reset")
        if DEBUGSERVER:
            print(f"Response: {response.json()}")
        return response

    def step_simulation(self):
        """Execute one simulation step"""
        if DEBUGSERVER:
            print("\n=== Stepping simulation ===")
        response = requests.post(f"{self.base_url}/api/simulation/control/step")
        if DEBUGSERVER:
            print(f"Response: {response.json()}")
        return response

    def get_status(self):
        """Get simulation status"""
        if DEBUGSERVER:
            print("\n=== Getting status ===")
        response = requests.get(f"{self.base_url}/api/status")
        status = response.json()
        if DEBUGSERVER:
            print(f"Status: {json.dumps(status, indent=2)}")
        return response

    def is_running(self):
        """Check if the simulation is running"""
        self.get_status().json().get('running')

    def get_pictures(self, picture_format: str = "jpeg", width: int = 600, height: int = 400, view_type: str = "3d"):
        """Capture current simulation state as image"""
        response = requests.get(
            f"{self.base_url}/api/capture",
            params={
                "format": picture_format,
                "width": width,
                "height": height,
                "view_type": view_type
            }
        )

        if response.status_code == 200:
            # Success - binary image data received
            if DEBUGSERVER:
                print(f"✓ Image captured: {picture_format} ({len(response.content)} bytes)")
            return response
        else:
            # Error response (JSON)
            print(f"✗ Image capture failed: {response.json()}")
            return response

    def get_all_drones(self):
        """Get information about all drones"""
        if DEBUGSERVER:
            print("\n=== Getting all drones ===")
        response = requests.get(f"{self.base_url}/api/drones")
        drones = response.json()["drones"]
        if DEBUGSERVER:
            for drone in drones:
                print(f"  Drone {drone['id']}: position={drone['position']}")
        return response

    def get_drone(self, drone_id: int):
        """Get information about a specific drone"""
        if DEBUGSERVER:
            print(f"\n=== Getting drone {drone_id} ===")
        response = requests.get(f"{self.base_url}/api/drones/{drone_id}")
        if DEBUGSERVER:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response

    def get_history(self):
        """Get simulation history"""
        print("\n=== Getting history ===")
        response = requests.get(f"{self.base_url}/api/history")
        history = response.json()
        if DEBUGSERVER:
            print(f"History length: {history['length']}")
        return response

    def check_server(self):
        """Check if the API server is running"""
        print(f"\n=== Checking if server is running at {self.base_url} ===")
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=2)
            print("✓ Server is running")
            return True
        except requests.exceptions.ConnectionError:
            print(f"✗ Server is not running at {self.base_url}")
            print(f"\nPlease start the API server first:")
            print(f"  python main.py api --port 5001")
            return False
        except Exception as e:
            print(f"✗ Error connecting to server: {e}")
            return False

    def run_basic_test(self, config_name: str = "server_tester_config.yaml"):
        """Run a basic test sequence"""
        print("="*60)
        print("DRONE SIMULATION API - BASIC TEST")
        print("="*60)

        # Check if server is running first
        if not self.check_server():
            return

        try:
            # Load simulation
            self.load_simulation(config_name)

            # Get status
            self.get_status()

            # Start simulation
            self.start_simulation()

            # Get status again
            self.get_status()

            # Get all drones
            self.get_all_drones()

            # Stop simulation
            self.stop_simulation()

            # Get history
            self.get_history()

            print("\n" + "="*60)
            print("TEST COMPLETED SUCCESSFULLY")
            print("="*60)

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise

    def _image_capture_worker(self, task_queue, result_queue, picture_format, image_dir):
        """Worker thread for capturing and saving images asynchronously"""
        while True:
            task = task_queue.get()
            if task is None:  # Poison pill to stop the worker
                break

            frame_count, current_time = task

            try:
                # Capture image
                response = self.get_pictures(picture_format)
                if response.status_code == 200:
                    # Save image with timestamp in filename
                    image_path = image_dir / f"frame_{frame_count:04d}_t{current_time:.2f}.{picture_format}"
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    result_queue.put(('success', frame_count, image_path.name))
                else:
                    result_queue.put(('error', frame_count, f"HTTP {response.status_code}"))
            except Exception as e:
                result_queue.put(('error', frame_count, str(e)))
            finally:
                task_queue.task_done()

    def run_simulation_test(self, config_name: str = "server_tester_config.yaml"):
        """Run a basic test sequence with async image capture"""
        print("="*60)
        print("DRONE SIMULATION API - BASIC TEST")
        print("="*60)

        # Check if server is running first
        if not self.check_server():
            return

        try:
            # Create output directory for images
            image_dir = self.project_root / "output" / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 Saving images to: {image_dir}")
            picture_format = "png"  # Capture PNG frames to build buffer for GIF

            # Setup async image capture
            task_queue = queue.Queue()
            result_queue = queue.Queue()
            worker = threading.Thread(
                target=self._image_capture_worker,
                args=(task_queue, result_queue, picture_format, image_dir),
                daemon=True
            )
            worker.start()

            # Load simulation
            self.load_simulation(config_name)

            self.start_simulation()

            is_running = True
            frame_count = 0
            while is_running:
                status_data = self.get_status().json()
                current_time = status_data.get('current_time')
                print(f"current time: {current_time}")

                # Queue image capture (non-blocking)
                task_queue.put((frame_count, current_time))
                frame_count += 1

                # Check for completed captures
                while not result_queue.empty():
                    status, fnum, info = result_queue.get_nowait()
                    if status == 'success' and DEBUGSERVER:
                        print(f"  💾 Saved: {info}")
                    elif status == 'error':
                        print(f"  ✗ Frame {fnum} failed: {info}")

                for drone in self.get_all_drones().json().get("drones"):
                    print(f"{drone.get('id')} -> {drone.get('position')} -- History: ({drone.get('trajectory')})")

                # Check if simulation is still running
                is_running = status_data.get('running')

                # Add delay to allow simulation to advance between captures
                # This ensures we capture different frames, not the same state repeatedly
                if is_running:
                    time.sleep(0.1)  # 100ms delay - adjust based on simulation step rate

            print("\n📊 Simulation finished, stopping image capture...")

            # Stop worker and wait for remaining tasks
            task_queue.put(None)  # Poison pill

            # Wait for worker to finish with a timeout to prevent hanging
            print("⏳ Waiting for image worker to complete...")
            worker.join(timeout=5)  # 5 second timeout
            if worker.is_alive():
                print("⚠ Worker thread didn't finish in time, but continuing...")

            # Process any remaining results
            while not result_queue.empty():
                status, fnum, info = result_queue.get_nowait()
                if status == 'success' and DEBUGSERVER:
                    print(f"  💾 Saved: {info}")

            print(f"\n✓ Simulation complete! Captured {frame_count} PNG images in {image_dir}")

            # Now create a GIF from all the buffered frames
            print("\n🎬 Creating animated GIF from captured frames...")
            try:
                gif_response = self.get_pictures(picture_format="gif")
                if gif_response.status_code == 200:
                    gif_path = image_dir / "simulation_animation.gif"
                    with open(gif_path, 'wb') as f:
                        f.write(gif_response.content)
                    print(f"✓ GIF saved: {gif_path} ({len(gif_response.content)} bytes)")
                else:
                    print(f"✗ GIF creation failed: {gif_response.json()}")
            except Exception as e:
                print(f"✗ GIF creation error: {e}")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise


def main():
    """Main entry point for the tester"""
    import argparse

    parser = argparse.ArgumentParser(description='Test the Drone Simulation API')
    parser.add_argument('--url', type=str, default='http://localhost:5001',
                       help='Base URL of the API server (default: http://localhost:5001)')
    parser.add_argument('--config', type=str, default='server_tester_config.yaml',
                       help='Config file name (default: server_tester_config.yaml)')

    args = parser.parse_args()

    tester = SimulationTester(base_url=args.url)

    # basic_test will run one time all functionality for demonstration
    # tester.run_basic_test(config_name=args.config)

    # simulation test starts a simple simulation and shows a little example how to use positions directly (video will follow)
    tester.run_simulation_test(config_name=args.config)


if __name__ == '__main__':
    main()
