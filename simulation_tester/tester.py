import requests
import json
import os
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

    def run_simulation_test(self, config_name: str = "server_tester_config.yaml"):
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

            self.start_simulation()

            is_running = True
            while is_running:
                print(f"current time: {self.get_status().json().get('current_time')}")
                for drone in self.get_all_drones().json().get("drones"):
                    print(f"{drone.get('id')} -> {drone.get('position')} -- History: ({drone.get('trajectory')})")
                is_running = self.get_status().json().get('running')

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
