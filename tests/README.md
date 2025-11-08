# Drone Simulation Unit Tests

Comprehensive unit tests for the drone simulation project.

## Test Coverage

### test_drone.py
Tests for drone models and state management:
- DroneState dataclass (initialization, copying)
- Drone class (initialization, state updates, trajectory history)
- Route assignment and waypoint navigation
- Serialization to dictionary

### test_route.py
Tests for routes and waypoints:
- Waypoint class (initialization, distance checking, serialization)
- Route class (initialization, factory methods)
- Circular route generation
- Rectangular route generation
- Distance calculations

### test_environment.py
Tests for simulation environment:
- Room class (initialization, boundary checking)
- Position clamping and bounds
- Environment collision detection
- Safe position calculation
- Wall collision detection

### test_flight_model.py
Tests for drone flight physics:
- PhysicalFlightModel initialization
- State updates and movement
- Maximum speed and acceleration constraints
- 3D movement
- Collision avoidance integration
- Deceleration near targets

### test_config.py
Tests for configuration system:
- Configuration dataclasses (DroneConfig, RoomConfig, etc.)
- YAML configuration loading
- Multiple drone groups
- Custom colors and positions
- Invalid configuration handling

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run specific test file
```bash
python -m pytest tests/test_drone.py
```

### Run with coverage
```bash
python -m pytest tests/ --cov=model --cov=backend
```

### Run specific test class or method
```bash
python -m pytest tests/test_drone.py::TestDrone::test_initialization
```

### Run tests with unittest (alternative)
```bash
# All tests
python -m unittest discover tests/

# Specific test file
python -m unittest tests.test_drone

# Specific test
python -m unittest tests.test_drone.TestDrone.test_initialization
```

## Test Structure

Each test file follows this structure:
- **Setup**: Test fixtures and helper methods
- **Test cases**: Individual test methods with descriptive names
- **Assertions**: numpy.testing for array comparisons, unittest assertions for others
- **Teardown**: Cleanup (when needed)

## Adding New Tests

When adding new functionality to the simulation:

1. Create or update the appropriate test file
2. Add test cases covering:
   - Normal operation
   - Edge cases
   - Error conditions
   - Boundary values
3. Run tests to ensure they pass
4. Update this README if adding a new test file

## Dependencies

Tests require:
- unittest (standard library)
- numpy
- All simulation dependencies

Optional:
- pytest (for advanced features)
- pytest-cov (for coverage reports)
