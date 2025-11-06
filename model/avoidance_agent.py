"""
Collision avoidance agents for drone navigation
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .drone import Drone


class AvoidanceAgent(ABC):
    """
    Abstract base class for collision avoidance strategies
    """

    @abstractmethod
    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance vector based on nearby drones

        Args:
            drone: The drone to calculate avoidance for
            other_drones: List of other drones in the environment

        Returns:
            Avoidance vector (direction to move), or None if no avoidance needed
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this avoidance agent"""
        pass


class RightAvoidanceAgent(AvoidanceAgent):
    """
    Simple avoidance agent: always try to avoid to the right
    Uses "right-hand rule" like traffic conventions
    """

    def __init__(self, detection_radius: float = 3.0, avoidance_strength: float = 1.0):
        """
        Initialize right avoidance agent

        Args:
            detection_radius: Distance at which to start avoiding (meters)
            avoidance_strength: Strength of avoidance response (multiplier)
        """
        self.detection_radius = detection_radius
        self.avoidance_strength = avoidance_strength

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance by moving to the right of approaching drones

        Strategy:
        1. Find drones within detection radius
        2. For each nearby drone, calculate "right" direction
        3. Combine all avoidance vectors
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Calculate distance and direction to other drone
            to_other = other.state.position - drone.state.position
            distance = np.linalg.norm(to_other)

            # Only avoid if within detection radius
            if distance < self.detection_radius and distance > 0.01:
                # Calculate right direction (perpendicular to line between drones)
                # In 3D, "right" is perpendicular in XY plane
                to_other_normalized = to_other / distance

                # Calculate right vector (rotate 90 degrees in XY plane)
                right_vector = np.array([
                    -to_other_normalized[1],  # Right in XY plane
                    to_other_normalized[0],
                    0.0  # No vertical component
                ])

                # Strength inversely proportional to distance
                strength = (1.0 - distance / self.detection_radius) * self.avoidance_strength

                # Add to avoidance vector
                avoidance_vector += right_vector * strength
                num_threats += 1

        if num_threats == 0:
            return None

        # Normalize and return
        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "RightAvoidanceAgent"


class RepulsiveAvoidanceAgent(AvoidanceAgent):
    """
    Avoidance agent using repulsive forces (move directly away from threats)
    """

    def __init__(self, detection_radius: float = 3.0, repulsion_strength: float = 2.0):
        """
        Initialize repulsive avoidance agent

        Args:
            detection_radius: Distance at which to start avoiding (meters)
            repulsion_strength: Strength of repulsion force
        """
        self.detection_radius = detection_radius
        self.repulsion_strength = repulsion_strength

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance by moving away from nearby drones

        Uses inverse-square law: closer drones have stronger repulsion
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Calculate distance and direction
            to_other = other.state.position - drone.state.position
            distance = np.linalg.norm(to_other)

            # Only avoid if within detection radius
            if distance < self.detection_radius and distance > 0.01:
                # Direction away from other drone
                away_direction = -to_other / distance

                # Inverse square law for repulsion strength
                strength = self.repulsion_strength * (self.detection_radius / distance) ** 2

                avoidance_vector += away_direction * strength
                num_threats += 1

        if num_threats == 0:
            return None

        # Normalize and return
        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "RepulsiveAvoidanceAgent"


class VelocityObstacleAvoidanceAgent(AvoidanceAgent):
    """
    Advanced avoidance using velocity obstacle concept
    Considers both position and velocity of other drones
    """

    def __init__(self, detection_radius: float = 3.0, time_horizon: float = 2.0):
        """
        Initialize velocity obstacle avoidance

        Args:
            detection_radius: Distance at which to consider drones
            time_horizon: Time to predict collisions (seconds)
        """
        self.detection_radius = detection_radius
        self.time_horizon = time_horizon

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate avoidance considering future collisions based on current velocities
        """
        avoidance_vector = np.zeros(3)
        num_threats = 0

        for other in other_drones:
            if other.id == drone.id:
                continue

            # Current separation
            relative_position = other.state.position - drone.state.position
            distance = np.linalg.norm(relative_position)

            if distance > self.detection_radius:
                continue

            # Relative velocity
            relative_velocity = other.state.velocity - drone.state.velocity

            # Time to closest approach
            if distance > 0.01:
                rel_pos_norm = relative_position / distance
                closing_speed = np.dot(relative_velocity, rel_pos_norm)

                # If moving towards each other
                if closing_speed > 0:
                    time_to_collision = distance / (closing_speed + 1e-6)

                    # If collision likely within time horizon
                    if time_to_collision < self.time_horizon:
                        # Avoid perpendicular to relative velocity
                        if np.linalg.norm(relative_velocity) > 0.01:
                            # Move perpendicular to collision course
                            perp_vector = np.cross(relative_velocity, [0, 0, 1])
                            if np.linalg.norm(perp_vector) > 0.01:
                                perp_vector = perp_vector / np.linalg.norm(perp_vector)
                                strength = (self.time_horizon - time_to_collision) / self.time_horizon
                                avoidance_vector += perp_vector * strength
                                num_threats += 1

        if num_threats == 0:
            return None

        magnitude = np.linalg.norm(avoidance_vector)
        if magnitude < 0.01:
            return None

        return avoidance_vector / magnitude

    def get_name(self) -> str:
        return "VelocityObstacleAvoidanceAgent"


class MPCAvoidanceAgent(AvoidanceAgent):
    """
    Model Predictive Control (MPC) based collision avoidance

    Adapted from collisionAvoidMPC project. This agent computes optimal acceleration
    that minimizes tracking error while avoiding collisions.

    IMPORTANT: This agent includes tracking cost, so MPCFlightModel should NOT blend
    with target - just apply the returned acceleration directly.
    """

    def __init__(
        self,
        detection_radius: float = 6.0,
        prediction_horizon: int = 10,
        dt: float = 0.05,
        Q_weight: float = 1.0,
        R_weight: float = 0.1,
        room_dimensions: Optional[np.ndarray] = None,
        debug: bool = False
    ):
        """
        Initialize MPC avoidance agent

        Args:
            detection_radius: Distance at which to activate MPC
            prediction_horizon: Number of steps to predict ahead
            dt: Time step for dynamics
            Q_weight: Weight for tracking cost
            R_weight: Weight for control effort
            room_dimensions: Room bounds [x, y, z] for boundary penalties
            debug: Enable debug logging
        """
        self.detection_radius = detection_radius
        self.N = prediction_horizon
        self.dt = dt
        self.Q = Q_weight * np.eye(3)
        self.R = R_weight * np.eye(3)
        self.room_dimensions = room_dimensions
        self.debug = debug

        # Linearized drone dynamics (same as collisionAvoidMPC)
        # State: [x, y, z, vx, vy, vz]
        # Control: [ax, ay, az]
        self.A = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])

        self.B = np.block([
            [0.5 * dt**2 * np.eye(3)],
            [dt * np.eye(3)]
        ]).reshape(6, 3)

    def calculate_avoidance(self, drone: Drone, other_drones: List[Drone]) -> Optional[np.ndarray]:
        """
        Calculate MPC-based optimal acceleration

        Returns:
            Acceleration vector [ax, ay, az] or None if no threats
        """
        # Check if any drones are within detection radius
        threats = []
        min_dist = float('inf')
        for other in other_drones:
            if other.id == drone.id:
                continue

            dist = np.linalg.norm(other.state.position - drone.state.position)
            min_dist = min(min_dist, dist)

            if dist < self.detection_radius:
                threats.append(other)
                if self.debug:
                    print(f"[MPC] Drone {drone.id}: Threat detected - Drone {other.id} at distance {dist:.3f}m")

        if not threats:
            if self.debug and min_dist < 10.0:  # Only log when reasonably close
                print(f"[MPC] Drone {drone.id}: No threats detected (closest: {min_dist:.3f}m, detection radius: {self.detection_radius}m)")
            return None

        # Get target position
        target = drone.get_current_target()
        if target is None:
            return None

        # Solve MPC optimization
        try:
            acceleration = self._solve_mpc(drone, target, threats)
            if self.debug:
                print(f"[MPC] Drone {drone.id}: Optimal acceleration = {acceleration}, magnitude = {np.linalg.norm(acceleration):.3f}")
            return acceleration if np.linalg.norm(acceleration) > 0.01 else None
        except Exception as e:
            if self.debug:
                print(f"[MPC] Drone {drone.id}: Optimization failed: {e}")
            return None

    def _solve_mpc(self, drone: Drone, target: np.ndarray, threats: List[Drone]) -> np.ndarray:
        """
        Solve MPC optimization problem

        Adapted from collisionAvoidMPC/mpc.py mpc_control() function
        """
        from scipy.optimize import minimize

        def cost_function(u_flat: np.ndarray) -> float:
            """
            MPC cost function (adapted from collisionAvoidMPC)

            Includes:
            - Tracking cost: follow target waypoint
            - Control effort: penalize large accelerations
            - Distance cost: penalize proximity to other drones
            - Room boundary penalty: soft constraint for room bounds
            """
            u = np.clip(u_flat.reshape((self.N, 3)), -2.5, 2.5)

            total_cost = 0.0
            x_pred = np.hstack([drone.state.position, drone.state.velocity])

            for k in range(self.N):
                # Predict next state
                x_pred = self.A @ x_pred + self.B @ u[k]
                p_pred = x_pred[:3]  # Extract position

                # 1. Tracking cost: penalize deviation from target
                tracking_error = p_pred - target
                tracking_cost = tracking_error.T @ self.Q @ tracking_error

                # 2. Control effort cost
                control_cost = u[k].T @ self.R @ u[k]

                # 3. Distance cost and lateral evasion incentive
                # Use CURRENT threat positions (matching collisionAvoidMPC)
                distance_cost = 0.0
                lateral_evasion_reward = 0.0

                for threat in threats:
                    threat_pos = threat.state.position
                    to_threat = threat_pos - p_pred
                    dist = np.linalg.norm(to_threat)
                    safety_dist = (drone.size + 1.1 * threat.size)

                    # 3a. Distance penalty
                    if dist < safety_dist:
                        # Very strong penalty for being inside safety zone
                        distance_cost += 10000.0 * (safety_dist - dist) ** 2
                    else:
                        # Moderate inverse distance penalty outside safety zone
                        distance_cost += 1000.0 / max(dist, 0.5)

                    # 3b. Lateral evasion reward (when close, reward perpendicular movement)
                    if dist < safety_dist * 2.0 and dist > 0.01:
                        # Calculate direction to threat
                        to_threat_norm = to_threat / dist

                        # Get predicted velocity at this step
                        v_pred = x_pred[3:]

                        # Decompose velocity into parallel and perpendicular components
                        v_parallel = np.dot(v_pred, to_threat_norm) * to_threat_norm
                        v_perpendicular = v_pred - v_parallel

                        # Reward perpendicular movement (lateral evasion)
                        # Penalize parallel movement (moving toward/away on same line)
                        lateral_speed = np.linalg.norm(v_perpendicular)
                        parallel_speed = abs(np.dot(v_pred, to_threat_norm))

                        # Strong reward for lateral movement when close
                        strength = (safety_dist * 2.0 - dist) / (safety_dist * 2.0)
                        lateral_evasion_reward -= 100.0 * strength * lateral_speed  # Negative = reward
                        distance_cost += 50.0 * strength * parallel_speed  # Penalize staying on collision line

                # 4. Room boundary penalty (soft constraint)
                room_penalty = 0.0
                if self.room_dimensions is not None:
                    margin = 0.5  # Keep away from walls
                    for dim in range(3):
                        # Lower boundary
                        if p_pred[dim] < margin:
                            room_penalty += 100.0 * (margin - p_pred[dim])**2
                        # Upper boundary
                        if p_pred[dim] > self.room_dimensions[dim] - margin:
                            room_penalty += 100.0 * (p_pred[dim] - (self.room_dimensions[dim] - margin))**2

                total_cost += tracking_cost + control_cost + distance_cost + lateral_evasion_reward + room_penalty

            return float(total_cost)

        def distance_constraint(u_flat: np.ndarray) -> np.ndarray:
            """
            Inequality constraint: distance to threats >= safety distance
            (Adapted from collisionAvoidMPC/mpc.py)

            NOTE: Uses CURRENT positions of threats, not predicted positions.
            This is simpler and matches the original collisionAvoidMPC implementation.
            """
            u = np.clip(u_flat.reshape((self.N, 3)), -2.5, 2.5)
            x_pred = np.hstack([drone.state.position, drone.state.velocity])

            constraints = []
            for k in range(self.N):
                x_pred = self.A @ x_pred + self.B @ u[k]
                p_pred = x_pred[:3]

                for threat in threats:
                    # Use CURRENT threat position (like collisionAvoidMPC)
                    threat_pos = threat.state.position
                    dist = np.linalg.norm(p_pred - threat_pos)
                    safety_dist = (drone.size + 1.1 * threat.size)
                    # Constraint: dist - safety_dist >= 0
                    constraints.append(dist - safety_dist)

            return np.array(constraints)

        # Initial guess: accelerate toward target
        displacement = target - drone.state.position
        avg_velocity = displacement / self.N
        constrained_velocity = np.clip(avg_velocity, -2.5, 2.5)
        u0 = np.tile(constrained_velocity, (self.N, 1))

        # Bounds on acceleration
        bounds = [(-2.5, 2.5) for _ in range(self.N * 3)]

        # Constraints - DISABLED for now as they make head-on collisions infeasible
        # Try soft cost only approach first
        constraints = []  # [{'type': 'ineq', 'fun': distance_constraint}]

        # Optimize
        result = minimize(
            cost_function,
            u0.flatten(),
            bounds=bounds,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if self.debug:
            # Check optimization result
            print(f"[MPC] Optimization success: {result.success}, status: {result.message}")
            print(f"[MPC] Final cost: {result.fun:.3f}")

            # Check constraint violations
            constraint_values = distance_constraint(result.x)
            min_constraint = np.min(constraint_values)
            print(f"[MPC] Min constraint value: {min_constraint:.3f} (should be >= 0)")
            if min_constraint < 0:
                print(f"[MPC] WARNING: Constraint violated! Minimum predicted distance below safety threshold")

        # Extract first control action
        optimal_acceleration = result.x[:3].reshape(3,)

        return optimal_acceleration

    def get_name(self) -> str:
        return "MPCAvoidanceAgent"
