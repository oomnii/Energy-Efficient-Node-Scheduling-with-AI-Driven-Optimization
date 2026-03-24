from dataclasses import dataclass


@dataclass
class SimConfig:
    width: float = 100.0
    height: float = 100.0

    n_nodes: int = 120
    sensing_radius: float = 15.0
    threshold_coeff: float = 0.02
    target_coverage: float = 0.99

    energy_active_cost: float = 1.0
    energy_sleep_cost: float = 0.05
    battery_budget_per_node: float = 100.0

    enable_fault_tolerance: bool = True
    failure_prob_per_round: float = 0.01
    failure_model: str = "random"
    recovery_model: str = "greedy_coverage"
    n_rounds: int = 50

    enable_ai: bool = False
    sensor_type: str = "homogeneous"
    map_mode: bool = False
    map_bounds: list[list[float]] | None = None

    export_include_compare: bool = True
    export_include_experiment: bool = True

    seed: int = 7
