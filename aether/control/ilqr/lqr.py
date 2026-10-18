import os

from aether.control.ilqr.lqr_solver import ILQRSolverParameters, ILQRWarmStartParameters, ILQRSolver
import numpy as np


def _max_solve_time():
    """Wall-clock cap for one iLQR solve.

    The nuPlan default (0.05 s) makes the number of iterations, and therefore
    the control command, depend on CPU load: the same closed-loop episode can
    diverge between runs.  AETHER is not real-time, so the default is no cap
    (the solve still stops at convergence or ``max_ilqr_iterations``).  Set
    ``AETHER_ILQR_MAX_SOLVE_TIME=0.05`` to reproduce older runs.
    """
    value = os.environ.get("AETHER_ILQR_MAX_SOLVE_TIME", "").strip()
    return float(value) if value else None


solver_params = ILQRSolverParameters(
    discretization_time=0.5,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
    input_cost_diagonal_entries=[1.0, 10.0],
    state_trust_region_entries=[1.0] * 5,
    input_trust_region_entries=[1.0] * 2,
    max_ilqr_iterations=100,
    convergence_threshold=1e-6,
    max_solve_time=_max_solve_time(),
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3.0,
    max_steering_angle_rate=0.4,
    min_velocity_linearization=0.01,
    wheelbase=2.7
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

lqr = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)

def plan2control(plan_traj, init_state):
    current_state = init_state
    solutions = lqr.solve(current_state, plan_traj)
    optimal_inputs = solutions[-1].input_trajectory
    accel_cmd = optimal_inputs[0, 0]
    steering_rate_cmd = optimal_inputs[0, 1]
    return accel_cmd, steering_rate_cmd

if __name__ == '__main__':
    # plan_traj = np.zeros((6,5))
    # plan_traj[:, 0] = 1
    # plan_traj[:, 1] = np.ones(6)
    # plan_traj = np.cumsum(plan_traj, axis=0)
    # print(plan_traj)
    plan_traj = np.array([[-0.18724936,  2.29100776,  0.,          0.,          0.,        ],
                        [-0.29260731,  2.2971828 ,  0.,          0.,          0.        ],
                        [-0.46831554,  2.55596018,  0.,          0.,          0.        ],
                        [-0.5859955 ,  2.73183298,  0.,          0.,          0.        ],
                        [-0.62684   ,  2.84659386,  0.,          0.,          0.        ],
                        [-0.67761713,  2.80647802,  0.,          0.,          0.        ]])
    plan_traj = plan_traj[:, [1,0,2,3,4]]
    init_state = np.array([0.00000000e+00, 3.46944695e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    print(plan_traj.shape, init_state.shape)
    acc, steer = plan2control(plan_traj, init_state)
    print(acc, steer)
