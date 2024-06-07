import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool

import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

LATTICE_CONSTANT = 1.5/1000  # 1/1000, or 7/(5*1000)or 1.5/1000

# Random walk step choices
steps = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]) * LATTICE_CONSTANT
# steps = np.array([[0, 7/5], [0, -7/5], [7/5, 0], [-7/5, 0]]) / 1000


def get_analytical_parameter_corresp(D_A, D_B, lambda_A, lambda_B, alpha, lattice_constant=LATTICE_CONSTANT):
    tau_A = lattice_constant**2/(4*D_A)
    tau_B = lattice_constant**2/(4*D_B)
    sim_lambda_A = lambda_A
    sim_lambda_B = lambda_B
    sim_alpha = alpha
    return tau_A, tau_B, sim_lambda_A, sim_lambda_B, sim_alpha


def simulate_trajectory2(args):
    t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    t = 0
    position = np.array([0, 0])
    trajectory = {'t': [], 'x': [], 'y': [], 'state': []}
    state = 'A'

    while t < t_cutoff:
        trajectory['t'].append(t)
        trajectory['x'].append(position[0])
        trajectory['y'].append(position[1])
        trajectory['state'].append(state)

        if np.random.rand() < (alpha if state == 'A' else 0) * dt:
            break  # A disappears

        if state == 'A' and np.random.rand() < lambda_A * dt:
            state = 'B'
        elif state == 'B' and np.random.rand() < lambda_B * dt:
            state = 'A'

        step_size = tau_A if state == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        position += np.round(step).astype(int)
        t += dt

    return trajectory


def simulate_trajectory_notopt(args):
    t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    lattice_constant = 1.0
    num_steps = int(t_cutoff / dt) + 1
    positions = np.zeros((num_steps, 2), dtype=np.float64)
    states = np.empty(num_steps, dtype=np.str_)
    states[0] = 'A'
    t = 0

    for i in range(1, num_steps):
        if np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            break
        # if states[i - 1] == 'C' or np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
        #    states[i] = 'C'

        if states[i - 1] == 'A' and np.random.rand() < lambda_A * dt:
            states[i] = 'B'
        elif states[i - 1] == 'B' and np.random.rand() < lambda_B * dt:
            states[i] = 'A'
        else:
            states[i] = states[i - 1]

        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        # np.round(step).astype(np.int64)
        # if states[i] != 'C':
        positions[i] = positions[i - 1] + step
        # else:
        #    positions[i] = (0, 0)
        t += dt

    return {'t': np.arange(t, step=dt), 'x': positions[:, 0], 'y': positions[:, 1], 'state': states}


def simulate_trajectory_sample(args):
    t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    lattice_constant = 1.0
    t_avg = 10 ** np.linspace(-3., 3.0, 100)
    t_avg = np.sort(np.append(0.0, t_avg))
    t_avg = np.sort(t_avg)
    t_cutoff_index = int(t_cutoff // dt)
    t_avg_index = np.sort(np.unique((t_avg // dt).astype(int)))
    t_avg_num = 1  # current position in t_avg_index, zero time values are already written
    num_steps = int(t_cutoff / dt) + 1
    positions = np.zeros((t_cutoff_index+1, 2), dtype=np.float64)
    states = np.empty(t_cutoff_index+1, dtype=np.str_)
    states[0] = 'A'
    t = 0

    result = {
        "t": (t_avg_index+1) * dt,
        "x": np.zeros(shape=(t_avg_index.size)),
        "y": np.zeros(shape=(t_avg_index.size)),
        # 0-fast, 1-slow, 3 - pl
        "state": np.empty(shape=(t_avg_index.size), dtype=np.str_)
    }

    for i in range(1, t_cutoff_index+1):
        if np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            break

        if states[i - 1] == 'A' and np.random.rand() < lambda_A * dt:
            states[i] = 'B'
        elif states[i - 1] == 'B' and np.random.rand() < lambda_B * dt:
            states[i] = 'A'
        else:
            states[i] = states[i - 1]

        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        # np.round(step).astype(np.int64)
        # if states[i] != 'C':
        positions[i] = positions[i - 1] + step
        # else:
        #    positions[i] = (0, 0)
        t += dt
        if t_avg_num <= len((t_avg_index+1)) and t_avg_index[t_avg_num] == i:
            result["x"][t_avg_num] = positions[i][0]
            result["y"][t_avg_num] = positions[i][1]
            result["state"][t_avg_num] = states[i]
            t_avg_num += 1

    return result


def simulate_trajectory(args):
    t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    num_steps = int(t_cutoff / dt) + 1
    positions = np.zeros((num_steps, 2), dtype=np.float64)
    states = np.empty(num_steps, dtype=np.uint8)
    states[0] = 0
    t = 0

    for i in range(1, num_steps):
        if np.random.rand() < (alpha if states[i - 1] == 0 else 0) * dt:
            break

        if states[i - 1] == 0 and np.random.rand() < lambda_A * dt:
            states[i] = 1
        elif states[i - 1] == 1 and np.random.rand() < lambda_B * dt:
            states[i] = 0
        else:
            states[i] = states[i - 1]

        def generate_position():  # Generator for positions
            pos = np.array([0, 0])
            yield pos  # Initial position
            for state in states:
                step_size = tau_A if state == 0 else tau_B
                step = steps[np.random.randint(4)] / step_size
                pos += step
                yield pos

        return {'t': np.arange(t, step=dt),
                # Efficient extraction
                'x': np.fromiter(generate_position(), np.float64)[:, 0],
                'y': np.fromiter(generate_position(), np.float64)[:, 1],
                'state': states}


def simulate_trajectory_tavg(args):
    t_avg, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    t_avg = np.sort(np.append(0.0, t_avg))
    if t_avg.max() > t_cutoff:
        raise ValueError("Specified times should not be greater than t_cutoff")

    t_avg_index = np.sort(np.unique((t_avg // dt).astype(int)))
    t_cutoff_index = int(t_cutoff // dt) + 1
    position = np.array([0, 0])
    trajectory = {'t': t_avg_index * dt,
                  'x': np.zeros(t_avg_index.size),
                  'y': np.zeros(t_avg_index.size),
                  'state':  np.empty(t_cutoff_index, dtype=np.str_)
                  }
    state = 'A'

    t_avg_num = 0

    for t_index in range(1, t_cutoff_index + 1):

        if np.random.rand() < (alpha if state == 'A' else 0) * dt:
            break  # A disappears

        if state == 'A' and np.random.rand() < lambda_A * dt:
            state = 'B'
        elif state == 'B' and np.random.rand() < lambda_B * dt:
            state = 'A'

        step_size = tau_A if state == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        position += np.round(step).astype(int)

        while t_avg_num < len(t_avg_index) and t_avg_index[t_avg_num] == t_index:
            trajectory['x'][t_avg_num] = position[0]
            trajectory['y'][t_avg_num] = position[1]
            trajectory['state'][t_avg_num] = state
            t_avg_num += 1

    return trajectory


def simulate_trajectory_tavg3(args):
    t_avg, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    num_steps = int(t_cutoff / dt) + 1
    t_avg = np.sort(np.append(0.0, t_avg))
    if t_avg.max() > t_cutoff:
        raise ValueError("Specified times should not be greater than t_cutoff")

    t_avg_index = np.sort(np.unique((t_avg // dt).astype(int)))
    trajectory = {'t': t_avg_index * dt,
                  'x': np.zeros(t_avg_index.size),
                  'y': np.zeros(t_avg_index.size),
                  'state':  np.empty(t_avg_index.size, dtype=np.str_)
                  }
    positions = np.zeros((num_steps, 2), dtype=np.int64)
    states = np.empty(num_steps, dtype=np.str_)
    states[0] = 'A'
    t = 0
    t_avg_num = 0

    for i in range(1, num_steps):
        if np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            break

        if states[i - 1] == 'A' and np.random.rand() < lambda_A * dt:
            states[i] = 'B'
        elif states[i - 1] == 'B' and np.random.rand() < lambda_B * dt:
            states[i] = 'A'
        else:
            states[i] = states[i - 1]

        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        positions[i] = positions[i - 1] + np.round(step).astype(np.int64)
        t += dt
        while t_avg_num < len(t_avg_index) and t_avg_index[t_avg_num] == i:
            trajectory['x'][t_avg_num] = positions[t_avg_num, 0]
            trajectory['y'][t_avg_num] = positions[t_avg_num, 1]
            trajectory['state'][t_avg_num] = states[t_avg_num]
            t_avg_num += 1

    # return {'t': np.arange(t, step=dt), 'x': positions[:, 0], 'y': positions[:, 1], 'state': states}
    return trajectory


def simulate_trajectory_tavg2(args):
    t_avg, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args

    # Calculate number of steps
    num_steps = int(t_cutoff / dt) + 1

    # Sort and preprocess t_avg
    t_avg = np.sort(np.append(0.0, t_avg))
    if t_avg.max() > t_cutoff:
        raise ValueError("Specified times should not be greater than t_cutoff")
    t_avg_index = np.unique((t_avg // dt).astype(int))
    num_avg_points = len(t_avg_index)

    # Preallocate trajectory arrays
    trajectory = {
        't': t_avg_index * dt,
        'x': np.zeros(num_avg_points, dtype=np.int64),
        'y': np.zeros(num_avg_points, dtype=np.int64),
        'state': np.empty(num_avg_points, dtype=np.str_)
    }

    # Preallocate positions and states arrays
    positions = np.zeros((num_steps, 2), dtype=np.int64)
    states = np.empty(num_steps, dtype=np.str_)
    states[0] = 'A'

    # Initialize variables
    t_avg_num = 0
    t = 0

    # Main simulation loop
    for i in range(1, num_steps):
        # Check if A disappears
        if np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            break

        # Update state based on probabilities
        if states[i - 1] == 'A':
            if np.random.rand() < lambda_A * dt:
                states[i] = 'B'
            else:
                states[i] = 'A'
        else:
            if np.random.rand() < lambda_B * dt:
                states[i] = 'A'
            else:
                states[i] = 'B'

        # Determine step size and direction
        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        positions[i] = positions[i - 1] + np.round(step).astype(np.int64)
        t += dt
        # Record trajectory at specified time points
        if t_avg_num < num_avg_points and t_avg[t_avg_num] <= i*dt:
            trajectory['x'][t_avg_num] = positions[i][0]
            trajectory['y'][t_avg_num] = positions[i][1]
            trajectory['state'][t_avg_num] = states[i]
            t_avg_num += 1

    return trajectory
    # return {'t': np.arange(t, step=dt), 'x': positions[:, 0], 'y': positions[:, 1], 'state': states}


def simulate_trajectory(args):
    t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha = args
    lattice_constant = 1.0
    num_steps = int(t_cutoff / dt) + 1
    positions = np.zeros((num_steps, 2), dtype=np.float64)
    states = np.empty(num_steps, dtype=np.str_)
    states[0] = 'A'
    t = 0

    for i in range(1, num_steps):
        if np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            break

        if states[i - 1] == 'A' and np.random.rand() < lambda_A * dt:
            states[i] = 'B'
        elif states[i - 1] == 'B' and np.random.rand() < lambda_B * dt:
            states[i] = 'A'
        else:
            states[i] = states[i - 1]

        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        # np.round(step).astype(np.int64)
        positions[i] = positions[i - 1] + step
        t += dt

    return {'t': np.arange(t, step=dt), 'x': positions[:, 0], 'y': positions[:, 1], 'state': states}


def simulate_trajectory_new(args):
    start, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, trajectory_id = args
    lattice_constant = 1.0
    num_steps = int(t_cutoff / dt) + 1
    positions = np.zeros((num_steps, 2), dtype=np.float64)
    states = np.empty(num_steps, dtype=np.str_)
    states[0] = 'A'
    t = 0

    for i in range(start, num_steps):
        if states[i - 1] == 'C' or np.random.rand() < (alpha if states[i - 1] == 'A' else 0) * dt:
            states[i:] = 'C'
            positions[i:] = [0, 0]
            # t += dt
            # states[i] = 'C'
            # positions[i] = [0, 0]
            break  # Terminate the loop when 'C' state is reached

        if states[i - 1] == 'A' and np.random.rand() < lambda_A * dt:
            states[i] = 'B'
        elif states[i - 1] == 'B' and np.random.rand() < lambda_B * dt:
            states[i] = 'A'
        else:
            states[i] = states[i - 1]

        step_size = tau_A if states[i] == 'A' else tau_B
        step = steps[np.random.randint(4)] / step_size
        # if states[i] != 'C':
        positions[i] = positions[i - 1] + step
        # else:
        #    positions[i] = [0, 0]
        t += dt

    data = {'t': np.arange(t, step=dt),
            'x': positions[:, 0],
            'y': positions[:, 1],
            'state': states}

    if t < (num_steps - 1) * dt:  # Check if the final 't' was missed
        data['t'] = np.arange(0, num_steps * dt, dt)

    df = pd.DataFrame(data)
    df['trajectory'] = trajectory_id

    return df

def run_simulation_multi_parallel(t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, num_trajectories):
    args_list = [(t_cutoff, tau_A, tau_B, dt, lambda_A,
                  lambda_B, alpha)] * num_trajectories

    with Pool() as pool:
        trajectories = list(
            tqdm.tqdm(pool.imap(simulate_trajectory_sample, args_list), total=num_trajectories))

    return trajectories

def run_simulation_multi_parallel_new(start, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, num_trajectories):
    args_list = [(start, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, i)
                 for i in range(num_trajectories)]  # Add trajectory IDs

    with Pool() as pool:
        results = list(tqdm.tqdm(
            pool.imap(simulate_trajectory_new, args_list), total=num_trajectories))

    # Concatenate DataFrame results
    return pd.concat(results, ignore_index=True)


###### vectorised simulation model
@staticmethod
def _next_step_square_lattice(size=None):
    """Yields the next step along the square lattice.'"""

    choice = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])*LATTICE_CONSTANT # ((7/5)/1000) or (1/1000)

    while True:
        yield choice[np.random.choice(4, size=size), :]


def sim_traj(n_ensemble, t_cutoff, t_step, tau_A, tau_B, alpha, lambda_A, lambda_B, random_state = 42):
    np.random.seed(random_state)
    t_avg = 10 ** np.linspace(-3., 3.0, 100)

    next_step_along_lattice = _next_step_square_lattice(size=n_ensemble)
    t_avg = np.sort(np.append(0.0, t_avg))
    t_avg = np.sort(t_avg)

    if t_avg.max() > t_cutoff:
        raise ValueError("Specified times should not be greater than t_cutoff")

    t_cutoff_index = int(t_cutoff // t_step)
    t_avg_index = np.sort(np.unique((t_avg // t_step).astype(int)))
    t_avg_num = 1 #current position in t_avg_index, zero time values are already written

    result = {
        "msd": {
            "t": t_avg_index * t_step,
            "x": np.zeros(shape=(n_ensemble, t_avg_index.size)),
            "y": np.zeros(shape=(n_ensemble, t_avg_index.size)),
            "status": np.zeros(shape=(n_ensemble, t_avg_index.size), dtype=int) # 0-fast, 1-slow, 3 - pl
        }
    }

    state_n_index = np.arange(n_ensemble, dtype=int)
    state_x, state_y = np.zeros(shape=n_ensemble), np.zeros(shape=n_ensemble)  # x,y coordinates of quasi-particle
    state_status = np.zeros(shape=n_ensemble, dtype=int)  # 0-fast, 1-slow, 3 - pl

    result["msd"]["x"][:, 0] = state_x.copy()
    result["msd"]["y"][:, 0] = state_y.copy()

    for t_index in tqdm.tqdm(range(1, t_cutoff_index + 1)):
        #  photoluminescence
        mask_pl = (state_status == 0)
        mask_pl = mask_pl & (np.random.random(size=len(mask_pl)) <= alpha*t_step)
        state_status[mask_pl] = 3

        # A --> B
        mask = (state_status == 0)
        mask = mask & (np.random.random(size=len(mask)) <= lambda_A*t_step)
        state_status[mask] = 1

        # B --> A
        mask = (state_status == 1)
        mask = mask & (np.random.random(size=len(mask)) <= lambda_B*t_step)
        state_status[mask] = 0

        dxy = next_step_along_lattice.__next__()
        if alpha>0:
            mask_movement = state_status != 3
            state_x[mask_pl] = 0  # leaves the system
            state_y[mask_pl] = 0  # leaves the system
            state_x[(mask_movement) & (state_status == 0)] += dxy[:, 0][(mask_movement) & (state_status == 0)]/tau_A
            state_y[(mask_movement) & (state_status == 0)] += dxy[:, 1][(mask_movement) & (state_status == 0)]/tau_A
            state_x[(mask_movement) & (state_status == 1)] += dxy[:, 0][(mask_movement) & (state_status == 1)]/tau_B
            state_y[(mask_movement) & (state_status == 1)] += dxy[:, 1][(mask_movement) & (state_status == 1)]/tau_B
        else:
            state_x[state_status == 0] += dxy[:, 0][state_status == 0]/tau_A
            state_y[state_status == 0] += dxy[:, 1][state_status == 0]/tau_A
            state_x[state_status == 1] += dxy[:, 0][state_status == 1]/tau_B
            state_y[state_status == 1] += dxy[:, 1][state_status == 1]/tau_B


        while t_avg_num < len(t_avg_index) and t_avg_index[t_avg_num] == t_index:
            result["msd"]["x"][:, t_avg_num] = state_x.copy()
            result["msd"]["y"][:, t_avg_num] = state_y.copy()

            result["msd"]["x"][state_n_index[(state_status >= 2)], t_avg_num] = np.nan
            result["msd"]["y"][state_n_index[(state_status >= 2)], t_avg_num] = np.nan

            result["msd"]["status"][:, t_avg_num] = state_status.copy()

            t_avg_num += 1
    
    return result
