import numpy as np
from concurrent.futures import ProcessPoolExecutor
from utility import config, functions

def process_batch_entry(index, state_buffer, next_state_buffer):
    phase_state = functions.phase_state_transform(state_buffer[index])
    control_state = functions.phase_obs_transform(state_buffer[index])
    nxt_phase_state = functions.phase_state_transform(next_state_buffer[index])
    nxt_control_state = functions.phase_obs_transform(next_state_buffer[index])
    return phase_state, control_state, nxt_phase_state, nxt_control_state

def parallel_process(state_buffer, next_state_buffer, batch_indices):
    num_batches = len(batch_indices)

    # Prepare empty arrays to store results
    phase_state_mod = np.zeros((num_batches, config.num_phases, (config.max_robot_lane * 2), config.state_size))
    control_state_mod = np.zeros((num_batches, config.num_veh * config.state_size))
    nxt_phase_state_mod = np.zeros((num_batches, config.num_phases, (config.max_robot_lane * 2), config.state_size))
    nxt_control_state_mod = np.zeros((num_batches, config.num_veh * config.state_size))

    # Use ProcessPoolExecutor to parallelize the processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda i: process_batch_entry(i, state_buffer, next_state_buffer), range(num_batches)))

    # Unpack the results
    for i, (phase_state, control_state, nxt_phase_state, nxt_control_state) in enumerate(results):
        phase_state_mod[i] = phase_state
        control_state_mod[i] = control_state
        nxt_phase_state_mod[i] = nxt_phase_state
        nxt_control_state_mod[i] = nxt_control_state

    return phase_state_mod, control_state_mod, nxt_phase_state_mod, nxt_control_state_mod
