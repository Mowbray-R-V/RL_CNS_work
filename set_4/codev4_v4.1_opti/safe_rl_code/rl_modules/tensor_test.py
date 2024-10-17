import tensorflow as tf
import numpy as np
import time
import os
import sys
sys.path.append(os.path.abspath('../'))
from utility import config, functions, vehicle

# Determine the number of CPU cores
num_cores = os.cpu_count()

# Define your transformation functions and wrap them with @tf.function
@tf.function
def phase_obs_transform(state):
    # Your transformation logic here
    obs = tf.zeros((config.num_veh * config.state_size,), dtype=tf.float32)  # Example output
    return obs

@tf.function
def phase_state_transform(state):
    # Your transformation logic here
    phase_state = tf.zeros((config.num_phases, config.max_robot_lane * config.state_size), dtype=tf.float32)  # Example output
    return phase_state

# Wrap the processing function with @tf.function
@tf.function
def process_batch(batch):
    phase_state = phase_state_transform(batch)
    control_state = phase_obs_transform(batch)
    return phase_state, control_state

# Parallel processing function using graph execution
@tf.function
def parallel_process(state_buffer, next_state_buffer, batch_indices):
    # Convert batch_indices to a tensor with int32 type
    batch_indices_tensor = tf.convert_to_tensor(batch_indices, dtype=tf.int32)

    # Create a TensorFlow Dataset from the state buffer using the indices tensor
    state_buffer_selected = tf.gather(state_buffer, batch_indices_tensor)
    dataset = tf.data.Dataset.from_tensor_slices(state_buffer_selected)

    # Use map to apply the transformation functions in parallel
    results = dataset.map(
        lambda x: process_batch(x),
        num_parallel_calls=num_cores  # Using number of cores
    )

    # Collect the results
    phase_state_mod, control_state_mod = zip(*results)

    return tf.stack(phase_state_mod), tf.stack(control_state_mod)

# Sample usage
state_buffer = np.random.rand(100, 10).astype(np.float32)  # Example data
next_state_buffer = np.random.rand(100, 10).astype(np.float32)  # Example data
batch_indices = np.random.choice(100, size=10).astype(np.int32)  # Random batch indices

start_time = time.time()
phase_state_mod, control_state_mod = parallel_process(state_buffer, next_state_buffer, batch_indices)
print(f"Time taken: {time.time() - start_time}")
