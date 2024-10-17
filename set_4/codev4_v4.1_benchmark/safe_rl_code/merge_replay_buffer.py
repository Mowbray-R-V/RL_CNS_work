
import data_file

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

import numpy as np

import pickle
import lzma



arr_rate_array = data_file.arr_rates_to_simulate #[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

# read_file_path = f"./data/individual_replay_buffers"

write_file_path = f"../data/merged_replay_buffer_with_next_state/"

individual_buffer_size = data_file.buff_size #   5000

merged_buffer_size = individual_buffer_size*len(arr_rate_array)

state_size = (data_file.num_features+ data_file.num_lanes*4)*data_file.num_veh
#print("data.....","fea:",data_file.num_features, "num_veh", data_file.num_veh, "state_sx:",state_size )
#exit()

#action_size = (data_file.num_dem_param + 1)*data_file.num_veh
action_size = data_file.num_veh + data_file.num_phases 


for train_sim in range(1,2):

	merged_replay_buffer = {}

	merged_replay_buffer["state_buffer"] = np.zeros((merged_buffer_size, state_size))

	merged_replay_buffer["action_buffer"] = np.zeros((merged_buffer_size, action_size))

	merged_replay_buffer["reward_buffer"] = np.zeros((merged_buffer_size, 1))

	merged_replay_buffer["next_state_buffer"] = np.zeros((merged_buffer_size, state_size))

	for ind, arr_rate in enumerate(arr_rate_array):

		read_file = open(f"../data/arr_{arr_rate}/train_homo_stream/train_sim_{train_sim}/replay_buffer_sim_{train_sim}", 'rb')

		#print(f"arrival rate:{arr_rate}, trainsim:{train_sim}, sim:{train_sim}")
		buffer = pickle.load(read_file)

		read_file.close()

		#print("data.....",individual_buffer_size,individual_buffer_size*(ind + 1),"M_buff",np.shape(merged_replay_buffer["state_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)]), "buff", np.shape(buffer["state_buffer"]))
		#assert np.shape(buffer["state_buffer"])[1] != 560, f'index: {ind}'
		

		merged_replay_buffer["state_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["state_buffer"]

		merged_replay_buffer["action_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["action_buffer"]

		merged_replay_buffer["reward_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["reward_buffer"]

		merged_replay_buffer["next_state_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["next_state_buffer"]

		print("state_buff",len(buffer["state_buffer"]))
		print(f"train sim: {train_sim}, arrival rate: {arr_rate}")

	print("merge_buff_size",len(merged_replay_buffer["state_buffer"]))
	dbfile = lzma.open(f"{write_file_path}/merged_replay_buffer", 'wb')
	pickle.dump(merged_replay_buffer, dbfile)
	dbfile.close()





