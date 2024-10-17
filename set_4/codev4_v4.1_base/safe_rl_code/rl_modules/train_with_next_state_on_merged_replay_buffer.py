# this is the main file which calls all the remaining functions and
# keeps track of simulation time

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
#from numba import jit, cuda 
import time
import copy
import numpy as np
import random
import math
from collections import deque
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import csv
import pickle	
import time
import data_file
# import vehicle
# import gen_vehi
# import prov_phase
# import coord_phase
# import functions
# import set_class
# import vehicle
import sys
from multiprocessing import Pool
import logging 
import lzma

def func(_args):
	algo_option = "rl_modified_ddswa"

	train_iter = _args[0]
	sim = _args[1]
	

	if data_file.rl_flag:

		import tensorflow as tf

		if data_file.rl_algo_opt == "DDPG":
			from DDPG_model_seq_v_two import DDPG as Agent

		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent


		
		ss = [int(data_file.buff_size*len(data_file.arr_rates_to_simulate)), 64]
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0.99

		max_learn_iter = 100100

		read_file_path = f"../data/merged_replay_buffer_with_next_state/merged_replay_buffer"

		write_trained_policy_file_path = f"../data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"

		init_weights_path = f"../data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"
		


		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
				
				# agent.actor_model_.load_weights(f"../data/init_weights/train_sim_{sim}/actor_weights_itr_0")
				# agent.critic_model_.load_weights(f"../data/init_weights/train_sim_{sim}/critic_weights_itr_0")

				# agent.actor_model_.load_weights(f"{init_weights_path}/actor_weights_itr_final")
				# agent.critic_model_.load_weights(f"{init_weights_path}/critic_weights_itr_final")

			
				read_file = lzma.open(f"{read_file_path}", 'rb')
				buffer = pickle.load(read_file)
				read_file.close()

				agent.buffer.state_buffer = buffer["state_buffer"]
				agent.buffer.action_buffer = buffer["action_buffer"]
				agent.buffer.reward_buffer = buffer["reward_buffer"]
				agent.buffer.next_state_buffer = buffer["next_state_buffer"]

				agent.buffer.buffer_counter = ss[0]

				#print("sta_buff_size",len(buffer["state_buffer"]))

				for i in range(max_learn_iter):
					agent.buffer.learn()
					#print(fmhsmhbfms)

					#agent.update_target(agent.target_model.actor_model.variables, agent.target_model.actor_model.variables, agent.tau_)
					#agent.update_target(agent.target_model.critic_model.variables, agent.target_model.critic_model.variables, agent.tau_)
					#print(f'model*****weights:{len(agent.model.variables)}, \n\n tar_weights:{len(agent.target_model.variables)}')
					agent.update_target(agent.target_model.variables, agent.model.variables, agent.tau_)
					#exit()

					#agent.update_target(agent.target_actor_.variables, agent.actor_model_.variables, agent.tau_)
					#agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)
					
					if (i % 5000) == 0:
						#logging.info('iter:{}'.format(i))

						directory = f"{write_trained_policy_file_path}/actor_weights_itr_{i}"
						os.makedirs(directory, exist_ok=True)
						filename = f"actor_weights_itr_{int(i)}.weights.h5"
						file_path = os.path.join(directory, filename)
						agent.model.save_weights(file_path)

						#agent.model.actor_model.save_weights(f"{write_trained_policy_file_path}/actor_weights_itr_final")
						#agent.model.critic_model.save_weights(f"{write_trained_policy_file_path}/critic_weights_itr_final")
						agent.model.save_weights(f"{write_trained_policy_file_path}/critic_weights_itr_final")




						#agent.actor_model_.save_weights(f"{write_trained_policy_file_path}/actor_weights_itr_{i}")

						#agent.actor_model_.save_weights(f"{write_trained_policy_file_path}/actor_weights_itr_final")
						#agent.critic_model_.save_weights(f"{write_trained_policy_file_path}/critic_weights_itr_final")

					print(f"learning iteration: {i+1} out of {max_learn_iter}", end="\r")




			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		
		elif data_file.rl_algo_opt == "MADDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
			

if __name__ == '__main__':

	#with open('train_cml.log', 'w'):	pass # empty loffer for append mode
	#logging.basicConfig(filename="train_cml.log", format='%(asctime)s %(message)s',filemode='a')
	#logger = logging.getLogger()
	#logger.setLevel(logging.DEBUG)
	args = []
	#_sim_num = int(sys.argv[1])
	#logging.info('policy:{}'.format(_sim_num))
	
	for _train_iter in range(1):
		for _sim_num in range(1, 11):
			args.append([_train_iter, _sim_num])
		#args = [_train_iter, _sim_num]
			#func(args[-1])

# 10 diff policy from CML 
	pool = Pool(10)

	pool.map(func, args)
	#func(args)
