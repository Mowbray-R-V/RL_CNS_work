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
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import csv
import pickle
import time
import data_file
import vehicle
import gen_vehi
import prov_phase
import coord_phase
import functions
import set_class
import vehicle

import sys

from multiprocessing import Pool

def func(_args):

	train_iter = _args[0]
	sim = _args[1]
	train_sim = _args[2]
	arr_rate_array_ = _args[3]
	arr_rate_ = _args[4]

	### algorithm option if data_file.rl_flag is 0 ###

	###### available algorithm options ######

			# algo_option = "comb_opt" -> for combined optimization
			# algo_option = "ddswa" -> for ddswa

	###### available algorithm options ######

	algo_option = "comb_opt"

	### algorithm option if data_file.rl_flag is 0 ###



	### flag to switch between real-time vehicle generation and generating all vehicles before simulation ###

	real_time_spawning_flag = 1

	### flag to switch between real-time vehicle generation and generating all vehicles before simulation ###


	capture_snapshot_flag = 0

	time_to_capture = 100

	#captured_snapshots = []


	### flag to run combined optimization for testing RL performance ###

	comb_test_probability = 0

	baseline_test_flag = 0

	baseline_test_freq = 10

	### flag to run combined optimization for testing RL performance ###





	if data_file.rl_flag:

		import tensorflow as tf

		if data_file.rl_algo_opt == "DDPG":
			from ddpg_related_class import DDPG as Agent

		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent


		#### initializing variables to store values ####

		rl_ret_collection = []
		comb_opt_test = []
		moving_avg_ret = []
		ddswa_comp_ret = []
		comb_opt_ret_collection = []
		rl_explore_data = []

		#### initializing variables to store values ####


		
		### algorithm option if data_file.rl_flag is 1 ###

		###### available algorithm options ######

				# algo_option = "rl_ddswa" -> for rl based ddswa (not implemented)
				# algo_option = "rl_modified_ddswa" -> for rl based modified ddswa

		###### available algorithm options ######

		algo_option = "rl_modified_ddswa"

		### algorithm option if data_file.rl_flag is 1 ###

		learning_flag = 0



		### flag to switch between stream learning and snapshot learning ####

		stream_rl_flag = 0
		
		### flag to switch between stream learning and snapshot learning ####

		#sim = 1
		ss = [5000, 64]
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0
		d_factor = 0
		


		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
			
			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		
		elif data_file.rl_algo_opt == "MADDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
			

		#### RL agent object creation ####
		#if train_iter == 0:
		#	agent.actor_model_.load_weights(f"./data/multi_snap_train_rand_500/init_weights/sim_{sim}/actor_itr_0")

		#else:
		agent.actor_model_.load_weights(f"../data/merged_replay_buffer_with_next_state/train_sim_{train_sim}/trained_weights/actor_weights_itr_{train_iter}")


	time_track = 0  # time tracking variable

	cumulative_throuput_count = 0  # cumulative throuput

	throughput_id_in_lane = [0 for _ in data_file.lanes] # variable to help track last vehicle in each lane

	sim_obj = set_class.sets() # a set of sets to classify vehicles whether they are unspawned, in provisional phase, in coordinated phase or have crossed the region of interest

	if not real_time_spawning_flag:
		file = open(f"../data/compare_files/homogeneous_traffic/arr_{arr_rate_}/sim_obj_num_{sim}", 'rb')
		sim_obj = pickle.load(file)
		file.close()
		total_veh_in_simulation = functions.get_num_of_objects(sim_obj.unspawned_veh)

	if real_time_spawning_flag:
		veh_id_var = 0
		next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]
		for lane in data_file.lanes:
			if not (arr_rate_array_[lane] == 0):
				next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array_[lane],1)),1)




	while (time_track < data_file.max_sim_time):

		curr_time = time.time()

		if (not real_time_spawning_flag):
			if (functions.get_num_of_objects(sim_obj.done_veh) >= total_veh_in_simulation):
				break

		### provisional phase and preparation for coordinated phase ###

		for lane in data_file.lanes:

			if (real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (data_file.no_veh_per_lane[lane] > 0) and (not (arr_rate_array_[lane] == 0)):
				
				next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array_[lane],1)),1)

				new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L)
				#new_veh.lane = copy.deepcopy(lane)
				#new_veh.incomp = data_file.incompdict[new_veh.lane]
				#new_veh.intsize = data_file.intersection_path_length[new_veh.lane%3]
				#new_veh.arr = arr_rate_array_[lane]
				new_veh.id = copy.deepcopy(veh_id_var)
				new_veh.sp_t = copy.deepcopy(time_track)
				# new_veh.priority = copy.deepcopy(data_file.lane_priorities[new_veh.lane])
				veh_id_var += 1


				sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))


			n = len(sim_obj.unspawned_veh[lane])
			v_itr = 0
			while v_itr < n:
				v = sim_obj.unspawned_veh[lane][v_itr]

				pre_v = None
				if len(sim_obj.prov_veh[lane]) > 0:
					pre_v = sim_obj.prov_veh[lane][-1]

				elif len(sim_obj.coord_veh[lane]) > 0:
					pre_v = sim_obj.coord_veh[lane][-1]

				if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)):

					# if (round((time_track % data_file.t_opti),1) == 0):
					# 	v.sp_t = round(time_track,1)
					# 	v.t_ser.append(round(v.sp_t,1))
					# 	v.p_traj.append(round(v.p0,1))
					# 	v.v_traj.append(v.v0)
					# 	sim_obj.prov_veh[lane].append(copy.deepcopy(v))
					# 	sim_obj.unspawned_veh[lane].popleft()
					# 	n = len(sim_obj.unspawned_veh[lane])
					# 	break

					# else:
					v.sp_t = round(time_track,1)

					prov_sucess = False
					prov_anomaly = -1

					# while prov_sucess == False:
					prov_anomaly += 1
					prov_veh = copy.deepcopy(v)
					prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track)
					# with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/anamoly.csv", "a", newline="") as f:
					# 	writer = csv.writer(f)
					# 	writer.writerows([[prov_veh.id, prov_anomaly, prov_veh.p_traj[-1], prov_veh.v_traj[-1]]])

					sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))
					sim_obj.unspawned_veh[lane].popleft()
					n = len(sim_obj.unspawned_veh[lane])
					break

				else:
					next_spawn_time[lane] = round(next_spawn_time[lane] + data_file.dt, 1)
					break

		### provisional phase and preparation for coordinated phase done ###


		### coordinated phase starting ###
		if (round((time_track % data_file.t_opti),1) == 0):

			comb_test_flag = 0

			if np.random.uniform(0,1) < comb_test_probability:
				comb_test_flag = 1

			prov_time = time.time() - curr_time

			if (functions.get_num_of_objects(sim_obj.prov_veh) > 0) and (capture_snapshot_flag == 1) and (round(time_track) >= time_to_capture - 3*data_file.t_opti) and (round(time_track) <= time_to_capture + 3*data_file.t_opti):

				#captured_snapshots.append(copy.deepcopy(sim_obj))
				m = {}
				m[0] = copy.deepcopy(sim_obj)
				m[1] = time_track
				m[2] = arr_rate_
				dbfile = open(f'../data/captured_snaps/arr_rate{arr_rate_}_time_{time_track}', 'wb')
				pickle.dump(m, dbfile)
				dbfile.close()

			success = False


			if data_file.rl_flag:
				num_of_veh = functions.get_num_of_objects(sim_obj.prov_veh)

				### making copies to run different algorithms

				prov_veh_copy = copy.deepcopy(sim_obj.prov_veh)
				coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)

				prov_veh_copy__copy = copy.deepcopy(prov_veh_copy)
				coord_veh_copy__copy = copy.deepcopy(coord_veh_copy)

				if comb_test_flag:

					prov_set_copy__comb_opt = copy.deepcopy(prov_veh_copy)
					coord_veh_copy__comb_opt = copy.deepcopy(coord_veh_copy)

				### making copies to run different algorithms

				len_lane_prov_set = [0 for _ in data_file.lanes]
				len_lane_coord_set = [0 for _ in data_file.lanes]

				for _l in data_file.lanes:
					len_lane_prov_set[_l] = copy.deepcopy(len(prov_veh_copy__copy[_l]))
					len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy__copy[_l]))


				coord_init_time = time.time()
				sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, state_t, action_t, success = coord_phase.coord_algo(time_track, prov_veh_copy__copy, coord_veh_copy__copy, algo_option, agent, learning_flag, 1, sim, train_iter, train_sim)
				coord_time = time.time() - coord_init_time

				if comb_test_flag:
					comb_test_start = time.time()
					_, comb_cost_test, _, _, _, _, _ = coord_phase.coord_algo(time_track, prov_set_copy__comb_opt, coord_veh_copy__comb_opt, "comb_opt", None, None, 1)
					comb_test_dura = time.time() - comb_test_start

				if (num_of_veh > 0):
					rl_ret = coord_cost_with_comb_opt_para / num_of_veh
					rl_ret_collection.append(rl_ret)
					if comb_test_flag:
						comb_opt_test.append(comb_cost_test / num_of_veh)
					else:
						comb_opt_test.append(0)

					if data_file.rl_algo_opt == "DDPG":
					
						with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/rl_action_record.csv", "a", newline="") as f:
							writer = csv.writer(f)
							action_record =[]
							for act in action_t:
							    #action_record.append(act[0])
							    action_record.append(act)
							action_record.insert(0, num_of_veh)
							writer.writerows([action_record])

					elif data_file.rl_algo_opt == "MADDPG":
						with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/rl_action_record.csv", "a", newline="") as f:
							writer = csv.writer(f)
							action_record =[]
							for act in action_t:
							    action_record.append(act)
							action_record.insert(0, num_of_veh)
							writer.writerows([action_record])


				else:
					rl_ret = None


				if (algo_option == "rl_modified_ddswa") and (learning_flag):
					if num_of_veh > 0:
						agent.buffer.remember((state_t, action_t, rl_ret, state_t))


					if data_file.rl_algo_opt == "DDPG":

						if (agent.buffer.buffer_counter > 0):

							train_init_time = time.time()
							agent.buffer.learn()
							train_time = time.time() - train_init_time

						'''current_weights_set = agent._actor._model_weights

						weights_for_pis = []
						weights_for_dis = []
						pi_bias = []
						di_bias = []

						for i, weight in enumerate(current_weights_set):
							temp_var = tf.keras.backend.eval(weight)
							if i == 0:
								weights_for_pis.append(temp_var.reshape((len(temp_var))))
								
							if i == 1:
								weights_for_pis = np.asarray(weights_for_pis).reshape((data_file.num_features))
								weights_for_pis = np.append(weights_for_pis, temp_var.reshape((len(temp_var))))
								with open(f"../test_inhomo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/ddswa_weights_for_pi.csv", "a", newline="") as f:
								    writer = csv.writer(f)
								    writer.writerows([weights_for_pis])

							if i == 2:
								weights_for_dis.append(temp_var.reshape((len(temp_var))))

							if i == 3:
								weights_for_dis = np.asarray(weights_for_dis).reshape((data_file.num_features))
								weights_for_dis = np.append(weights_for_dis, temp_var.reshape((len(temp_var))))
								with open(f"../test_inhomo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/ddswa_weights_for_di.csv", "a", newline="") as f:
								    writer = csv.writer(f)
								    writer.writerows([weights_for_dis])'''

					elif data_file.rl_algo_opt == "MADDPG":
						train_init_time = time.time()
						agent.train(num_of_veh+1)
						train_time = time.time() - train_init_time

				if num_of_veh > 0:
					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/coord_phase_info.csv", "a", newline="") as f:
						writer = csv.writer(f)
						curr_avg = 0
						'''for _ln in data_file.lanes:
							num_veh_prov_in_lane = len_lane_prov_set[_ln]
							temp_coord_veh = sim_obj.coord_veh[_ln]
							list_to_iterate = list(copy.deepcopy(temp_coord_veh))[-num_veh_prov_in_lane:]
							curr_avg += sum([data_file.W_pos*_veh_.p_traj[min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1)] for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])
							curr_avg -= sum([sum([data_file.W_acc* _veh_.u_traj[inde]**2 for inde in range(min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1))]) for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])
							curr_avg -= sum([sum([data_file.W_acc* (_veh_.u_traj[inde+1] - _veh_.u_traj[inde])**2 for inde in range(1, min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1))]) for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])
							'''
						#curr_avg = curr_avg/num_of_veh

						try:
							if comb_test_flag:
								writer.writerows([[functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, comb_cost_test / num_of_veh, prov_time, coord_time, comb_test_dura, train_time, rl_ret]])		
							else:
								writer.writerows([[functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, prov_time, coord_time, train_time, rl_ret]])		

						except:
							if comb_test_flag:
								writer.writerows([[functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, comb_cost_test / num_of_veh, prov_time, coord_time, comb_test_dura, 0, rl_ret]])		
							else:
								writer.writerows([[functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, prov_time, coord_time, 0, rl_ret]])		

					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/coord_phase_Vc.csv", "a", newline="") as f:
						writer = csv.writer(f)
						#for j in wri:
						writer.writerows([len_lane_prov_set])

					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/coord_phase_Vs.csv", "a", newline="") as f:
						writer = csv.writer(f)
						#for j in wri:
						writer.writerows([len_lane_coord_set])

				#print("\n-------------------------")
				#print("len of replay buffer:", agent.buffer.buffer_counter) 
				#print("-------------------------\n")

				sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))]

				if functions.get_num_of_objects(prov_veh_copy__copy) != 0:
					for lane in data_file.lanes:
						for v in prov_veh_copy__copy[lane]:
							pre_v = None
							if len(sim_obj.prov_veh[lane]) > 0:
								pre_v = sim_obj.prov_veh[lane][-1]

							elif len(sim_obj.coord_veh[lane]) > 0:
								pre_v = sim_obj.coord_veh[lane][-1]

							prov_veh = copy.deepcopy(v)
							prov_sucess = False
							prov_anomaly = -1
							prov_anomaly += 1
							prov_veh = copy.deepcopy(v)
							prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track)

							assert len(prov_veh.p_traj) > len(v.p_traj)

							sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))


				###### to evaluate the current policy against the baselines ######

				'''if baseline_test_flag and ((round((time_track/data_file.t_opti), 1) % baseline_test_freq) == 0):

					list_of_snapshots = []
					time_of_capture = []
					list_arrival_rate = []

					for c in os.listdir("./captured_snaps/"):
					    file = open(f"./captured_snaps/{c}",'rb')
					    object_file = pickle.load(file)
					    file.close()
					    list_of_snapshots.append(copy.deepcopy(object_file[0]))
					    time_of_capture.append(copy.deepcopy(object_file[1]))
					    list_arrival_rate.append(copy.deepcopy(object_file[2]))


					for mean_arr, time, snapshot in zip(list_arrival_rate, time_of_capture, list_of_snapshots):

						num_of_prov_veh = functions.get_num_of_objects(snapshot.prov_veh)

						_, _, coord_cost_with_comb_opt_para, _, _, _, _ = coord_phase.coord_algo(time, snapshot.prov_veh, snapshot.coord_veh, algo_option, agent, learning_flag, 1)

						with open(f"./baseline_eval_data/coord_phase_info_time_{time}_arr_{mean_arr}.csv", "a", newline="") as f:
							writer = csv.writer(f)
							writer.writerows([[coord_cost_with_comb_opt_para/num_of_prov_veh]])'''

				###### evaluation against the baselines done ######
				

				### for snapshot learning ###

				'''if (len(agent._memory) >= data_file.rl_ddpg_buff_size) and (num_of_veh > 6) and (stream_rl_flag == 0):

					stream_rl_flag = 1
					prov_veh_in_if = copy.deepcopy(prov_veh_copy)
					coord_veh_in_if = copy.deepcopy(coord_veh_copy)

					rep_size_samp_size = [[100,64]]
					for ss in rep_size_samp_size:
						sim = 0
						for sim in range(data_file.num_sim):

							#agent = Agent(memory_size=ss[0], batch_size=ss[1], algo_opt=algo_option, state_size=data_file.num_features*data_file.max_vehi_per_lane*data_file.lane_max, action_size=2*data_file.max_vehi_per_lane*data_file.lane_max)

							if data_file.rl_algo_opt == "DDPG":
								if algo_option == "rl_modified_ddswa":
									agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.max_vehi_per_lane*data_file.lane_max, action_size=2*data_file.max_vehi_per_lane*data_file.lane_max)
								
								elif algo_option == "rl_ddswa":
									agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)

								agent = Agent(memory_size=ss[0], batch_size=ss[1], algo_opt=algo_option, state_size=data_file.num_features*data_file.max_vehi_per_lane*data_file.lane_max, action_size=2*data_file.max_vehi_per_lane*data_file.lane_max)

							
							elif data_file.rl_algo_opt == "MADDPG":
								if algo_option == "rl_modified_ddswa":
									agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

								elif algo_option == "rl_ddswa":
									agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)


							rl_ret_collection = []
							rl_explore_data = []

							for itr in range(200):
								print("iteration:", itr, ss[0], ss[1])
								
								success = False
								prov_veh_in_for_loop = copy.deepcopy(prov_veh_copy)
								coord_veh_in_for_loop = copy.deepcopy(coord_veh_copy)

								sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, state_t, action_t, success, rl_explore_flag = coord_phase.coord_algo(time_track, prov_veh_in_for_loop, coord_veh_in_for_loop, algo_option, agent, learning_flag, 0)

								rl_ret = coord_cost_with_comb_opt_para / num_of_veh
								rl_ret_collection.append(coord_cost_with_comb_opt_para / num_of_veh)

								rl_explore_data.append(rl_explore_flag)

								#### to run RL under rl_modified_ddswa####
								if (algo_option == "rl_modified_ddswa") and (learning_flag == 1): 
									next_state = "_"
									done = 1
									agent.remember(state_t, action_t, rl_ret, done, next_state)

									if data_file.rl_algo_opt == "DDPG":
										agent.train()

									elif data_file.rl_algo_opt == "MADDPG":
										agent.train(num_of_veh+1)

								#### to run RL under rl_modified_ddswa####

								coord_cost_with_comb_opt_para = "_"
							plt.clf()
							plt.plot(range(len(rl_ret_collection)), rl_ret_collection, "o")
							plt.savefig(f"sim_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.jpg")

							wri = []
							for k in range(len(rl_ret_collection)):
								_wri = [rl_ret_collection[k], rl_explore_data[k]]
								wri.append(_wri)
							with open(f"../rl_test_sim_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.csv", "w", newline="") as f:
							    writer = csv.writer(f)
							    for j in wri:
							   		writer.writerows([j])

							weight = agent._actor._model.get_weights()
							np.savetxt(f'weight_actor_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.csv' , weight , fmt='%s', delimiter=',')
						
							weight = agent._actor._target_model.get_weights()
							np.savetxt(f'weight_actor_target_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.csv' , weight , fmt='%s', delimiter=',')
						
							weight = agent._critic._model.get_weights()
							np.savetxt(f'weight_critic_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.csv' , weight , fmt='%s', delimiter=',')
						
							weight = agent._critic._target_model.get_weights()
							np.savetxt(f'weight_critic_target_{sim}_train_iter_{train_iter}_buff_{ss[0]}_sam_{ss[1]}.csv' , weight , fmt='%s', delimiter=',')
						
						sim += 1'''

				### snapshot learning done ###

				### RL done ###


				### code for DDSWA or combined optimization ###

			else:
				num_of_veh = functions.get_num_of_objects(sim_obj.prov_veh)
				#print("number of veh:", num_of_veh)
				prov_veh_copy = copy.deepcopy(sim_obj.prov_veh)
				coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)

				prov_veh_copy__copy = copy.deepcopy(prov_veh_copy)
				coord_veh_copy__copy = copy.deepcopy(coord_veh_copy)

				len_lane_prov_set = [0 for _ in data_file.lanes]
				len_lane_coord_set = [0 for _ in data_file.lanes]

				for _l in data_file.lanes:
					len_lane_prov_set[_l] = copy.deepcopy(len(prov_veh_copy__copy[_l]))
					len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy__copy[_l]))

				coord_time_init = time.time()			
				sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, _, _, _ = coord_phase.coord_algo(time_track, sim_obj.prov_veh, sim_obj.coord_veh, algo_option, None, None, 1, sim)
				coord_dura = time.time() - coord_time_init
				

				if num_of_veh > 0:
					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/arr_{arr_rate_}/coord_phase_info_{arr_rate_}_{sim}_train_iter_{train_iter}.csv", "a", newline="") as f:
						writer = csv.writer(f)
						curr_avg = 0
						for _ln in data_file.lanes:
							num_veh_prov_in_lane = len_lane_prov_set[_ln]
							temp_coord_veh = sim_obj.coord_veh[_ln]
							list_to_iterate = list(copy.deepcopy(temp_coord_veh))[-num_veh_prov_in_lane:]
							curr_avg += sum([data_file.W_pos*_veh_.p_traj[min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1)] for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])
							curr_avg -= sum([sum([data_file.W_acc* _veh_.u_traj[inde]**2 for inde in range(min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1))]) for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])
							curr_avg -= sum([sum([data_file.W_acc* (_veh_.u_traj[inde+1] - _veh_.u_traj[inde])**2 for inde in range(1, min(int(math.ceil(((round(data_file.T_sc,1))/round(data_file.dt,1)))), len(_veh_.p_traj)-1))]) for _veh_ in list_to_iterate if (num_veh_prov_in_lane > 0)])

						curr_avg = curr_avg/num_of_veh

						#print("\n\n\n\n curr_avg:", curr_avg, "\n\n\n")


						writer.writerows([[functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(sim_obj.prov_veh), coord_cost_with_comb_opt_para/num_of_veh, prov_time, coord_dura, curr_avg]])		

					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/arr_{arr_rate_}/coord_phase_Vc_{arr_rate_}_{sim}_train_iter_{train_iter}.csv", "a", newline="") as f:
						writer = csv.writer(f)
						#for j in wri:
						writer.writerows([len_lane_prov_set])

					with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/arr_{arr_rate_}/coord_phase_Vs_{arr_rate_}_{sim}_train_iter_{train_iter}.csv", "a", newline="") as f:
						writer = csv.writer(f)
						#for j in wri:
						writer.writerows([len_lane_coord_set])





				sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))]

				### DDSWA or combined optimization done ###

			#print("RL reward:", rl_ret)

		
		### coordinated phase done ###


		### update current time###
		time_track = round((time_track + data_file.dt), 1)
		print("current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_, "................", end="\r")
		### update current time###


		### throuput calculation ###
		for l in data_file.lanes:
			for v in sim_obj.coord_veh[l]:
				t_ind = functions.find_index(v, time_track)

				if  ((t_ind == None) or (v.p_traj[t_ind] > (v.intsize + data_file.L))) and (v.id >= throughput_id_in_lane[l]):

					throughput_id_in_lane[l] = copy.deepcopy(v.id)
					cumulative_throuput_count += 1
				
				else:
					break

		with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/throughput_{arr_rate_}_{sim}_train_iter_{train_iter}.csv", "a", newline="") as f:
			writer = csv.writer(f)
			#for j in wri:
			writer.writerows([[time_track, cumulative_throuput_count]])

		### throuput calculation ###




		### removing vehicles which have crossed the region of interest ###
		for l in data_file.lanes:
			n_in_coord = len(sim_obj.coord_veh[l])
			v_ind = 0
			while v_ind < n_in_coord:
				coord_while_flag = 0
				v = sim_obj.coord_veh[l][v_ind]
				t_ind = functions.find_index(v, time_track)

				if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length - v.int_start)):
							
					sim_obj.done_veh[v.lane].append(v)
					sim_obj.coord_veh[v.lane].popleft()
					#del v
					n_in_coord -= 1

				else:
					break

		### removed vehicles which have crossed the region of interest ###



		


	### plotting and other processing
	if data_file.rl_flag:
		wri = []
		for k in range(len(rl_ret_collection)):
			if comb_test_flag:
				_wri = [rl_ret_collection[k], comb_opt_test[k], comb_opt_test[k]-rl_ret_collection[k]]  # , (a_t[0]), (a_t[2]), (a_t[4]), (a_t[6]), (a_t[8]), (a_t[10]),
				wri.append(_wri)
				
		with open(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/rl_test_sim_{sim}_train_iter_{train_iter}.csv", "w", newline="") as f:
		    writer = csv.writer(f)
		    for j in wri:
		   		writer.writerows([j])

		plt.clf()
		plt.plot(range(len(rl_ret_collection)), rl_ret_collection, "o")
		plt.plot(range(len(comb_opt_test)), comb_opt_test, "v")
		plt.savefig(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/sim_{sim}_train_iter_{train_iter}.png", dpi=300)
		#plt.show()


if __name__ == '__main__':

	args = []

	train_iter_num = int(sys.argv[1])
	arr_rates_to_sim = data_file.arr_rates_to_simulate
	_train_iter_list = [train_iter_num]#, 500000, 1500000, 2500000, 3300000] # list(range(0, 1000, 100)) + list(range(1000, 6000, 1000)) # + list(range(10000, 50000, 5000)) + [49900] # + list(range(30000, 60000, 5000)) + [59900]

	#_train_iter_list = list(range(20000, 26000, 2500)) + [29900]
	
	for _arr_rate_ in arr_rates_to_sim:
		for _train_iter in _train_iter_list:
			for _sim_num in list(range(1, 11)): # + list(range(6, 11)):
				for _train_sim in list(range(1, 11)): # + list(range(6, 11)):
					arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
					
					file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"

					try:
						with open(f"{file_path}") as f:
							f.close()

					except:
						args.append([_train_iter, _sim_num, _train_sim, arr_rate_array_, _arr_rate_])
						print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")
						
					#func(args[-1])


	pool = Pool(18)

	pool.map(func, args)



