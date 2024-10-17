import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
#from numba import jit, cuda 
import time
import copy
import numpy as np
from collections import deque
from numpy.random import seed
import matplotlib
matplotlib.use('Agg') #Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pickle
import time
import data_file
import vehicle
import safe_set_map_bisect
import functions
import set_class
import contr_sig_rl
import cloudpickle
from multiprocessing import Pool
from multiprocessing import Process
import random
from immutabledict import immutabledict

def func(_args):

	train_iter = _args[0] 
	sim = _args[1]   
	arr_rate_array = _args[2]
	arr_rate_ = _args[3]
	version = _args[4]
	train_sim = _args[5]    
	algo_option = data_file.algo_option 
	capture_snapshot_flag = 0
	learning_flag = 0
	max_rep_sim = 1
 
	time_delay = 10*np.random.random()
	last_time = 0
 
	signal_ts = {key: None for key in range(1, data_file.max_sim_time+1)}
	override_ts = {key: None for key in range(1, data_file.max_sim_time+1)}  
	# problem == 'eight lane'	

	if data_file.problem == 'single lane':
		selected_phase = {1: 'R', 2: 'R', 4: 'G', 5: 'G', 7: 'R', 8: 'R', 10: 'R', 11: 'R' } #copy.deepcopy(data_file.phase_ref[6])    #np.random.randint(12)])

		sel_lane = []
		print(selected_phase)
		for key in selected_phase.keys():
			if selected_phase[key] == 'G':
				sel_lane.append(key)
		print(sel_lane,"selected phase=",selected_phase)
		red_phase = {1:'R',2:'R',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}    
		init_flag = False

	if data_file.rl_flag:

		import tensorflow as tf
        ##### Selection of  ddpg and MADDPG   ######
		if data_file.rl_algo_opt == "DDPG":
			from DDPG import DDPG as Agent    #DDPG_model_seq_v_one import DDPG as Agent
		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent

		wr_coun = 0
		algo_option = "rl_modified_ddswa"

		if train_iter == -1:    
			learning_flag = 1  
		else:
			learning_flag = 0

		ss = [data_file.buff_size, 64] # buffer size and sample size
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0
		agent = None 
  
		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				# print(f'***************TEST:sim---{sim},{random.random()}')
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
				if learning_flag:         
					curr_state = None
					prev_state = None

					curr_act = None
					prev_act = None

					curr_rew = None
					prev_rew = None

					curr_obs = None
					prev_obs = None
					max_rep_sim =  100 # for CML 61 episodes of data, each of 500 sec [data collection phase]
			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		elif data_file.rl_algo_opt == "MADDPG": pass
		 

	## load trained model
	if (not learning_flag) and (not data_file.env_test): 
      #f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/model_weights_itr_{int(time_track)}" 
		directory = f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/sim_data/trained_weights/model_weights_itr_{train_iter}"
		os.makedirs(directory, exist_ok=True)
		#print(os.path.isdir(directory))
		filename = f"model_weights_itr_500.weights.h5"
		file_path = os.path.join(directory, filename)
		agent.model.load_weights(file_path)		


	return_list=[]
	episode_list=[]
	phase_explore= []
	f = open("debug_over.txt", "w+")
	f.close()
	f = open("over_green.txt", "w+")
	f.close()
	f = open("RL_actions.txt", "w+")
	f.close()


 	############## simulation ######################
	for rep_sim in range(0, max_rep_sim): 

		reward_list_per_episode = []
		time_track_list = []
		cumulative_reward = 0
		time_track = 0  
		cumulative_throuput_count = 0 
		throughput_id_in_lane = [0 for _ in data_file.lanes] # variable to help track last vehicle in each lane, a list of zeros, change to 1 if particular one crosses the intersection
		sim_obj = set_class.sets() 
	

		if data_file.real_time_spawning_flag:

			############# lane independent ID ###############
			veh_id_var = 0

			############# lane dependent ID ###############
			dep_veh_id  = [(100*lane) for lane in data_file.lanes] 

			next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]  
			for lane in data_file.lanes:
				if not (arr_rate_array[lane] == 0):   #if only the arrival rate not zero we add poisson spawning else the previous values not changed
					next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)
	
	
		#### signal dict ####
		override_lane =  {}
		lane_sig_stat = {0:['R',0], 1:['R',0], 2:['R',0], 3:['R',0], 4:['R',0], 5:['R',0], 6:['R',0], 7:['R',0], 8:['R',0], 9:['R',0], 10:['R',0], 11:['R',0]}
		dict_sig = {1:'R',2:'R',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}   
		if  data_file.env_test:
			# print("inside")
			dict_sig = {1:'G',2:'G',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}   
			dict_sig_copy = {1:'G',2:'G',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}   
		### override_lane, key - lane, value - vehicle ID of bad agent   ###
		### lane signal and time of previous change  ###
		#### signal dict ####

		

		### start of simulation ###
		while (time_track < (data_file.max_sim_time)+1):  

			curr_time = time.time()
			if data_file.rl_algo_opt == "DDPG" and learning_flag and (agent.buffer.buffer_counter > 0) and ((time_track % 1) == 0):
				# learn_init_time = time.time()
				agent.buffer.learn()
				# print(f"[IN MAIN.py]: learning time: {round(time.time() - learn_init_time, 3)}")

				# tar_update_init_time = time.time()
				agent.update_target(agent.target_model.variables, agent.model.variables, agent.tau_)
				# print(f"[tar_update MAIN.py]: update time: {round(time.time() - tar_update_init_time, 3)}")

			if learning_flag and (time_track % 100) == 0:
				directory = f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/sim_data/trained_weights/model_weights_itr_{int(time_track)}"  
				os.makedirs(directory, exist_ok=True)
				filename = f"model_weights_itr_{int(time_track)}.weights.h5"
				file_path = os.path.join(directory, filename)
				agent.model.save_weights(file_path)
	
			for lane in data_file.lanes: 
				if (data_file.real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (not (arr_rate_array[lane] == 0)):
					#next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)  #NEW SPAWN TIME FOR NEXT ROBOT
					next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)), 1)
					new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
					# next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)), 1)
					# new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
					
					############### veh.ID ####################
					## for lane dependent ID
					#new_veh.id = copy.deepcopy(dep_veh_id[lane])
					#dep_veh_id[lane] += lane  #making the lane dependent ID
					# continuous ID
					new_veh.id = copy.deepcopy(veh_id_var)
					veh_id_var += 1
					################ veh.ID #######################
					new_veh.sp_t = copy.deepcopy(time_track)  
					new_veh.phase_encode = copy.deepcopy(functions.phase_encoding(new_veh, lane))
					# print(f'\n ID:{new_veh.id}, phase_encode:{new_veh.phase_encode}')
					new_veh.pos_status = 'lane'
					new_veh.goal_dist =  copy.deepcopy(data_file.int_start[lane])   ###to update
					sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))
				#################### end of spawning  ################### 	
		
				##### rear end safety validation  ##### 	
				n = len(sim_obj.unspawned_veh[lane])

				########### CHECKING MODULE ###########
				functions.seq_checker(sim_obj.unspawned_veh)
				functions.seq_checker(sim_obj.spawned_veh)
				########### CHECKING MODULE ###########
				v_itr = 0
				if n>1:print(attt)


				while v_itr<n:
					v = sim_obj.unspawned_veh[lane][v_itr]
					pre_v = None
					if len(sim_obj.spawned_veh[lane]) > 0:       
						pre_v = sim_obj.spawned_veh[lane][-1]
					if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)): 
						assert functions.check_init_config(v, pre_v, time_track) == True,f"initual condition check"
						v.sp_t = round(time_track,1)	
						if dict_sig[lane]=='R': 
							v.sig[time_track] = 'red' 
						elif dict_sig[lane]=='G': 
							v.sig[time_track] = 'green' #'green_sp'

						# v.sig_stat_sym = copy.deepcopy(lane_sig_stat[lane][1])   ### to update
						v.prev_phase_encode = [0]*12                              ### to update
						# if v.id == 0: print(f'\n before_append to_spawn-1, {v.prev_phase_encode}')

						# encode_flag = 0
						for phase_iter in range(len(data_file.phase_ref)):
							# for _ in data_file.lanes:
							if data_file.phase_ref[phase_iter] == dict_sig:
								v.prev_phase_encode[phase_iter] = 1
								# encode_flag = 1
								break
						if lane in override_lane:  	                              ###to update
							v.over_flag = 1 
						elif lane not in override_lane:  	 
							v.over_flag = 0 
						
						# if v.id == 0: print(f'\n before_append to_spawn-2, {v.prev_phase_encode}')
						sim_obj.spawned_veh[lane].append(v)
						sim_obj.unspawned_veh[lane].popleft()

						# print(f'\n ID:{sim_obj.spawned_veh[lane][-1].id}, phase_encode:{sim_obj.spawned_veh[lane][-1].prev_phase_encode}')
						assert v.sp_t == sim_obj.spawned_veh[lane][-1].sp_t and v.id == sim_obj.spawned_veh[lane][-1].id, f'spawn transfer error'
						n = len(sim_obj.unspawned_veh[lane])
					else:
						break	
			
			########### CHECKING MODULE ###########
			functions.seq_checker(sim_obj.unspawned_veh)
			functions.seq_checker(sim_obj.spawned_veh)
			for _ in data_file.lanes: assert(sorted(list(set([v.id for v in (sim_obj.spawned_veh[_])]))) ==  [v.id for  v in (sim_obj.spawned_veh[_])]), \
       			f'ID: {[v.id for  v in (sim_obj.spawned_veh[_])]}, lane:{_}, 1:{sort(list(set([v.id for v in (sim_obj.spawned_veh[_])])))}, \
               	2:{[[v.id,iter]  for  iter, v in enumerate(sim_obj.spawned_veh[_])]}'   #### check all id in a lane in ascending order
			########### CHECKING MODULE ###########
			##### END - spawning & rear end safety validation ##### 

			if functions.get_num_of_objects(sim_obj.spawned_veh) > 0 :
				spawned_veh_copy = copy.deepcopy(sim_obj.spawned_veh)

				### RL decision ###
				if learning_flag:
					prev_state = curr_state
					prev_rew = curr_rew
					# prev_obs = curr_obs
	
				################## PLOT ################################
					if (not (curr_rew == None)):
						reward_list_per_episode.append(curr_rew)
						time_track_list.append(time_track)
						cumulative_reward=cumulative_reward+curr_rew
				################## PLOT ################################


				# for _ in data_file.lanes:
				# 	for veh in sim_obj.spawned_veh[_]:
				# 		if veh.id == 0: print(f'before_rl , {veh.prev_phase_encode}')
	
				# print("\n RL actions")``
				# learn_rl_time = time.time()
				if not data_file.env_test: 
					# sim_obj.spawned_veh signal, dict_sig_copy, state_t, obs_t, action_t, explore_flag, phase_id = contr_sig_rl.rl_outputs(time_track, sim_obj.spawned_veh, agent, algo_option, learning_flag)
					sim_obj.spawned_veh,signal, dict_sig_copy, state_t, obs_t, action_t, explore_flag, phase_id = contr_sig_rl.rl_outputs(time_track, sim_obj.spawned_veh, agent, algo_option, learning_flag)

				# print("\n RL actions  end")
				# print(f"[RL query]- time: {round(time.time() - learn_rl_time, 3)}")
				

				######## alpha manual block #############
				# alpha = []
				# spawned_veh = sim_obj.spawned_veh
				# for lane in data_file.lanes:
				# 	if len(spawned_veh[lane])>0:
				# 		for v in spawned_veh[lane]:
				# 			v.alpha = random.random() #0.25     
				# 			v.beta =  random.random() #0.25 
				######## alpha manual block #############


				if  not data_file.env_test: phase_explore.append(phase_id) #DTCS
				if learning_flag:
					curr_state = copy.deepcopy(state_t)
					prev_act = copy.deepcopy(curr_act)
					curr_act = copy.deepcopy(action_t)
					# curr_obs = copy.deepcopy(obs_t)
					curr_rew = 0



					#### is override beig stored in buffer , if we not how will RL learn override is bad


				### END - RL decision ###

					########### Penalty ###########
					if functions.get_num_of_objects(sim_obj.spawned_veh)>0:
						#print(f'pos:{v.p_traj},{type(v.p_traj)}')
						if len(v.p_traj)==1: 
							curr_rew += 25*v.priority
							assert int(v.sp_t) == int(time_track)
						curr_rew -= sum([v.priority*(-v.p_traj[-1] + v.intsize) for _ in data_file.lanes for v in sim_obj.spawned_veh[_]])
					else: curr_rew = 0	
					########### Penalty ###########

				for _ in dict_sig_copy: dict_sig[_] = copy.deepcopy(dict_sig_copy[_]) #DTCS
				signal =copy.deepcopy(dict_sig)


				######## sigal manual block #############
				# if (time_track - last_time) >time_delay:
				# 	dict_sig_copy = copy.deepcopy(dict(data_file.phase_ref[np.random.randint(0,12)]))
				# 	# print(f'\n data_file_signal value:{dict_sig}')
				# 	for _ in dict_sig_copy: dict_sig[_] = copy.deepcopy(dict_sig_copy[_])
				# 	time_delay = 10 * np.random.random()
				# 	last_time = time_track
				# 	signal =copy.deepcopy(dict_sig_copy)
				# ######## sigal manual block #############
     

				if data_file.problem == 'single lane':
					for lane in sel_lane:
						if not init_flag and len(spawned_veh[lane])<7:
							dict_sig = copy.deepcopy(red_phase)
							break
						elif init_flag and len(spawned_veh[lane])<1:
							init_flag = False
							dict_sig = copy.deepcopy(red_phase)
							break
						else:
							init_flag = True
							dict_sig = copy.deepcopy(selected_phase)

				########### CHECKING MODULE ###########
				functions.seq_checker(sim_obj.spawned_veh)
				if data_file.output =='Signal': 
					assert len(dict_sig) == data_file.num_lanes*4 
				elif data_file.output =='Phase': #len(dict_sig) == data_file.num_phases
					assert len(dict_sig) == data_file.num_lanes*4, f'dict:{len(dict_sig)},phase:{data_file.num_lanes*4}'
     

				########### CHECKING MODULE ###########
				assert all ([v.id == spawned_veh_copy[_][iter].id for _ in data_file.lanes for iter, v in enumerate(sim_obj.spawned_veh[_])]),f'spawned set passed from RL WITH ERROR'
				assert all([ _!= None for _ in signal ]), f'signal value is none:{signal}, time:{time_track}'  #DTCS
				# assert all([ _!= None for _ in alpha ]), f'alpha value is none:{alpha}, time:{time_track}'
				assert all([ dict_sig[_]!= None for _ in dict_sig ]), f'signal value is none:{dict_sig}, time:{time_track}' 
				
				######### phase override for delta time ##############
				for _ in override_lane: 

 					##  dict_sig[_]='R' ####IMP:####  the specific override can get green signal in the next time step ####
					for lane in data_file.incompdict[_]:
						dict_sig[lane]='R'
				######### phase override for delta time ##############
	
				for _ in override_lane:								
					if dict_sig[_] == 'G':  
						f = open("over_green.txt", "a")
						f.write(f'\n time:{time_track}, over ID in that lane:{override_lane[_]}, ID in lane:{[ veh.id for lane in data_file.lanes for veh in sim_obj.spawned_veh[lane]]}')
						f.close()


				################## update the veh with signal values #################
				for lane in data_file.lanes:
					n = len(sim_obj.spawned_veh[lane])
					for iter in range(n):
						sim_obj.spawned_veh[lane][iter].global_sig[time_track] = copy.deepcopy(signal)
						sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = copy.deepcopy(override_lane) ### NOT REPETITIVE ###
					for veh_iter in sim_obj.done_veh[lane]:
						veh_iter.global_sig[time_track] = copy.deepcopy(signal)
						veh_iter.ovr_stat[time_track] = copy.deepcopy(override_lane) ### NOT REPETITIVE ###
				################## update the veh with signal values #################


				############# CHECKS whether all red or green due to override ##############
				cg = 0
				cr = 0
				for _ in dict_sig: 
					if dict_sig[_]=='G': cg+=1.
					elif dict_sig[_]=='R': cr+=1
				#print(f'{dict_sig},{cr},{cg}')
				if (cr ==0 or cg ==0) == True: assert len(override_lane)!=0 or functions.get_num_of_objects(sim_obj.spawned_veh)== 0,f'dict_sig:{dict_sig}, \
					over:{override_lane},num:{functions.get_num_of_objects(sim_obj.spawned_veh)}'


				###### IMP ######
				if (cr ==0 or cg ==0) == True: pass ####print(f'\n*******ALL RED time:{time_track}, ovr_sigal:{dict_sig}')   #assert False ##### never to have all red or green 
				############# CHECKS whether all red or green due to override ##############

				if len(override_lane) == 0: assert all([ dict_sig[_] == dict_sig_copy[_]  for _ in dict_sig]),f'dict_sig:{dict_sig}, or_sig:{dict_sig_copy}'


				########### control variable estimation ##############
				for lane in data_file.lanes:
					# green_zone = -1*data_file.vm[lane]*data_file.dt + (data_file.vm[lane]**2)/(2*(max(data_file.u_min[lane],-(data_file.vm[lane]/data_file.dt))))  
					if len(data_file.incompdict[lane])>0:
						override_veh = []
						if dict_sig[lane]=='G':
							n = len(sim_obj.spawned_veh[lane])
							for iter in range(n):
								
								if sim_obj.spawned_veh[lane][iter].pos_status=='inter':  
									sim_obj.spawned_veh[lane][iter] = copy.deepcopy(functions.intersection_control(sim_obj.spawned_veh[lane], iter, n, time_track, learning_flag, signal='green'))
									if lane in override_lane:
										if sim_obj.spawned_veh[lane][iter].id  in override_lane[lane]: sim_obj.spawned_veh[lane][iter].sig[time_track] = 'orange'
										elif sim_obj.spawned_veh[lane][iter].id  not in override_lane[lane]: sim_obj.spawned_veh[lane][iter].sig[time_track] = 'green' #'green_inter'
										else: assert False
									else: 
										sim_obj.spawned_veh[lane][iter].sig[time_track] = 'green' #'green_inter'

									if not learning_flag:functions.storedata(sim_obj.spawned_veh[lane][iter], train_sim, sim, train_iter, version) 

								elif sim_obj.spawned_veh[lane][iter].pos_status=='lane':   #curr_lane_veh, iter, num, time_track, signal, override_lane, lane
									sim_obj.spawned_veh[lane][iter], pre_v, success = copy.deepcopy(functions.lane_control(sim_obj.spawned_veh[lane], iter, n, time_track, 'green', override_lane, lane, learning_flag))
									if not learning_flag:functions.storedata(sim_obj.spawned_veh[lane][iter], train_sim, sim, train_iter, version) 
         
								else: 
									print('error')
									assert False,f'pos_status:{sim_obj.spawned_veh[lane][iter].pos_status}'

	
						elif dict_sig[lane]=='R':
							n = len(sim_obj.spawned_veh[lane])
							for iter in range(n):
								if sim_obj.spawned_veh[lane][iter].pos_status == 'inter':  #urr_lane_veh, iter, num, time_track, signal
									###### override on intersection ######

									sim_obj.spawned_veh[lane][iter]  = copy.deepcopy(functions.intersection_control(sim_obj.spawned_veh[lane], iter, n, time_track, learning_flag, 'green'))
									sim_obj.spawned_veh[lane][iter].sig[time_track] = 'orange'

									# print(f'\n override_veh_before_adding:{override_veh}, ID:{sim_obj.spawned_veh[lane][iter].id}')
						
									if lane in override_lane:
										if sim_obj.spawned_veh[lane][iter].id not in override_lane[lane]:
											override_veh.append(sim_obj.spawned_veh[lane][iter].id)
											# print(f'\n override_veh_after_adding:{override_veh}, ID:{sim_obj.spawned_veh[lane][iter].id}')
											# print(f'\n ovr_ride_lane before_adding:{override_lane}, ID:{sim_obj.spawned_veh[lane][iter].id}')
											override_lane[lane] = copy.deepcopy(override_veh)
											# print(f'\n ovr_ride_lane before_adding:{override_lane}, ID:{sim_obj.spawned_veh[lane][iter].id}')
											# sim_obj.spawned_veh[lane][iter].sig[time_track] = 'orange'

											assert sim_obj.spawned_veh[lane][iter].id in override_veh and override_lane[lane] == override_veh ,f'id: \
												{sim_obj.spawned_veh[lane][iter].id}, over:{override_lane}, over_list:{override_veh}'


										elif sim_obj.spawned_veh[lane][iter].id in override_lane[lane]: 
											override_veh.append(sim_obj.spawned_veh[lane][iter].id)
											# sim_obj.spawned_veh[lane][iter].sig[time_track] = 'orange'   
										else: assert False

									elif lane not in override_lane:
										override_veh.append(sim_obj.spawned_veh[lane][iter].id)
										# print(f'\n override_veh_after_adding:{override_veh}, ID:{sim_obj.spawned_veh[lane][iter].id}')
										# print(f'\n ovr_ride_lane before_adding:{override_lane}, ID:{sim_obj.spawned_veh[lane][iter].id}')
										override_lane[lane] = copy.deepcopy(override_veh)
										# print(f'\n ovr_ride_lane before_adding:{override_lane}, ID:{sim_obj.spawned_veh[lane][iter].id}')

										assert sim_obj.spawned_veh[lane][iter].id in override_veh and override_lane[lane] == override_veh ,f'id: \
											{sim_obj.spawned_veh[lane][iter].id}, over:{override_lane}, over_list:{override_veh}'

										
									# print(f'\n override_veh_after_adding:{override_veh}, ID:{sim_obj.spawned_veh[lane][iter].id}')

									if not learning_flag:
										f = open("debug_over.txt", "a")
										f.write(f'\n entry ID:{sim_obj.spawned_veh[lane][iter].id}, over:{override_lane}, at time:{time_track}')
										f.close()
          
									if not learning_flag:functions.storedata(sim_obj.spawned_veh[lane][iter], train_sim, sim, train_iter, version) 
          
								elif sim_obj.spawned_veh[lane][iter].pos_status=='lane':   #curr_lane_veh, iter, num, time_track, signal, override_lane, lane

									sim_obj.spawned_veh[lane][iter], pre_v, success = copy.deepcopy(functions.lane_control(sim_obj.spawned_veh[lane], iter, n, time_track, 'red', override_lane, lane, learning_flag))
									if success == False: ##OVERRIDE
										override_veh.append(sim_obj.spawned_veh[lane][iter].id)
										override_lane[lane] = copy.deepcopy(override_veh)

										if sim_obj.spawned_veh[lane][iter].id in override_veh: assert override_lane[lane] == override_veh,f'id {sim_obj.spawned_veh[lane][iter].id}, ovr{override_lane}'	
										if not learning_flag:
											f = open("debug_over.txt", "a")
											f.write(f'\n entry ID:{sim_obj.spawned_veh[lane][iter].id}, over:{override_lane},  at time:{time_track}')
											f.close()

										if pre_v == None or pre_v.id in override_lane[lane]:   #curr_lane_veh, iter, num, time_track, signal, override_lane, lane
											sim_obj.spawned_veh[lane][iter], pre_v, success = copy.deepcopy(functions.lane_control(sim_obj.spawned_veh[lane], iter, n, time_track, 'lane_over', override_lane, lane, learning_flag))
											sim_obj.spawned_veh[lane][iter].sig[time_track] = 'orange'
										else: assert False,f'pos_prev:{pre_v.p_traj}, id:{pre_v.id}, pos_curr:{sim_obj.spawned_veh[lane][iter].p_traj}, id:{sim_obj.spawned_veh[lane][iter].id}'

									if not learning_flag:functions.storedata(sim_obj.spawned_veh[lane][iter], train_sim, sim, train_iter,version)
								else: 
									print(f'error, time:{time_track}, pos_status:{sim_obj.spawned_veh[lane][iter].pos_status}, pos:{sim_obj.spawned_veh[lane][iter].p_traj}')
									assert False

							assert len(set(override_veh)) == len(override_veh),f'duplicates present in over_ride lane: {override_veh}'
							if lane in override_lane:  assert override_lane[lane] == override_veh

							# if len(override_veh)>0: override_lane[lane] = copy.deepcopy(override_veh)   #### final update 
						else: assert False,f'signal neither green or red'	
      
				assert len(set(override_lane.keys())) == len(override_lane),f'duplicates present in over_ride lane: {override_lane}'
				#assert all([list(override_lane.keys())[_] not in override_lane  for _ in range(len(override_lane))]),f'duplicates present :{override_lane}'

				############# override trajectories of incompatible lanes  #############
				##### Note ######### : bad agent shouldn't get this trajectory ########
				for _ in override_lane:
					# dict_sig[_]='R'    ######   chnages #####
					for lane in data_file.incompdict[_]:
						dict_sig[lane] = 'R'   #### override
						n = len(sim_obj.spawned_veh[lane])
						for iter in range(n):
							if sim_obj.spawned_veh[lane][iter].pos_status=='lane':
								v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
								success = None
								pre_v = None
								if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
								sim_obj.spawned_veh[lane][iter], success = copy.deepcopy(safe_set_map_bisect.acc_map(v, pre_v, time_track, learning_flag, flag = 'red')) 
								assert success == True #False
								if not learning_flag: functions.storedata(sim_obj.spawned_veh[lane][iter], train_sim, sim, train_iter, version) 
							else: assert False
				########### override trajectories of incompatible lanes  ##############
	
				# assert all ([veh.p_traj[-1]< sim_obj.spawned_veh[lane][0].p_traj[-1] or (veh.p_traj[-1] == sim_obj.spawned_veh[lane][0].p_traj[-1] and veh.id == sim_obj.spawned_veh[lane][0].id) \
				# 	for lane in data_file.lanes for veh in sim_obj.spawned_veh[lane]]),f'robot skippped---position exceeded,'

				assert all ([veh.p_traj[-1]< sim_obj.spawned_veh[lane][index].p_traj[-1] or (veh.p_traj[-1] == sim_obj.spawned_veh[lane][0].p_traj[-1] and veh.id == sim_obj.spawned_veh[lane][0].id) \
				for lane in data_file.lanes for count, veh in enumerate(sim_obj.spawned_veh[lane]) for index in range(count) ]),f'robot skippped---position exceeded,'


				for _ in data_file.lanes:
					for veh in sim_obj.spawned_veh[_]:
						if  (data_file.L + veh.intsize) >= veh.p_traj[-1] > data_file.int_start[_]:
          
							if veh.sig[time_track]==None:
								print(f'\n ID:{veh.id}, time:{time_track}, \n signal_dict:{veh.sig}, pos:{veh.pos_status}\n pos:{veh.p_traj},{len(veh.p_traj)} \n acc:{veh.u_traj},\
									{len(veh.u_traj)}, signal:{dict_sig[_]}, pos_stat:{veh.pos_status}, lane{_}')
								assert False
							else: pass

							assert veh.sig[time_track]  in {'red', 'orange', 'green'} and dict_sig[_] in {'R', 'G'}, f'veh.sig:{veh.sig[time_track]}, dict_sig:{dict_sig[_] }'
       
							if (veh.sig[time_track] == 'red' and dict_sig[_] == 'G') or (veh.sig[time_track] == 'green' and dict_sig[_] == 'R'):
								print(f'\n robot wrong traj-3, lane:{_} id: {veh.id}, time:{time_track}, sig: {veh.sig[time_track]}, dict_sig: {dict_sig[_]}, \n pos: {veh.p_traj}, overeide:{override_lane} \n sig_stat:{veh.sig}, pos:{veh.pos_status}'  )
								assert False
							else: pass	
       
							if _ in override_lane:
								if veh.id in override_lane[_]:
									if (veh.sig[time_track] != 'orange'):
										print(f'\n robot wrong traj, lane:{_} id: {veh.id}, time:{time_track}, sig: {veh.sig[time_track]}, dict_sig: {dict_sig[_]}, \n pos: {veh.p_traj}, overeide:{override_lane} \n sig_stat:{veh.sig}, pos:{veh.pos_status}'  )
										assert False,f'sig:{veh.sig[time_track]}'
							else: pass	
       
						elif (data_file.L + veh.intsize) >= veh.p_traj[-2] > data_file.int_start[_]: pass # assert False,f'pos:{veh.p_traj}, pos_status:{veh.pos_status}'
						else: assert False


				if len(override_lane) == 0: assert all([ dict_sig[_] == dict_sig_copy[_]  for _ in dict_sig]),f'dict_sig:{dict_sig}, or_sig:{dict_sig_copy}'
    
    
    
    
				###################### need to check this code ##############################

				############# update veh and REF dict with signal and time ##################
				for lane in data_file.lanes:
					if len(data_file.incompdict[lane])>0:
						if lane_sig_stat[lane][0] !=  dict_sig[lane]:
							lane_sig_stat[lane][0] =  copy.deepcopy(dict_sig[lane])
							lane_sig_stat[lane][1] =  time_track
						n = len(sim_obj.spawned_veh[lane])
						if dict_sig[lane]=='G': sig = 1 
						elif dict_sig[lane]=='R': sig = 0
						for iter in range(n):
							sim_obj.spawned_veh[lane][iter].sig_stat = copy.deepcopy(sig)
							# sim_obj.spawned_veh[lane][iter].sig_stat_sym = copy.deepcopy(lane_sig_stat[lane][1]) 
							sim_obj.spawned_veh[lane][iter].over_sig[time_track] = copy.deepcopy(dict_sig)
							sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = copy.deepcopy(override_lane)
							sim_obj.spawned_veh[lane][iter].goal_dist = copy.deepcopy(sim_obj.spawned_veh[lane][iter].p_traj[-1])
							sim_obj.spawned_veh[lane][iter].prev_phase_encode = [0]*12                
							for phase_iter in data_file.phase_ref.keys(): #range(len([data_file.phase_ref.keys()][0])):
								if data_file.phase_ref[phase_iter] == dict_sig:
									sim_obj.spawned_veh[lane][iter].prev_phase_encode[phase_iter] = 1
									break
							if lane in override_lane:  	                              
								sim_obj.spawned_veh[lane][iter].over_flag = 1 
							elif lane not in override_lane:  	 
								sim_obj.spawned_veh[lane][iter].over_flag = 0 


						for veh_ in sim_obj.done_veh[lane]:
							if functions.done_pos_check(veh_):    
								veh_.sig_stat = copy.deepcopy(sig)
								# veh_.sig_stat_sym = copy.deepcopy(lane_sig_stat[lane][1]) 
								veh_.over_sig[time_track] = copy.deepcopy(dict_sig)  
								veh_.ovr_stat[time_track] = copy.deepcopy(override_lane)
								veh_.goal_dist = copy.deepcopy(veh_.p_traj[-1])    #to entry of intersection
								veh_.prev_phase_encode = [0]*12                

								# print(f'\n ERRROR CHECK************** keys:{data_file.phase_ref.keys()}, len:{len([data_file.phase_ref.keys()][0])} ')
								# exit()
								for phase_iter in data_file.phase_ref.keys():  #range(len([data_file.phase_ref.keys()][0])):
									if data_file.phase_ref[phase_iter] == dict_sig:
										veh_.prev_phase_encode[phase_iter] = 1
										break
								if lane in override_lane:  	                              
									veh_.over_flag = 1 
								elif lane not in override_lane:  	 
									veh_.over_flag = 0 

				############# update veh and REF dict  with signal and time  ##################

				###################### need to check this code ##############################


			if not learning_flag:
				signal_ts[time_track] = copy.deepcopy(dict_sig)
				override_ts[time_track] = copy.deepcopy(override_lane)


			override_lane_copy = copy.deepcopy(override_lane)		
			### Done  ###
			for lane in data_file.lanes:
				if  lane in override_lane: c = len(override_lane[lane])
				else: c = 0
				if len(sim_obj.done_veh[lane])>0:
					for iter in range(len(sim_obj.done_veh[lane])):
						if functions.done_pos_check(sim_obj.done_veh[lane][iter]) and sim_obj.done_veh[lane][iter].pos_status == 'done':
							temp_ind = functions.find_index(sim_obj.done_veh[lane][iter], time_track)
							assert temp_ind!=None,f'time_track:  {sim_obj.done_veh[lane][iter].t_ser}, time:{time_track} \n pos:{sim_obj.done_veh[lane][iter].p_traj},pos{functions.done_pos_check(sim_obj.done_veh[lane][iter])}, id:{sim_obj.done_veh[lane][iter].id}'
							sim_obj.done_veh[lane][iter].sig[time_track] = 'lime'
							# sim_obj.done_veh[lane][iter].pos_status = 'done'
							sim_obj.done_veh[lane][iter]  = copy.deepcopy(functions.max_physical_acc(sim_obj.done_veh[lane][iter], time_track))
							if functions.done_pos_check(sim_obj.done_veh[lane][iter]): 
								sim_obj.done_veh[lane][iter].t_ser.append(round((time_track + (data_file.dt)), 1))
								# print(f'%%%%%%%%%%%%%%%%%% cuur_time:{time_track}, t_ser:{sim_obj.done_veh[lane][iter].t_ser}, ID:{sim_obj.done_veh[lane][iter].id}')
							else: 
								pass
						elif not(functions.done_pos_check(sim_obj.done_veh[lane][iter])) and sim_obj.done_veh[lane][iter].pos_status == 'done':
								# print(f'ID:{sim_obj.done_veh[lane][iter].id},  pos:value:{ sim_obj.done_veh[lane][iter].p_traj[-1] }')
								sim_obj.done_veh[lane][iter].pos_status = 'done_end'							
								sim_obj.done_veh[lane][iter].sig[time_track] = 'lime'
						elif not(functions.done_pos_check(sim_obj.done_veh[lane][iter])) and sim_obj.done_veh[lane][iter].pos_status == 'done_end': pass
						else: assert False
      
						assert sim_obj.done_veh[lane][iter].sig[time_track] != None or not(functions.done_pos_check(sim_obj.done_veh[lane][iter])),f'sig:{sim_obj.done_veh[lane][iter].sig[time_track]}'
						if not learning_flag: functions.storedata(sim_obj.done_veh[lane][iter], train_sim, sim, train_iter,version) 

				if len(sim_obj.spawned_veh[lane])>0:		
					ind  = 0
					veh_num = len(sim_obj.spawned_veh[lane])
					while ind < veh_num:
						veh_iter = sim_obj.spawned_veh[lane][ind]
						temp_ind = functions.find_index(veh_iter, time_track)
						if veh_iter.p_traj[-1] > (veh_iter.intsize + data_file.L) and veh_iter.pos_status == 'inter':         
						# if veh_iter.p_traj[temp_ind] > (veh_iter.intsize + data_file.L) and veh_iter.pos_status == 'inter':  
							veh_iter.pos_status = 'done'
							sim_obj.done_veh[lane].append(veh_iter)
							sim_obj.spawned_veh[lane].popleft()
							veh_num =  len(sim_obj.spawned_veh[lane])
							if lane in override_lane:
								if veh_iter.id in override_lane[lane]: 
									if not learning_flag:
										f = open("debug_over.txt", "a")
										f.write(f'\n exit ID:{veh_iter.id}, over:{override_lane}, removed at time:{time_track}\n')
										f.close()
									if c>1: 
										override_lane[lane].remove(veh_iter.id)
									c -= 1
							else: pass #print(f'\n ************done************id:{veh_iter.id}, override{override_lane}, time:{time_track}')
						else: break
      
				if c==0 and lane in override_lane and len(override_lane[lane]) : del override_lane[lane]


				for veh_ in sim_obj.done_veh[lane]:
					if functions.done_pos_check(veh_): pass
						# veh_.sig_stat = sig
						# veh_.sig_stat_sym = lane_sig_stat[lane][1] 
						# veh_.over_sig[time_track] = dict_sig  
						# veh_.ovr_stat[time_track] = override_lane
						# if not learning_flag: functions.storedata(veh_, train_sim, sim, train_iter,version) 	

				assert all([ veh.id not in override_lane_copy.values() and (veh.pos_status=='done' or veh.pos_status=='done_end')  for lane in data_file.lanes for veh in sim_obj.done_veh[lane] ])



				for veh_iter in sim_obj.spawned_veh[lane]:
					if veh_iter.p_traj[-1] > 0 and veh_iter.pos_status == 'lane':     
						veh_iter.pos_status = 'inter'
						if learning_flag : curr_rew += 25*veh_iter.priority	#### reward entering intersection 


			
				for veh_iter in sim_obj.spawned_veh[lane]:
					if veh_iter.p_traj[-1] > 0 and veh_iter.pos_status == 'lane':     ###############try out:     veh_iter.length:     
						veh_iter.pos_status = 'inter'

			for lane in data_file.lanes:
				for veh in sim_obj.done_veh[lane]:
					if functions.done_pos_check(veh):
						if veh.sig[time_track]==None:
							print(f'\n ID:{veh.id}, time:{time_track}, \n signal_dict:{veh.sig}, pos:{veh.pos_status}\n pos:{veh.p_traj},{len(veh.p_traj)} \n acc:{veh.u_traj},\
								{len(veh.u_traj)}, signal:{dict_sig[lane]}, pos_stat:{veh.pos_status}, lane{lane}')
							assert False
						else: pass   #print(f'\n time:{veh.sig[time_track]}')


			for _ in data_file.lanes:
				for veh in sim_obj.spawned_veh[_]:
					if (veh.p_traj[-1]<= 0 and veh.pos_status == 'lane') or (veh.p_traj[-1]> 0 and veh.pos_status == 'inter'):pass
					else: 
						print(f'veh failed: lane:{_},{veh.p_traj},{veh.t_ser},status:{veh.pos_status}, time:{time_track}')
						assert False
				

			# assert all([(veh_.p_traj[-1]<= 0 and veh_.pos_status == 'lane') or (veh_.p_traj[-1]> 0 and veh_.pos_status == 'inter') \
			# 	for lane in data_file.lanes  for veh_ in sim_obj.spawned_veh[lane]]),f'error in spawn in done set,pos:' 

			for _ in data_file.lanes:
				for veh in sim_obj.spawned_veh[_]:
					if veh.sig[time_track]==None:
						print(f'ID:{veh.id}, time:{time_track}, \n signal_dict:{veh.sig}, pos:{veh.pos_status}\n pos:{veh.p_traj},{len(veh.p_traj)} \n acc:{veh.u_traj},\
							{len(veh.u_traj)}, signal:{dict_sig[_]}, pos_stat:{veh.pos_status}, lane{_}')
						assert False
					else: pass

	

			### intersection ###
			### update current time###
			time_track = round((time_track + data_file.dt), 1)
			if learning_flag: #pass
				print(f"arr_rate: {arr_rate_}, rep: {rep_sim}", "current time:", time_track, "sim:", sim, "train_iter:", train_iter,"size_buff",{(sys.getsizeof(agent.buffer.state_buffer)+ sys.getsizeof(agent.buffer.reward_buffer) + sys.getsizeof(agent.buffer.action_buffer) + sys.getsizeof(agent.buffer.next_state_buffer))/(1024*1024)},"......", end="\r")
			else: #pass
				print("arr_rate:", arr_rate_,"current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_,"********", end="\r")# "heuristic:", data_file.used_heuristic, "................", end="\r")
			### update current time###
			### throuput calculation ###
			for l in data_file.lanes:
				for v in sim_obj.spawned_veh[l]:
					t_ind = functions.find_index(v, time_track)
					if  ((t_ind == None) or (v.p_traj[t_ind] > (v.intsize + data_file.L))) and (v.id >= throughput_id_in_lane[l]):
						throughput_id_in_lane[l] = copy.deepcopy(v.id)
						cumulative_throuput_count += 1
					else:
						break
			### throuput calculation ###
			## removed vehicles which have crossed the region of interest ###


			### storing data in buffer
			if (algo_option == "rl_modified_ddswa") and (learning_flag):
				if (not (prev_rew == None)) and (not (len(prev_state) == 0)):

					# print(f'prev_state:{prev_state},prev_act:{prev_act.shape},prev_rew:{prev_rew.shape},curr_state:{curr_state}')
					# print(f'prev_state:{type(prev_state)},prev_act:{type(prev_act)},prev_rew:{type(prev_rew)},curr_state:{type(curr_state)}')

					agent.buffer.remember((prev_state, prev_act, prev_rew, curr_state))
					qwe = {}
					qwe["state_buffer"] = agent.buffer.state_buffer
					#qwe["observe_buffer"] = agent.buffer.observe_buffer
					qwe["action_buffer"] = agent.buffer.action_buffer
					qwe["reward_buffer"] = agent.buffer.reward_buffer
					qwe["next_state_buffer"] = agent.buffer.next_state_buffer
					#qwe["next_observe_buffer"] = agent.buffer.next_observe_buffer
					if rep_sim%10==0 and time_track == data_file.max_sim_time:
						#print(f'ver:{version}, arr: {arr_rate_}, sim: {sim}')
						dbfile = open(f'../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/replay_buffer_sim_{sim}','wb')
						pickle.dump(qwe, dbfile,protocol = pickle.HIGHEST_PROTOCOL)
						dbfile.close()
						#print(dbfile)
						"""
						directory = f' ../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}'
						os.makedirs(directory, exist_ok=True)      
						filename = f'replay_buffer_sim_{sim}'
						file_path = os.path.join(directory, filename)
						"""
						###################################
			# print(f"[END MAIN.PY]: one timestep time: {round(time.time() - curr_time, 3)}")
			############## simulation ######################

			##############################
		return_list.append(cumulative_reward)
		episode_list.append(rep_sim)
		'''if (rep_sim)%20==0:
			#print(f'len,{len(reward_list_per_episode)}')
			plt.plot(time_track_list,reward_list_per_episode)
			plt.xlabel('Track_time')
			plt.ylabel('Rewards')
			plt.title(f'rewards for episode: {rep_sim}')
			plt.savefig(f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/Rewards for episode:{rep_sim}",format='png')
			plt.clf()''' #DTCS
	
		### end of simulation ###
	
	
	if learning_flag :
		#print('return',return_list)
		plt.plot(episode_list,return_list)
		plt.xlabel('episodes')
		# naming the y axis
		plt.ylabel('Returns')
		plt.title(f'Returns for all the episodes')
		plt.savefig(f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/Returns of version:{version}",format='png')
			
		f = open("explore_details.txt", "w+")

		f.write(f'pahse explorartion: 0:{phase_explore.count(0)},1:{phase_explore.count(1)},2:{phase_explore.count(2)},3:{phase_explore.count(3)},4:{phase_explore.count(4)}, \
			5:{phase_explore.count(5)},6:{phase_explore.count(6)},7:{phase_explore.count(7)},8:{phase_explore.count(8)},9:{phase_explore.count(9)},10:{phase_explore.count(10)}')
		f.close()


if __name__ == '__main__':


	def init_pool_processes():
		seed()

	arr_rates_to_sim = data_file.arr_rates_to_simulate  #The 10 diff values from 0.01 to 0.1
	args = []
    

	if data_file.used_heuristic == None:
		if data_file.rl_flag:
			train_or_test = str(sys.argv[1])
			if train_or_test == "--train":
				##### for cluster #########
				_arr_rate_ = float(sys.argv[4]) 
				version = int(sys.argv[5]) 
				##### for cluster #########
				for _train_iter in range(1,2):
					for _sim_num in range(1, 2):
						#for _arr_rate_ in arr_rates_to_sim:
							arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
							args.append([-1, _sim_num, arr_rate_array_, _arr_rate_,version, _train_iter])
							# func(args[-1])
				func(args[0])
 
			elif train_or_test == "--test":
				#print(sim_obj.spawned_veh[0][0])
				if not data_file.run_coord_on_captured_snap:
					_train_iter_list = [int(sys.argv[2])]
					version = int(sys.argv[3]) 
					for _train_iter in _train_iter_list:
						for _sim_num in range(1,51):  #11
							############################### edited ################ # each policy run at speicific arrival rate for 10 times to increase the samples.
							for _train_sim in list(range(1,2)):   ##### edited############### 11 $#############################
								for _arr_rate_ in [0.1]: #arr_rates_to_sim:
									arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
									# file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}"								
									file_path = f"../d_version/version_{int(version)}/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"
									try:
										with open(f"{file_path}") as f:
											f.close()
									except:
										args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_,version, _train_sim])
					pool = Pool(6,initializer=init_pool_processes)
					pool.map(func, args)
				else:
					_arr_rate_ = 0.08
					for _sim_num in range(1,4):
						args.append([5000,_sim_num,0,_arr_rate_,8])
						func(args[-1])				

		elif not data_file.run_coord_on_captured_snap:
			for _arr_rate_ in arr_rates_to_sim:
				arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
				for _sim_num in range(1, 101):
					args.append([0, _sim_num, arr_rate_array_, _arr_rate_, 0])
					func(args[-1])

		else:
			_arr_rate_ = 0.08
			for _sim_num in range(1,4):
				args.append([0,_sim_num,0,_arr_rate_,0])
				func(args[-1])

	else:

		for _train_iter in [0]:
			for _sim_num in range(1, 101): # 100 diff simulations
				for _train_sim in list(range(1)):
					for _arr_rate_ in arr_rates_to_sim:

						arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} 
						heuristics_pickobj_save_path = f"../data/{data_file.used_heuristic}/arr_{_arr_rate_}/pickobj_sim_{_sim_num}"

						# if len(os.listdir(f"{heuristics_pickobj_save_path}")) < 290:
						args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
						
						# else:
						# 	...

						# print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

		pool = Pool(18)
		pool.map(func, args)








