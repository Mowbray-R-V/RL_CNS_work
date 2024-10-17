import glob
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
#from numba import jit, cuda 
import time
import copy
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pickle
import time
import data_file
#import vehicle
import safe_set_map_bisect_backup1
import functions
import set_class
import vehicle_snap as vehicle
import contr_sig_rl_snap_shot as contr_sig_rl
import cloudpickle
from multiprocessing import Pool
import csv
from numpy.random import seed
import random

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
 
 
	signal_ts = {key: None for key in range(1, data_file.max_sim_time+1)}
	override_ts = {key: None for key in range(1, data_file.max_sim_time+1)}  


	if data_file.rl_flag:

		import tensorflow as tf
        ##### Selection of  ddpg and MADDPG   ######
		if data_file.rl_algo_opt == "DDPG":
			from DDPG_graph_sanity_test_snapshot import DDPG as Agent    #DDPG_model_seq_v_one import DDPG as Agent
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
					max_rep_sim = 5 # for CML 61 episodes of data, each of 500 sec [data collection phase]
			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		elif data_file.rl_algo_opt == "MADDPG": pass

	## load trained model
	# if (not learning_flag): 
	# 	directory = f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/sim_data/trained_weights/model_weights_itr_{train_iter}"
	# 	#print(f'train_suim:{train_sim},\n dir:{directory}')
	# 	os.makedirs(directory, exist_ok=True)
	# 	#print(os.path.isdir(directory))
	# 	filename = f"model_weights_itr_500.weights.h5"
	# 	file_path = os.path.join(directory, filename)
	# 	agent.model.load_weights(file_path)		


	return_list=[]
	episode_list=[]
	phase_explore= []
	f = open("debug.txt", "w+")
	f.close()
	############## streamline ######################
	if not data_file.run_coord_on_captured_snap:
		for rep_sim in range(0, max_rep_sim): # through DDPG it will 61 episodes
			reward_list_per_episode=[]
			time_track_list=[]
			cumulative_reward=0
			cumulative_throuput_count = 0  # cumulative throuput
			time_track = 0
			cumulative_throuput_count = 0  # cumulative throuput
			throughput_id_in_lane = [0 for _ in data_file.lanes] # variable to help track last vehicle in each lane, a list of zeros, change to 1 if particular one crosses the intersection
			sim_obj = set_class.sets() 
        
			if not data_file.real_time_spawning_flag:  
				sim_list =  {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]} 
				id_extract = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]} 
				veh_pic = []
				path = f'../data_version/version_{int(version)}/arr_0.1/test_homo_stream/train_sim_1/train_iter_500/pickobj_sim_1'
				# print(f'********************************************************path:{path}')
				for filename in sorted(glob.glob(os.path.join(path,'*'))):
					with open(os.path.join(os.getcwd(), filename), 'rb') as f:
						# print(f'{f}')
						veh_pic= pickle.load(f)
						##print(f"{os.path.basename(filename)},{veh_pic[int(os.path.basename(filename))]}")
						#print(f"{veh_pic[os.path.basename(filename)].lane}")
						sim_list[veh_pic[int(os.path.basename(filename))].lane].append(veh_pic)
						#print(f'filename:{filename},*********{os.path.basename(filename)},{type(int(os.path.basename(filename)))}')
						#print(f'ID######{sim_list[-1][int(os.path.basename(filename))].lane}')
						id_extract[veh_pic[int(os.path.basename(filename))].lane].append(veh_pic[int(os.path.basename(filename))].id) 
						id_extract[veh_pic[int(os.path.basename(filename))].lane].sort()
						# if veh_pic[int(os.path.basename(filename))].id == 19:	
						# 	print(f"writing values of ID:{veh_pic[int(os.path.basename(filename))].id}")
						# 	f = open("debug.txt", "a")
						# 	f.write(f'\n ID:{veh_pic[int(os.path.basename(filename))].id} values: \n pos:{veh_pic[int(os.path.basename(filename))].p_traj}')
						# 	f.write(f'\n spawn_values:{veh_pic[int(os.path.basename(filename))].sp_t}')
						# 	f.write(f'\nvel:{veh_pic[int(os.path.basename(filename))].v_traj}')
						# 	f.write(f'\nacc:{veh_pic[int(os.path.basename(filename))].u_traj}')
						# 	f.write(f'\ntime:{veh_pic[int(os.path.basename(filename))].t_ser}')
						# 	f.write(f'\n sig_status:{veh_pic[int(os.path.basename(filename))].sig}')
						# 	f.write(f'\nlenght values: time:{len(veh_pic[int(os.path.basename(filename))].t_ser)},\
           				# 		acc:{len(veh_pic[int(os.path.basename(filename))].u_traj)},vel:{len(veh_pic[int(os.path.basename(filename))].v_traj)}, \
                     	# 		pos:{len(veh_pic[int(os.path.basename(filename))].p_traj)}')
						# 	temp_ind = functions.find_index(veh_pic[int(os.path.basename(filename))], 121)
						# 	f.write(f'\n pos at 121:{veh_pic[int(os.path.basename(filename))].p_traj[temp_ind]}, u: {veh_pic[int(os.path.basename(filename))].u_traj[temp_ind]} ')
							# f.write(f'\n acc_max:{veh_pic[int(os.path.basename(filename))].u_traj[temp_ind]}')
							# f.write(f'\n acc_min:{veh_pic[int(os.path.basename(filename))].u_traj[temp_ind]}')


							# f.close()
						# if veh_pic[int(os.path.basename(filename))].id == 3:	
						# 	f = open("debug.txt", "a")
						# 	f.write(f'\n ID{veh_pic[int(os.path.basename(filename))].id} values: \n pos:{veh_pic[int(os.path.basename(filename))].p_traj}')
						# 	f.write(f'\nvel:{veh_pic[int(os.path.basename(filename))].v_traj}')
						# 	f.write(f'\nacc:{veh_pic[int(os.path.basename(filename))].u_traj}')
						# 	f.write(f'\ntime:{veh_pic[int(os.path.basename(filename))].t_ser}')
						# 	f.write(f'\nlenght values: time:{len(veh_pic[int(os.path.basename(filename))].t_ser)},\
           				# 		acc:{len(veh_pic[int(os.path.basename(filename))].u_traj)},vel:{len(veh_pic[int(os.path.basename(filename))].v_traj)}, \
                     	# 		pos:{len(veh_pic[int(os.path.basename(filename))].p_traj)}')

						# 	f.close()


						
    			#print(f'**************Id_extract: {id_extract} \n')
				#print(f'**************sim_list:{sim_list}')

				#print(f'sdfghjk:{type(sim_list[1])},{sim_list[1]}')
				max_rep_sim = 2
				lane_sig_stat = {0:['R',0], 1:['R',0], 2:['R',0], 3:['R',0], 4:['R',0], 5:['R',0], 6:['R',0], 7:['R',0], 8:['R',0], 9:['R',0], 10:['R',0], 11:['R',0]}
				dict_sig = {1:'R',2:'R',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}    

				
				for lane_iter in data_file.lanes: 
					for sort_id in id_extract[lane_iter]:
						for _ in sim_list[lane_iter]: 
							if sort_id in _ : 
								#sim_obj.spawned_veh[lane].append(_[sort_id])
								#print(f'dddd:{_[sort_id].id}')
								lane =_[sort_id].lane
								init_vel = _[sort_id].v0
								prior = _[sort_id].priority
								init_pos = _[sort_id].p0
								new_veh = vehicle.Vehicle(lane,init_pos ,init_vel, 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array, prior)
								new_veh.id = copy.deepcopy(_[sort_id].id)
								new_veh.sp_t = copy.deepcopy(_[sort_id].sp_t)
								#new_veh.p0 = copy.deepcopy(_[sort_id].p0)
								new_veh.sig_stat_t = lane_sig_stat[lane_iter][1] 
								new_veh.pos_status ='lane'
								# print(f'{new_veh.sig_stat_t},{lane_sig_stat[lane_iter][1] },{lane_sig_stat},{lane_iter}***,{new_veh.id}')
								new_veh.sig_stat = 0
								#sim_obj.spawned_veh[lane].append(copy.deepcopy(new_veh))
								new_veh.u_traj = copy.deepcopy(_[sort_id].u_traj)
								new_veh.u_safe_max = copy.deepcopy(_[sort_id].u_safe_max)
								new_veh.u_safe_min = copy.deepcopy(_[sort_id].u_safe_min)      
								new_veh.sig = copy.deepcopy(_[sort_id].sig)
								new_veh.p_traj = copy.deepcopy(_[sort_id].p_traj)
								new_veh.v_traj = copy.deepcopy(_[sort_id].v_traj)
								new_veh.t_ser = copy.deepcopy(_[sort_id].t_ser)
								new_veh.alpha_dict = copy.deepcopy(_[sort_id].alpha_dict)
								sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))

				for time_ in range(0,500):
					lane_no = 1
					for _ in range(len(sim_obj.unspawned_veh[lane_no])):
						#print(f'\n\n')
						if (sim_obj.unspawned_veh[lane_no][_].id == 23 or sim_obj.unspawned_veh[lane_no][_].id ==110) and 48<time_<51:
							temp_ind = functions.find_index(sim_obj.unspawned_veh[lane_no][_], time_)
							# print(f'tser:{sim_obj.unspawned_veh[4][_].t_ser}')
							# print(f'ID:{sim_obj.unspawned_veh[4][_].id}, pos:{sim_obj.unspawned_veh[5][_].p_traj}, \n\n sig_values:{sim_obj.unspawned_veh[5][_].sig}')
							# print(f'ID:{sim_obj.unspawned_veh[4][0].id}, umax:{(sim_obj.unspawned_veh[4][0].u_safe_max)}, \n \n u_len:{len(sim_obj.unspawned_veh[4][0].u_traj)}   time_:{time_},{temp_ind},\
							# 	t_ser:{sim_obj.unspawned_veh[4][0].t_ser}, t_ser_len:{len(sim_obj.unspawned_veh[4][0].t_ser)}')
							# exit()
							f = open("debug.txt", "a")
							f.write(f'\n time:{time_}, ID:{sim_obj.unspawned_veh[lane_no][_].id}')
							# f.write(f'\n time:\t{sim_obj.unspawned_veh[lane_no][_].t_ser[temp_ind]}\n')
							# f.write(f'\tID:{sim_obj.unspawned_veh[lane_no][_].id}\t')
							f.write(f'\tpos:{sim_obj.unspawned_veh[lane_no][_].p_traj[temp_ind]}')
							f.write(f'\tvel:{sim_obj.unspawned_veh[lane_no][_].v_traj[temp_ind]}')
							f.write(f'\tU*:{sim_obj.unspawned_veh[lane_no][_].u_traj[temp_ind]}')
							f.write(f'\t u_min:{sim_obj.unspawned_veh[lane_no][_].u_safe_min[time_]}')
							f.write(f'\t u_max:{sim_obj.unspawned_veh[lane_no][_].u_safe_max[time_]}\n')
							f.write(f'\t signal:{sim_obj.unspawned_veh[lane_no][_].sig}')
							f.write(f'\npos:{sim_obj.unspawned_veh[lane_no][_].p_traj}')
							# print(f'*********:{sim_obj.unspawned_veh[lane_no][_].intsize +  sim_obj.unspawned_veh[lane_no][_].length - sim_obj.unspawned_veh[lane_no][_].int_start/2}')
							# f.write(f'\t time_:{sim_obj.unspawned_veh[lane_no][_].t_ser}')
							# f.write(f'\n pos:{sim_obj.unspawned_veh[lane_no][_].p_traj}\n')
							f.close()
				exit("END")



				# print(f'******************veh_13 id:{ [_.id  for _ in sim_obj.unspawned_veh[1]]}')
				# exit()				
				#print(f'values_in lane 7: {sim_obj.unspawned_veh[7][-1].p_traj}')
				
				#print(f"veh:{sim_obj.unspawned_veh}\n,{id_extract}")            
				# print(f"veh:{sim_obj.spawned_veh[1][0].id}")            
				# print("********8")
				# for iter in sim_obj.unspawned_veh[7]:
				#         print(f'-------{iter.id}')
				functions.seq_checker(sim_obj.unspawned_veh)
				total_veh_in_simulation = functions.get_num_of_objects(sim_obj.unspawned_veh) 
			"""      
                print("************")
                veh_o  = []
                for li in data_file.lanes:
                    veh_o = []
                    for iter in sim_obj.unspawned_veh[li]:
                        veh_o.append(iter.id)
                    print(f'*******lane:{li},{veh_o}')
                print("%%%%%%%%%%%%%%")
                exit()    
            """
                
                
			if data_file.real_time_spawning_flag:
				#############lane independent ID ###############
				veh_id_var = 0
				#############lane dependent ID ###############
				dep_veh_id  = [(100*lane) for lane in data_file.lanes] 

				next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]  
				for lane in data_file.lanes:
					if not (arr_rate_array[lane] == 0):   #if only the arrival rate not zero we add poisson spawning else the previous values not changed
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)
        
			assert wr_coun ==0, 'replay buffer not updated'
			horizon = {}
			override_lane =  {}
			### override_lane, key - lane, value - vehicle ID of bad agent   ###
			### lane signal and time of previous change  ###
			### start of simulation ###
			while (time_track < (120)):     #(data_file.max_sim_time)+1):  
				curr_time = time.time()
				if (not data_file.real_time_spawning_flag):
					if (functions.get_num_of_objects(sim_obj.done_veh) >= total_veh_in_simulation):
						break
				if data_file.rl_algo_opt == "DDPG" and learning_flag and (agent.buffer.buffer_counter > 0) and ((time_track % 1) == 0):
					#learn_init_time = time.time()
					agent.buffer.learn()
					#print(f"[IN MAIN.py]: learning time: {round(time.time() - learn_init_time, 3)}")

					#tar_update_init_time = time.time()
					agent.update_target(agent.target_model.variables, agent.model.variables, agent.tau_)
					#print(f"[tar_update MAIN.py]: update time: {round(time.time() - learn_init_time, 3)}")

				#print(f'ver:{version},  arr :{arr_rate_}, sim :{sim}, {train_iter}')
				if learning_flag and (time_track % 100) == 0:
					directory = f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/sim_data/trained_weights/model_weights_itr_{int(time_track)}"  
					os.makedirs(directory, exist_ok=True)
					filename = f"model_weights_itr_500.weights.h5"
					file_path = os.path.join(directory, filename)
					agent.model.save_weights(file_path)

				
				if not data_file.real_time_spawning_flag: 
					for lane in data_file.lanes: 
						if len(sim_obj.unspawned_veh[lane])>0:   
								v_iter = 0
								veh_unsp  =  sim_obj.unspawned_veh[lane][v_iter]
								if time_track == veh_unsp.sp_t: 
									#while v_iter<len(sim_obj.unspawned_veh[lane]):       
									# print(f'time_tack witjout round"{(time_track)},{(veh_unsp.sp_t)}')
									# print(f'time_tack"{round(time_track,1)},{round(veh_unsp.sp_t,1)}')
									sim_obj.spawned_veh[lane].append(veh_unsp)
									sim_obj.unspawned_veh[lane].popleft()
									#print(f'spawning: ****************{sim_obj.spawned_veh[lane][-1].id},{time_track},{sim_obj.spawned_veh[lane][-1].sp_t}')
									
									
				if (data_file.real_time_spawning_flag) :                    
					for lane in data_file.lanes: 
						if (data_file.real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (not (arr_rate_array[lane] == 0)):
							#next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)  #NEW SPAWN TIME FOR NEXT ROBOT
							next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)), 1)
							new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
							
							############### veh.ID ####################
							## for lane dependent ID
							#new_veh.id = copy.deepcopy(dep_veh_id[lane])
							#dep_veh_id[lane] += lane  #making the lane dependent ID
							# continuous ID
							new_veh.id = copy.deepcopy(veh_id_var)
							veh_id_var += 1
							################ veh.ID #######################
							new_veh.sp_t = copy.deepcopy(time_track)  # time at whic robot veh created 
							new_veh.sig_stat_t = lane_sig_stat[lane][1] 
							new_veh.sig_stat = 0
							sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))
							#################### end of spawning  ################### 	
											
				
						##### rear end safety validation  ##### 	
						n = len(sim_obj.unspawned_veh[lane])
						########### CHECKING MODULE ###########
						functions.seq_checker(sim_obj.unspawned_veh)
						functions.seq_checker(sim_obj.spawned_veh)
						########### CHECKING MODULE ###########
						v_itr = 0
						if n>1 and data_file.real_time_spawning_flag:print(a)

						while v_itr<n:
							v = sim_obj.unspawned_veh[lane][v_itr]
							pre_v = None
							if len(sim_obj.spawned_veh[lane]) > 0:       
								pre_v = sim_obj.spawned_veh[lane][-1]
							if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)): # CHECKS REAR END SAFETY
								v.sp_t = round(time_track,1)	# step at whcih the particular vehicle spawned or entred ROI
								sim_obj.unspawned_veh[lane].popleft()
								sim_obj.spawned_veh[lane].append(v)
								n = len(sim_obj.unspawned_veh[lane])
							else:
								break	
					
				########### CHECKING MODULE ###########
				functions.seq_checker(sim_obj.unspawned_veh)
				functions.seq_checker(sim_obj.spawned_veh)
				for _ in data_file.lanes: assert (sorted(list(set([v.id for v in (sim_obj.spawned_veh[_])]))) ==  [v.id for  v in (sim_obj.spawned_veh[_])]), f'ID:{[v.id for  v in (sim_obj.spawned_veh[_])]}, lane:{_},1:{sort(list(set([v.id for v in (sim_obj.spawned_veh[_])])))}, 2:{[[v.id,iter] for  iter, v in enumerate(sim_obj.spawned_veh[_])]}'   #### check all id in a lane in ascending order
				########### CHECKING MODULE ###########
				##### END - spawning & rear end safety validation ##### 
				if functions.get_num_of_objects(sim_obj.spawned_veh) > 0 :
					spawned_veh_copy = copy.deepcopy(sim_obj.spawned_veh)
					### RL decision ###
					if learning_flag:
						prev_state = curr_state
						prev_rew = curr_rew
						prev_obs = curr_obs
                    
                        ################## PLOT ################################
						# reward_list_per_episode.append(curr_rew)
						if (not (curr_rew == None)):
							reward_list_per_episode.append(curr_rew)
							time_track_list.append(time_track)
							cumulative_reward=cumulative_reward+curr_rew
                        ################## PLOT ################################

					if time_track >= 94:
						if time_track ==94:
							f = open("debug.txt", "w+")
							f.close()
						f = open("debug.txt", "a")
						f.write(f'\n\ntime:{time_track}, number of robots:{functions.get_num_of_objects(sim_obj.spawned_veh)}, robot in lane:{[_.id for __ in data_file.lanes  for _ in sim_obj.spawned_veh[__]]}\n')
						f.close()

						#learn_rl_time = time.time()
					sim_obj.spawned_veh, alpha, dict_alpha, signal, dict_sig_copy, state_t, obs_t, action_t, explore_flag, phase_id = contr_sig_rl.get_alpha_sig(time_track, sim_obj.spawned_veh, agent, algo_option, learning_flag)
					#print(f"[RL query]: learning time: {round(time.time() - learn_rl_time, 3)}")
					phase_explore.append(phase_id)
					if learning_flag:
						curr_state = copy.deepcopy(state_t)
						prev_act = copy.deepcopy(curr_act)
						curr_act = copy.deepcopy(action_t)
						curr_obs = copy.deepcopy(obs_t)
						curr_rew = 0
					### END - RL decision ###
						########### Penalty ###########
						if functions.get_num_of_objects(sim_obj.spawned_veh)>0:
							#print(f'pos:{v.p_traj},{type(v.p_traj)}')
							if len(v.p_traj)==1: curr_rew += 25*v.priority
							curr_rew -= sum([v.priority*(-v.p_traj[-1] + v.intsize)*(v.sp_t - time_track) for _ in data_file.lanes for v in sim_obj.spawned_veh[_]])
						else: curr_rew = 0	
						########### Penalty ###########

					for _ in dict_sig_copy: dict_sig[_] = copy.deepcopy(dict_sig_copy[_])

					########### CHECKING MODULE ###########
					functions.seq_checker(sim_obj.spawned_veh)
					if data_file.output =='Signal': 
						assert len(dict_sig) == data_file.num_lanes*4 
					elif data_file.output =='Phase': len(dict_sig) == data_file.num_phases
						#assert len(dict_sig) == data_file.num_phases,f'dict:{len(dict_sig)},phase:{data_file.num_phases}'

					########### CHECKING MODULE ###########
					assert all ([v.id == spawned_veh_copy[_][iter].id for _ in data_file.lanes for iter, v in enumerate(sim_obj.spawned_veh[_])]),f'spawned set passed from RL WITH ERROR'
					assert all([ _!= None for _ in signal ]), f'signal value is none:{signal}, time:{time_track}'
					assert all([ _!= None for _ in alpha ]), f'alpha value is none:{alpha}, time:{time_track}'

					###### override for a fixed time period T  #############
					###### T time period for bad agent to cross intersection #####n
					#print(f' override_lane:{override_lane}')
					for _ in override_lane: 
						dict_sig[_]='R'
						for lane in data_file.incompdict[_]:
							dict_sig[lane]='R'
					######### override log dictionary ##############

					################## update the veh with signal values #################
				for lane in data_file.lanes:
					n = len(sim_obj.spawned_veh[lane])
					for iter in range(n):
						sim_obj.spawned_veh[lane][iter].global_sig_val[time_track] = signal
						sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = override_lane # list(override_lane.keys())
					for veh_iter in sim_obj.done_veh[lane]:
						veh_iter.global_sig_val[time_track] = signal
						veh_iter.ovr_stat[time_track] = override_lane # list(override_lane.keys())

				###### override for a fixed time period T  #############
				#############CHECKS##############
					cg = 0
					cr = 0
					for _ in dict_sig: 
						if dict_sig[_]=='G': cg+=1
						elif dict_sig[_]=='R': cr+=1
					#print(f'{dict_sig},{cr},{cg}')
					if (cr ==0 or cg ==0) == True: assert len(override_lane)!=0 or functions.get_num_of_objects(sim_obj.spawned_veh)== 0,f'dict_sig:{dict_sig}, over:{override_lane},num:{functions.get_num_of_objects(sim_obj.spawned_veh)}'
				#############CHECKS##############
				########### control variable estimation ##############
				if functions.get_num_of_objects(sim_obj.spawned_veh) > 0: 
					for lane in data_file.lanes:
						green_zone = -1*data_file.vm[lane]*data_file.dt + (data_file.vm[lane]**2)/(2*(max(data_file.u_min[lane],-(data_file.vm[lane]/data_file.dt))))  
						if len(data_file.incompdict[lane])>0:
							override_veh = []
							if dict_sig[lane]=='G':
								n = len(sim_obj.spawned_veh[lane])
								for iter in range(n):
									pre_v = None
									success = None
									v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
									if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
									sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.green_map(v, pre_v, time_track)#, algo_option, learning_flag, 1, sim, train_iter, train_sim)
									if not learning_flag:functions.storedata(v, train_sim, sim, train_iter,version) 							
							elif dict_sig[lane]=='R':
								n = len(sim_obj.spawned_veh[lane])
								for iter in range(n):
									v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
									success = None
									pre_v = None
									if n >1 and iter >0 : pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
									sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.red_map(v, pre_v,time_track)# algo_option, learning_flag, 1, sim, train_iter, train_sim)
									if success == False: ##OVERRIDE
										sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.green_map(v, pre_v,time_track, over='override')# algo_option, learning_flag, 1, sim, train_iter, train_sim)
										override_veh.append(v.id)
									if not learning_flag:functions.storedata(v, train_sim, sim, train_iter,version) 
								if len(override_veh)>0: override_lane[lane] = override_veh
								#if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 

					assert len(set(override_lane.keys())) == len(override_lane), f'duplicates present in over_ride lane: {override_lane}'
					#assert all([list(override_lane.keys())[_] not in override_lane  for _ in range(len(override_lane))]),f'duplicates present :{override_lane}'

					############# all incompatible lane_override trajectories #############
	 				######################################################################
	 				##### Note ######### : bad agent shouldn't get this trajectory ########
	  				#######################################################################
					for _ in override_lane:
						dict_sig[_]='R'
						for lane in data_file.incompdict[_]:
							dict_sig[lane]='R'   #### override
							n = len(sim_obj.spawned_veh[lane])
							for iter in range(n):
								v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
								success = None
								pre_v = None
								if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
								sim_obj.spawned_veh[lane][iter], success= safe_set_map_bisect_backup1.red_map(v, pre_v,time_track) 
								assert success != False
								if not learning_flag:functions.storedata(v, train_sim, sim, train_iter,version) 
					########### all incompatible control variable estimation ##############
					assert all ([veh.p_traj[-1]<= sim_obj.spawned_veh[lane][0].p_traj[-1] for lane in data_file.lanes for veh in sim_obj.spawned_veh[lane]]),f'robot skippped---position exceeded,'
					############# update veh and REF dict with signal and time ##################
					for lane in data_file.lanes:
						if len(data_file.incompdict[lane])>0:
							if lane_sig_stat[lane][0] !=  dict_sig[lane]:
								lane_sig_stat[lane][0] =  dict_sig[lane]
								lane_sig_stat[lane][1] =  time_track
							n = len(sim_obj.spawned_veh[lane])
							if dict_sig[lane]=='G': sig = 1 
							elif dict_sig[lane]=='R': sig = 0
							for iter in range(n):
								sim_obj.spawned_veh[lane][iter].sig_stat = sig
								sim_obj.spawned_veh[lane][iter].sig_stat_t = lane_sig_stat[lane][1] 
								sim_obj.spawned_veh[lane][iter].global_sig[time_track] = dict_sig
								sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = override_lane 
							for veh_iter in sim_obj.done_veh[lane]:
								veh_iter.sig_stat = sig
								veh_iter.sig_stat_t = lane_sig_stat[lane][1] 
								veh_iter.global_sig[time_track] = dict_sig
								veh_iter.ovr_stat[time_track] = override_lane 
								

					############# update veh and REF dict  with signal and time  ##################
				### update current time###
					if not learning_flag:
						signal_ts[time_track] = copy.deepcopy(dict_sig)
						override_ts[time_track] = copy.deepcopy(override_lane)
    
				time_track = round((time_track + data_file.dt), 1)
				if learning_flag: 
					print(f"arr_rate: {arr_rate_}, rep: {rep_sim}", "current time:", time_track, "sim:", sim, "train_iter:", train_iter,"size_buff",{(sys.getsizeof(agent.buffer.state_buffer)+ sys.getsizeof(agent.buffer.reward_buffer) + sys.getsizeof(agent.buffer.action_buffer) + sys.getsizeof(agent.buffer.next_state_buffer))/(1024*1024)},"......", end="\r")
				else: 
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
    
				### intersection ###
				for lane in data_file.lanes:
					if len(sim_obj.spawned_veh[lane])>0:
						if  lane in override_lane: c = len(override_lane[lane])
						else: c = 0
						for veh_iter in sim_obj.spawned_veh[lane]:
							if veh_iter.pos_status == 'inter':
								temp_ind = functions.find_index(veh_iter, time_track)
								if 0< veh_iter.p_traj[-1]<= veh_iter.intsize: 
									veh_iter.u_traj.append(veh_iter.u_max)
									X = []
									X.append(veh_iter.p_traj[temp_ind])
									X.append(veh_iter.v_traj[temp_ind])
									x_next = functions.compute_state(X, veh_iter.u_traj[temp_ind],round( data_file.dt,1),veh_iter.v_max,0)
									assert x_next[1]<=veh_iter.v_max and  x_next[1]>=veh_iter.v_min
									veh_iter.p_traj.append(copy.deepcopy(x_next[0]))
									veh_iter.v_traj.append(copy.deepcopy(x_next[1]))
									veh_iter.t_ser.append(round((time_track + (data_file.dt)), 1))
		
									if lane in override_lane:
										if veh_iter.id not in override_lane[lane]: veh_iter.sig[time_track] = 'green'
										elif veh_iter.id  in override_lane[lane]: veh_iter.sig[time_track] = 'orange'
									else : veh_iter.sig[time_track] = 'green'
									if not learning_flag:functions.storedata(veh_iter, train_sim, sim, train_iter, version) 

							######################## try out: -1 or temp_ind   ######################
							########################################################################
	
						for veh_ in sim_obj.done_veh[lane]:
							if veh_.p_traj[-1] > veh_.intsize and veh_.p_traj[-1] <(veh_.intsize +  veh_.length - veh_.int_start/2):
								# if veh_.id==1:print(f'\n**********done ID:{veh_.id}, end:{veh_.intsize +  veh_.length - veh_.int_start/2},curr{veh_.p_traj[-1]}')
								veh_.u_traj.append(veh_.u_max)
								X = []
								X.append(veh_.p_traj[-1])
								X.append(veh_.v_traj[-1])
								x_next = functions.compute_state(X, veh_.u_traj[-1],round( data_file.dt,1),veh_.v_max,0)
								veh_.p_traj.append(copy.deepcopy(x_next[0]))
								veh_.v_traj.append(copy.deepcopy(x_next[1]))
								veh_.t_ser.append(round((time_track + (data_file.dt)), 1))
								veh_.sig[round((time_track + (data_file.dt)), 1)]='lime'
								veh_.pos_status = 'done'
								if not learning_flag: functions.storedata(veh_, train_sim, sim, train_iter,version) 
							

						ind  = 0
						veh_num = len(sim_obj.spawned_veh[lane])
						while ind < veh_num:
							veh_iter = sim_obj.spawned_veh[lane][ind]
							if veh_iter.p_traj[-1] > veh_iter.intsize and veh_iter.pos_status == 'inter':     ###############try out:     veh_iter.length:     
								veh_iter.sig[round((time_track + (data_file.dt)), 1)]='lime'
								veh_iter.ovr_stat[round((time_track + (data_file.dt)), 1)]=override_lane
								veh_iter.global_sig[round((time_track + (data_file.dt)), 1)] = dict_sig 
								sim_obj.done_veh[lane].append(veh_iter)
								sim_obj.spawned_veh[lane].popleft()
								veh_num =  len(sim_obj.spawned_veh[lane])
								if lane in override_lane and veh_iter.id in override_lane[lane]: c -= 1
							else: break

						for veh_iter in sim_obj.spawned_veh[lane]:
							if veh_iter.p_traj[-1] > 0 and veh_iter.pos_status == 'lane':     ###############try out:     veh_iter.length:     
								veh_iter.pos_status = 'inter'
								if learning_flag : curr_rew += 25*veh_iter.priority	#### reward entering intersection 
						if c==0 and lane in override_lane: del override_lane[lane]

						########################try out					
						#########assert [len() for iter in   ]
						############  check a,v , p len of  vectors are same as the timetrack-spawn time
				
				for _ in data_file.lanes:
					for veh in sim_obj.spawned_veh[_]:
						if (veh.p_traj[-1]<= 0 and veh.pos_status == 'lane') or (veh.p_traj[-1]> 0 and veh.pos_status == 'inter'):pass
						else: 
							print(f'veh failed: lane:{_},{veh.p_traj},{veh.t_ser},status:{veh.pos_status}')
							exit()


				### storing data in buffer
				if (algo_option == "rl_modified_ddswa") and (learning_flag):
					if (not (prev_rew == None)) and (not (len(prev_state) == 0)):
						agent.buffer.remember((prev_state[0], prev_act, prev_rew, curr_state[0]))
						qwe = {}
						qwe["state_buffer"] = agent.buffer.state_buffer
						#qwe["observe_buffer"] = agent.buffer.observe_buffer
						qwe["action_buffer"] = agent.buffer.action_buffer
						qwe["reward_buffer"] = agent.buffer.reward_buffer
						qwe["next_state_buffer"] = agent.buffer.next_state_buffer
						#qwe["next_observe_buffer"] = agent.buffer.next_observe_buffer
						if rep_sim%10==0 and time_track == data_file.max_sim_time :
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
							#print(f'insdie te db file')
							###################################
                        #Stoing Signal and Override Globally
				if not learning_flag and rep_sim%1==0 and time_track == data_file.max_sim_time:
					functions.store_signal_override(signal_ts, override_ts, version, _arr_rate_, train_sim, sim, train_iter)
       
				### removed vehicles which have crossed the region of interest ###
				#print(f"[END MAIN.PY]: one timestep time: {round(time.time() - curr_time, 3)}")
				##############################
			### end of simulation ###
			return_list.append(cumulative_reward)
			episode_list.append(rep_sim)
			if (rep_sim)%20==0:
				#print(f'len,{len(reward_list_per_episode)}')
				plt.plot(time_track_list,reward_list_per_episode)
				plt.xlabel('Track_time')
				plt.ylabel('Rewards')
				plt.title(f'rewards for episode: {rep_sim}')
				plt.savefig(f"../data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{train_sim}/Rewards for episode:{rep_sim}",format='png')
				plt.clf()
		
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
   
#### streamline ####

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
				version = float(sys.argv[5]) 
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
					version = float(sys.argv[3]) 
					for _train_iter in _train_iter_list:
						for _sim_num in range(1,2):  #11
							############################### edited ################ # each policy run at speicific arrival rate for 10 times to increase the samples.
							for _train_sim in list(range(1,2)):   ##### edited############### 11 $#############################
								for _arr_rate_ in arr_rates_to_sim:
									arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
									# file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}"								
									file_path = f"../d_version/version_{int(version)}/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"
									try:
										with open(f"{file_path}") as f:
											f.close()
									except:
										args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_,version, _train_sim])
					pool = Pool(18,initializer=init_pool_processes)
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








