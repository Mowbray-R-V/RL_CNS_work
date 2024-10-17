import numpy as np
import itertools
import data_file
import functions
import get_states
import tensorflow as tf
import copy
import math
import random
from immutabledict import immutabledict
import pprintpp
#import mylib


###!!!!!!!!! CAUTION !!!!!!!!!###
## DO NOT CHANGE THE ORDER OF APPENDING THE piS AND diS IN ANY WAY ##

#id_list =[]

def truncate_float_precision(value, precision):
    # Convert the value to float64
    float64_value = np.float64(value)

    # Truncate the precision to the desired level
    truncated_value = np.round(float64_value, precision)

    # Convert back to a regular float
    result = float(truncated_value)

    return result


def sigmoid(x):

	precision = 8
	truncated_value = truncate_float_precision(x, precision)
#print(f"Original Value: {original_value}")
#print(f"Truncated Value: {truncated_value}")
#	x = np.float128(x)
	return 1/(1+np.exp(-truncated_value ))

def get_state_big(all_veh_set,id_list):

	state = []
	feature = []

	f_agent = []
	for lane in data_file.lanes:
		for veh in all_veh_set[lane]:
			id_list.append(veh.id)
			#print(id_list)
			if len(veh.t_ser) < 1:
				veh.t_ser = [veh.sp_t]
				veh.p_traj = [veh.p0]
				veh.v_traj = [veh.v0]

			for _ in (range(8)):
				if data_file.lane_map[lane] == _: 
					state.append(veh.lane)
					state.append(veh.priority)
					state.append(veh.p_traj[-1])
					state.append(veh.v_traj[-1])
					state.append(veh.t_ser[-1] - veh.sp_t)
					state.append(veh.t_ser[-1] - veh.sig_stat_t)
					state.append(veh.sig_stat)
   
				elif _ in data_file.incompdict[lane] : state.extend([-1]*data_file.num_features)  #### FEATURE USED
				else: state.extend([1]*data_file.num_features)

	state = np.array(state, dtype=float)
	if functions.get_num_of_objects(all_veh_set)< data_file.num_veh: 
		state = np.pad(state, (0, ((-functions.get_num_of_objects(all_veh_set)+ data_file.num_veh)*data_file.num_features*data_file.num_lanes*4)), mode='constant', constant_values=0.0)
		#state = state.tolist()

	assert len(state) == (data_file.num_features*data_file.num_veh*data_file.num_lanes*4) ,f' feature state in wrong format, state:{(state)}, len:{len(state)} feature:{(data_file.num_features*data_file.num_veh*data_file.num_lanes*4)} '
	assert all([ not(math.isnan(iter)) for iter in state]),f'input values-bad : state: {state}'

			
	return np.asarray(state).reshape(1, len(state)), id_list




# def get_state_medium(all_veh_set, id_list):   

#     state = {i:[] for i in data_file.lanes}  # Initialize a dictionary for lanes
#     # phase_combinations = [
#     #     (1, 2), (1, 5), (1, 7), (1, 10),
#     #     (2, 8), (4, 10), (4, 5), (4, 8),
#     #     (5, 11), (7, 8), (7,11), (10, 11)
#     # ]

#     f_agent = []
#     for lane in data_file.lanes:
#         for veh in all_veh_set[lane]:

#             id_list.append(veh.id)
#             if len(veh.t_ser) < 1:
#                 veh.t_ser = [veh.sp_t]
#                 veh.p_traj = [veh.p0]
#                 veh.v_traj = [veh.v0]

#             lane_state = []
# 			lane_state.append(veh.priority)
# 			lane_state.append(veh.p_traj[-1])
# 			lane_state.append(veh.v_traj[-1])      #current time   needed ????
# 			lane_state.append(veh.t_ser[-1] - veh.sp_t)
# 			lane_state.append(veh.t_ser[-1] - veh.sig_stat_sym)
# 			lane_state.append(veh.prev_phase_encode)    #12 dim one-hot encoding prev phase
# 			lane_state.append(veh.goal_dist)    # dist to entry of inter
# 			lane_state.append(veh.phase_encode)    # 12 dim phase encoding
# 			lane_state.append(veh.over_flag)    # 1 or 0 for that lane
# 			lane_state.append(veh.umin_range)

#             # lane_state.append(veh.priority)
#             # lane_state.append(veh.intsize-veh.p_traj[-1])
#             # lane_state.append(veh.v_traj[-1])
#             # lane_state.append(veh.t_ser[-1] - veh.sp_t)
#             # lane_state.append(veh.t_ser[-1] - veh.sig_stat_sym)
#             # lane_state.append(veh.sig_stat)

#             # Add phase information
#             phase_info = [0] * 12
#             for idx, (l1, l2) in enumerate(phase_combinations):
#                 if veh.lane == l1 or veh.lane == l2:
#                     phase_info[idx] = 1

#             lane_state.extend(phase_info)

#             state[veh.lane].extend(lane_state)
# 	# assert len(state)==8, f'there are less than 8 lanes in the state dict, {len(state)}'
#     # print('state',state)
#     return [state], id_list


def phase_state_transform(state, id_lane):

	phase_state_list = []
	# phase_state_dict = {i:[] for i in data_file.lanes if len(data_file.incompdict[i])>0}  
	phase_state_dict = {i:[] for i in range(12)}  

	for phase_iter in range(12):
		counter = 0
		pad = 0
		# if isinstance(phase_state_dict.get(phase_iter, False), list):
		for lane_info in data_file.lanes:
			if len(data_file.incompdict[lane_info])>0:
				if data_file.phase_ref[phase_iter][lane_info] == 'G':
					# print(f'lane:{lane_info}, phase:{phase_iter}, phase_dict:{phase_state_dict}, state:{state[lane_info]}')
					phase_state_dict[phase_iter].extend(state[lane_info])

					if len(state[lane_info])< data_file.max_robot_lane*data_file.state_size:
						assert isinstance(len(state[lane_info]), int) and not(len(state[lane_info]) % data_file.state_size)
						num_robot = int(len(state[lane_info])/data_file.state_size)
						assert num_robot < data_file.max_robot_lane  and len(id_lane[lane_info]) == num_robot    ### redundant check
						pad +=  (data_file.max_robot_lane - num_robot)*data_file.state_size
					elif len(state[lane_info])> data_file.max_robot_lane*data_file.state_size: assert False   #redundant check
					elif len(state[lane_info]) == data_file.max_robot_lane*data_file.state_size: pass   #redundant check
					else : assert False
					counter+=1
				elif data_file.phase_ref[phase_iter][lane_info] == 'R': pass    #redundant check
				else : assert False
		assert counter ==2,f'count _value: {counter}'
		phase_state_dict[phase_iter] = np.array(phase_state_dict[phase_iter], dtype = float)
		phase_state_dict[phase_iter] = np.pad(phase_state_dict[phase_iter], (0, pad), mode='constant', constant_values=0.0)
		assert np.shape(phase_state_dict[phase_iter])[0] == data_file.state_size*(data_file.max_robot_lane*2),f'LHS:{np.shape(phase_state_dict[phase_iter])[0]}, RHS:{data_file.state_size*data_file.max_robot_lane*2}'
		phase_state_list.extend(phase_state_dict[phase_iter])
		# print(f'\n\n successful: phase_iter:{phase_iter}')
	 	
	phase_state_np = np.array(phase_state_list, dtype = float)
	assert  np.shape(phase_state_np)[0] == data_file.state_size*(2*data_file.max_robot_lane)*12,f'LHS:{np.shape(phase_state_np)[0]}, RHS:{data_file.state_size*(2*data_file.max_robot_lane)*12}'

	print(f'state!!!!!!!!!!!!!!{type(phase_state_list)},{type(phase_state_list[0])},{len(phase_state_list)}')
	print(f'\n state!!!!!!!!!!!!!!{type(phase_state_np)},{type(phase_state_np[0])},{np.shape(phase_state_np)}')

	exit()

	return phase_state_np


def get_state_medium(all_veh_set):   


	state = {i:[] for i in data_file.lanes if len(data_file.incompdict[i])>0}  # Initialize a dictionary for lanes
	id_lane = {i:[] for i in data_file.lanes if len(data_file.incompdict[i])>0}  # defined later after diviosn number check
	feature = []
	feature = []
	f_agent = []
	id_list = []

	for lane in data_file.lanes:
		lane_state = []
		for veh in all_veh_set[lane]:
			id_lane[lane].append(veh.id)
			id_list.append(veh.id)
			#print(id_list)
			#print(f"***********:{veh.t_ser},{len(veh.t_ser)}")
			if len(veh.t_ser) < 1:
				veh.t_ser = [veh.sp_t]
				veh.p_traj = [veh.p0]
				veh.v_traj = [veh.v0]
   
			lane_state.append(veh.priority)
			lane_state.append(veh.p_traj[-1])
			lane_state.append(veh.v_traj[-1])      
			lane_state.append(veh.t_ser[-1] - veh.sp_t)
			# if veh.sig_stat_sym == None: print()

			# lane_state.append(veh.t_ser[-1] - veh.sig_stat_sym)
			# print(f'inside get_state:ID: {veh.id}, prev_encode:{veh.prev_phase_encode}')
			# exit()
			lane_state.extend(veh.prev_phase_encode)    #12 dim one-hot encoding prev phase   # noen
			lane_state.append(veh.goal_dist)    # dist to entry of inter
			lane_state.append(veh.over_flag)    # 1 or 0 for that lane   #none
			# lane_state.append(veh.umin_range)
			lane_state.extend(veh.phase_encode)    # 12 dim phase encoding
			state[lane].extend(lane_state)

   
		##### lane compatibility and in compatibiliy informations 
			# state.append(veh.t_ser[-1] - veh.sig_stat_t)
			# state.append(veh.sig_stat)
   
			# state.append(veh.lane)
			# for _ in data_file.lanes :
			# 	if len(data_file.incompdict[_])>0:
			# 		if _ == lane : state.append(0)
			# 		elif _ in data_file.incompdict[lane] : state.append(-1)
			# 		elif _ not in data_file.incompdict[lane] : state.append(1)
		##### lane compatibility and in compatibiliy informations 

	for _ in  data_file.lanes:
		if len(data_file.incompdict[_])>0:
			assert len(state[_]) == data_file.state_size*(len(id_lane[_]))


	# print(f'state_after_update: {state}')
	phase_state = copy.deepcopy(phase_state_transform(state, id_lane))
	# print(f'\n state!!!!!!!!!!!!!!{type(phase_state)},{type(phase_state[0])},{np.shape(phase_state)}')
   
	assert all([ not(math.isnan(iter)) for iter in phase_state]),f' input values-bad: state : {state}'

	return phase_state, id_list   #return np.asarray(state).reshape(1, len(state)), id_list



def signals(alpha,id_list,signal):

	dict_alpha= {}
	dict_sig= {}
	map_lane = {}
	iter = 0

	for lane in data_file.lanes:
		if len(data_file.incompdict[lane])>0: 
			map_lane[iter] = lane    #### map order as per the incompdict dictionary
			iter += 1

	assert len(id_list) == len(alpha)
	assert len(map_lane) == data_file.num_lanes*4

	for i in range(len(id_list)): dict_alpha[id_list[i]] = alpha[i]	

	######## updates the agents with signal and control varaible  ##########

	sig_index = signal.index(max(signal))    # lanes  1,2,4,5,7,8,10, 11
	dict_sig[map_lane[sig_index]] = 'G'

	#print(type(map_lane), type(dict_alpha))
	#exit()

	#for _ in data_file.incompdict[map_lane[sig_index]]: dict_sig[map_lane[sig_index]] = 'R'
	for _ in data_file.incompdict[map_lane[sig_index]]: dict_sig[_] = 'R'  ###incompdict lane made red

	for _ in list(dict_sig.keys()):	signal[ list(map_lane.values( )).index(_)]  = 0   ### make signal values of lanes in dict_sig zer0

	#print(f'signal :{signal}, dict:{dict_sig}, other_lane :{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig] },other_sig_lane :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }')
	sig_index_1 = signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]))   

	assert map_lane[sig_index_1] not in  dict_sig, f'error in assigning SIGNAL, signal:{signal}, sig_index_1: {sig_index_1}, dict_sig:{dict_sig},other_lanes \
		:{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig]} \n , oth_lane_sig :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }, \
				fgdfg{(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }, gkk {signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }'
	#print(f'dict_sig:{dict_sig}, sig_1:{map_lane[sig_index_1]}')
	dict_sig[map_lane[sig_index_1]] = 'G'
	#print(f'dict_sig:{dict_sig}')
	for _ in data_file.incompdict[map_lane[sig_index_1]]: dict_sig[_] = 'R'
	#print(f'dict_sig:{dict_sig}')
	assert len(dict_sig) == data_file.num_lanes*4, f'error-signal, signal:{signal}, sig_index_1: {sig_index_1}, dict_sig:{dict_sig},other_lanes :{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig]} \n , oth_lane_sig :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }, fgdfg{(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }, gkk {signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }'			
	#print(f'signal :{signal}, dict:{dict_sig}')
	#exit()

	return dict_alpha, dict_sig


def phases(alpha,id_list,phase):

	dict_alpha = {}
	dict_phase = {}
 
	for i in range(len(id_list)): dict_alpha[id_list[i]] = alpha[i]	
	phase_index = phase.index(max(phase))
	dict_phase = data_file.phase_ref[phase_index]
 
	assert [_!=dict_phase[1] for _ in dict_phase.values()].count(False)!=len(dict_phase) and [_!=dict_phase[1] for _ in dict_phase.values()].count(True)!=len(dict_phase) ,f'{dict_phase}, \n original\
     value:{data_file.phase_ref}'

	return dict_alpha, dict_phase, phase_index

	
def rl_outputs(t_inst, _veh_set, rl_agent, alg_option, l_flag):

	exlporation_flag = 0
	# id_list =[]
	rl_state = []
	rl_action = []
	alpha = []
	phase = []
	signal = []
	tc_flag = []
	V_states = []
	Vp_set = _veh_set

	if data_file.rl_algo_opt == "MADDPG":
		flatten = itertools.chain.from_iterable
	
	if alg_option == "rl_modified_ddswa":
		Vp_set = _veh_set
	elif alg_option == "rl_ddswa":
		Vp_set = functions.get_set_f(_veh_set)


	
	if alg_option == "rl_modified_ddswa" and data_file.rl_algo_opt == "DDPG" :

		if data_file.obs_stat == 1:
			rl_state ,rl_obs, id_list = get_state_medium(_veh_set, id_list) 
			concatenated_array = np.hstack((rl_state, rl_obs))
			
			if l_flag:
				rl_action = rl_agent.policy(rl_state, rl_obs, rl_agent.ou_noise, num_veh=functions.get_num_of_objects(Vp_set))[0]

			else:
				rl_action = rl_agent.policy(rl_state, rl_obs, None,num_veh=functions.get_num_of_objects(Vp_set))[0]

		elif data_file.obs_stat == 0:
			rl_state, id_list = get_state_medium(_veh_set)   #, id_list)
			if l_flag:
				#print(f'state:{rl_state},{rl_state.shape},{type(rl_state)}, noise:{rl_agent.ou_noise},{type(rl_agent.ou_noise)} ')
				rl_action = rl_agent.policy(rl_state, t_inst, rl_agent.ou_noise, num_veh=functions.get_num_of_objects(Vp_set))[0]
			else:
				rl_action = rl_agent.policy(rl_state, t_inst, None, num_veh=functions.get_num_of_objects(Vp_set))[0]
		else: assert False
  
  
  
  
		exit()
		assert np.size(rl_action) == (data_file.num_veh+data_file.num_phases),f'action size wrong, len'
		for i in range(data_file.num_phases, functions.get_num_of_objects(Vp_set)+data_file.num_phases):
			alpha.append(rl_action[i])

		if data_file.output =='Signal': 
			for i in range(data_file.num_lanes*4): signal.append((rl_action[i]))
		elif data_file.output =='Phase': 
			for i in range(data_file.num_phases): 
				phase.append((rl_action[i]))
		
		#print(phase)



		if data_file.output =='Signal': 
			assert len(signal) == data_file.num_lanes*4 and len(alpha) == functions.get_num_of_objects(Vp_set) , 'error in on obtaining signal and alpha'
			dict_alpha, dict_sig = signals(alpha,id_list,signal)
		elif data_file.output =='Phase': 
			assert len(phase) == data_file.num_phases and len(alpha) == functions.get_num_of_objects(Vp_set) and len(alpha)==len(id_list) , f'error in on obtaining phase and alpha:{len(phase)},{data_file.num_phases}'
			dict_alpha, dict_phase, phase_id = phases(alpha,id_list,phase)
			dict_phase_ = copy.deepcopy(dict_phase) 

		assert all([ _>=0 and _<=1  for _ in alpha]),f'alpha value not in range :{alpha}'
  
		assert [_!=dict_phase[1] for _ in dict_phase.values()].count(False)!=len(dict_phase) and [_!=dict_phase[1] for _ in dict_phase.values()].count(True)!=len(dict_phase) ,f'{dict_phase}'
		
		#action overwrite
		#print(range((data_file.num_phases+functions.get_num_of_objects(Vp_set)), data_file.num_veh+data_file.num_phases ))


		for _ in range((data_file.num_phases+functions.get_num_of_objects(Vp_set)), data_file.num_veh+data_file.num_phases ): 
			rl_action[_] = -190.392
		
		#print(rl_action.tolist(), [rl_action.tolist()].count(-190.392), np.count_nonzero(rl_action == -190.392) )
		assert rl_action.tolist().count(-190.392) == data_file.num_veh - functions.get_num_of_objects(Vp_set),f'action overwrite done wrong,f:{rl_action.tolist().count(-190.392)},{data_file.num_veh - functions.get_num_of_objects(Vp_set)}'
		
		#action padding
		#rl_action = np.pad(rl_action, (0, ((-functions.get_num_of_objects(_veh_set)+ data_file.num_veh))), mode='constant', constant_values= -1.0)

		#state padding
		#if functions.get_num_of_objects(_veh_set)< data_file.num_veh: 
			#state = np.pad(state, (0, ((-functions.get_num_of_objects(_veh_set)+ data_file.num_veh)*data_file.num_features*data_file.num_lanes*4)), mode='constant', constant_values=0.0)
		#assert len(state) == (data_file.num_features*data_file.num_veh*data_file.num_lanes*4) 
		assert len(rl_action) == (data_file.num_veh+data_file.num_phases),f'{len(rl_action)},{(data_file.num_veh+data_file.num_phases)}'
		#print(f'rl_actions_inside:{rl_action}, control ')
		#print(f'signal_values_insside:{signal}')

		
		lane_itr_var = 0
		for li in data_file.lanes:
			if len(_veh_set[li]) > 0:
				for ind in range(len(_veh_set[li])):
					_veh_set[li][ind].alpha = alpha[lane_itr_var]
					_veh_set[li][ind].alpha_dict[t_inst] = _veh_set[li][ind].alpha 
					lane_itr_var += 1
		######## updates the agents with signal and control varaible  ##########


	#return _veh_set, alpha, dict_alpha, signal, dict_sig, rl_state, rl_obs, rl_action, exlporation_flag
	if data_file.output =='Signal': return _veh_set, alpha, dict_alpha, signal, dict_sig, rl_state, np.zeros([1,data_file.num_features]) ,rl_action, exlporation_flag
	elif data_file.output =='Phase': return _veh_set, alpha, dict_alpha, phase, dict_phase_, rl_state, np.zeros([1,data_file.num_features]) ,rl_action, exlporation_flag, phase_id