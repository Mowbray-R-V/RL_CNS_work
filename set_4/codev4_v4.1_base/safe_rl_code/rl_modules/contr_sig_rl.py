import numpy as np
import itertools
from utility import config, functions
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
	for lane in config.lanes:
		for veh in all_veh_set[lane]:
			id_list.append(veh.id)
			#print(id_list)
			if len(veh.t_ser) < 1:
				veh.t_ser = [veh.sp_t]
				veh.p_traj = [veh.p0]
				veh.v_traj = [veh.v0]

			for _ in (range(8)):
				if config.lane_map[lane] == _: 
					state.append(veh.lane)
					state.append(veh.priority)
					state.append(veh.p_traj[-1])
					state.append(veh.v_traj[-1])
					state.append(veh.t_ser[-1] - veh.sp_t)
					state.append(veh.t_ser[-1] - veh.sig_stat_t)
					state.append(veh.sig_stat)
   
				elif _ in config.incompdict[lane] : state.extend([-1]*config.num_features)  #### FEATURE USED
				else: state.extend([1]*config.num_features)

	state = np.array(state, dtype=float)
	if functions.get_num_of_objects(all_veh_set)< config.num_veh: 
		state = np.pad(state, (0, ((-functions.get_num_of_objects(all_veh_set)+ config.num_veh)*config.num_features*config.num_lanes*4)), mode='constant', constant_values=0.0)
		#state = state.tolist()

	assert len(state) == (config.num_features*config.num_veh*config.num_lanes*4) ,f' feature state in wrong format, state:{(state)}, len:{len(state)} feature:{(config.num_features*config.num_veh*config.num_lanes*4)} '
	assert all([ not(math.isnan(iter)) for iter in state]),f'input values-bad : state: {state}'

			
	return np.asarray(state).reshape(1, len(state)), id_list




def get_state_medium(all_veh_set):   


	state = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  # Initialize a dictionary for lanes
	id_lane = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  # defined later after diviosn number check

	feature = []
	feature = []
	f_agent = []
	id_list = []

	for lane in config.lanes:
		lane_state = []


		############################### add states of robots only inside intersection exit distnace ################
		############################### add states of robots only inside intersection exit distnace ################
		############################### add states of robots only inside intersection exit distnace ################



		############################### add states of robots only inside intersection exit distnace ################
		############################### add states of robots only inside intersection exit distnace ################
		############################### add states of robots only inside intersection exit distnace ################




		if len(config.incompdict[lane])>0:
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
				assert len(lane_state)  == config.state_size*(len(id_lane[lane]))
			state[lane].extend(lane_state)

   
		##### lane compatibility and in compatibiliy informations 
			# state.append(veh.t_ser[-1] - veh.sig_stat_t)
			# state.append(veh.sig_stat)
   
			# state.append(veh.lane)
			# for _ in config.lanes :
			# 	if len(config.incompdict[_])>0:
			# 		if _ == lane : state.append(0)
			# 		elif _ in config.incompdict[lane] : state.append(-1)
			# 		elif _ not in config.incompdict[lane] : state.append(1)
		##### lane compatibility and in compatibiliy informations 

	for _ in  config.lanes:
		if len(config.incompdict[_])>0:
			assert len(state[_]) == config.state_size*(len(id_lane[_])),f'LHS:{len(state[_])}, RHS:{config.state_size*(len(id_lane[_]))}, lane:{_}, id_lane:{id_lane}'

	obs_req =  copy.deepcopy(functions.phase_obs_transform(state, id_lane))
	phase_state, phase_state_req = copy.deepcopy(functions.phase_state_transform(state, id_lane))

	# exit()
	# print(f'\n raw_state!!!!!!!!!!!!!!{type(state)},{(state.keys())}, state:{state}')
	# exit()
	# print(f'\n trans_state!!!!!!!!!!!!!!{type(phase_state_req)},{type(phase_state_req[0])},{np.shape(phase_state_req)}')
	# print(f'\n trans_obs!!!!!!!!!!!!!!{type(obs_req)},{type(obs_req[0])},{np.shape(obs_req)}')


	# assert all([ not(math.isnan(iter)) for iter in phase_state]),f' input values-bad: state : {phase_state}'

	# print(f'phase_state:{phase_state_req}')

	# exit()
	

	return state, phase_state_req, obs_req, id_list   #return np.asarray(state).reshape(1, len(state)), id_list



def signals(alpha,id_list,signal):

	dict_alpha= {}
	dict_sig= {}
	map_lane = {}
	iter = 0

	for lane in config.lanes:
		if len(config.incompdict[lane])>0: 
			map_lane[iter] = lane    #### map order as per the incompdict dictionary
			iter += 1

	assert len(id_list) == len(alpha)
	assert len(map_lane) == config.num_lanes*4

	for i in range(len(id_list)): dict_alpha[id_list[i]] = alpha[i]	

	######## updates the agents with signal and control varaible  ##########

	sig_index = signal.index(max(signal))    # lanes  1,2,4,5,7,8,10, 11
	dict_sig[map_lane[sig_index]] = 'G'

	#print(type(map_lane), type(dict_alpha))
	#exit()

	#for _ in config.incompdict[map_lane[sig_index]]: dict_sig[map_lane[sig_index]] = 'R'
	for _ in config.incompdict[map_lane[sig_index]]: dict_sig[_] = 'R'  ###incompdict lane made red

	for _ in list(dict_sig.keys()):	signal[ list(map_lane.values( )).index(_)]  = 0   ### make signal values of lanes in dict_sig zer0

	#print(f'signal :{signal}, dict:{dict_sig}, other_lane :{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig] },other_sig_lane :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }')
	sig_index_1 = signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]))   

	assert map_lane[sig_index_1] not in  dict_sig, f'error in assigning SIGNAL, signal:{signal}, sig_index_1: {sig_index_1}, dict_sig:{dict_sig},other_lanes \
		:{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig]} \n , oth_lane_sig :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }, \
				fgdfg{(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }, gkk {signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }'
	#print(f'dict_sig:{dict_sig}, sig_1:{map_lane[sig_index_1]}')
	dict_sig[map_lane[sig_index_1]] = 'G'
	#print(f'dict_sig:{dict_sig}')
	for _ in config.incompdict[map_lane[sig_index_1]]: dict_sig[_] = 'R'
	#print(f'dict_sig:{dict_sig}')
	assert len(dict_sig) == config.num_lanes*4, f'error-signal, signal:{signal}, sig_index_1: {sig_index_1}, dict_sig:{dict_sig},other_lanes :{[ map_lane[_] for _ in range(len(signal)) if map_lane[_] not in  dict_sig]} \n , oth_lane_sig :{[ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig] }, fgdfg{(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }, gkk {signal.index(max([ val for iter, val in enumerate(signal) if map_lane[iter] not in  dict_sig]  )) }'			
	#print(f'signal :{signal}, dict:{dict_sig}')
	#exit()

	return dict_alpha, dict_sig


def phases(alpha,id_list,phase):

	dict_alpha = {}
	dict_phase = {}
 
	if config.num_actions!=12: 
		for i in range(len(id_list)): dict_alpha[id_list[i]] = alpha[i]	

	phase_index = phase.index(max(phase))

	assert all ([phase[phase_index] >= _ for _ in phase]),"max of phase score taken wrong"    

	dict_phase = config.phase_ref[phase_index]
 
	assert [_!=dict_phase[1] for _ in dict_phase.values()].count(False)!=len(dict_phase) and [_!=dict_phase[1] for _ in dict_phase.values()].count(True)!=len(dict_phase) ,f'{dict_phase}, \n original\
     value:{config.phase_ref}'

	return dict_alpha, dict_phase, phase_index

	
def rl_outputs(t_inst, _veh_set, rl_agent, alg_option, l_flag):

	exlporation_flag = 0
	# id_list =[]
	# rl_state = []
	# rl_state_req = []
	rl_action = []
	alpha = []
	beta = []
	phase = []
	signal = []
	tc_flag = []
	V_states = []
	Vp_set = _veh_set
	action_set =  np.zeros(config.num_actions)

	if config.rl_algo_opt == "MADDPG":
		flatten = itertools.chain.from_iterable
	
	if alg_option == "rl_modified_ddswa":
		Vp_set = _veh_set
	elif alg_option == "rl_ddswa":
		Vp_set = functions.get_set_f(_veh_set)


	
	if alg_option == "rl_modified_ddswa" and config.rl_algo_opt == "DDPG" :

		if config.obs_stat == 1:
			rl_state ,  rl_state_req,  rl_obs, id_list = get_state_medium(_veh_set, id_list) 
			concatenated_array = np.hstack((rl_state, rl_obs))
			
			if l_flag:
				rl_action = rl_agent.policy(rl_state, rl_obs, rl_agent.ou_noise, num_veh=functions.get_num_of_objects(Vp_set))[0]

			else:
				rl_action = rl_agent.policy(rl_state, rl_obs, None,num_veh=functions.get_num_of_objects(Vp_set))[0]

		elif config.obs_stat == 0:
			rl_state, rl_state_phase, rl_state_control, id_list = get_state_medium(_veh_set)   #, id_list)
			if l_flag:
				# print(f'rl_state:{rl_state},rl_state_req:{np.shape(rl_state_phase)}, rl_state_control:{type(rl_state_control)} ')
				# exit()
				rl_action = rl_agent.policy(rl_state_control, rl_state_phase, t_inst, rl_agent.ou_noise, num_veh=functions.get_num_of_objects(Vp_set))[0]
			else:
				rl_action = rl_agent.policy(rl_state_control, rl_state_phase, t_inst, None, num_veh=functions.get_num_of_objects(Vp_set))[0]
		else: assert False
  
  

		# print(f'Rl_action_shape:{rl_action.shape}')
  
		####action extract#####
		assert np.size(rl_action) == config.num_actions,'action size wrong len'
		for i in range(config.num_phases, config.num_phases + config.num_veh): #for i in range(config.num_phases, functions.get_num_of_objects(Vp_set)+config.num_phases):
			alpha.append(rl_action[i])
		for i in range(config.num_phases +  config.num_veh, config.num_actions): #for i in range((config.num_phases + config.num_veh ), (functions.get_num_of_objects(Vp_set) + 2*config.num_veh )):
			beta.append(rl_action[i])


		if config.output =='Signal': 
			for i in range(config.num_lanes*4): signal.append((rl_action[i]))
		elif config.output =='Phase': 
			for i in range(config.num_phases): 
				phase.append((rl_action[i]))
		####action extract#####
		assert len(alpha) == config.num_veh,f'alpha size wrong len size:{len(alpha)}'
		assert len(beta) == config.num_veh,f'beta size wrong len size:{len(beta)}'
		assert len(phase) == config.num_phases,f'phase size wrong len size:{len(phase)}'

		if config.output =='Signal': 
			assert len(signal) == config.num_lanes*4 and len(alpha) == functions.get_num_of_objects(Vp_set) , 'error in on obtaining signal and alpha'
			dict_alpha, dict_sig = signals(alpha,id_list,signal)
		elif config.output =='Phase': 
			# assert len(phase) == config.num_phases and len(alpha) == functions.get_num_of_objects(Vp_set) and len(alpha)==len(id_list) , f'error in on obtaining phase and alpha:{len(phase)},{config.num_phases}'
			dict_alpha, dict_phase, phase_id = phases(alpha,id_list,phase)
			dict_phase_ = copy.deepcopy(dict_phase) 

		assert all([ _>=0 and _<=1 and _!= None  for _ in alpha]),f'alpha value not in range :{alpha}'
		assert all([ _>=0 and _<=1 and _!= None  for _ in beta]),f'beta value not in range :{beta}'

  
		assert [_!=dict_phase[1] for _ in dict_phase.values()].count(False)!=len(dict_phase) and [_!=dict_phase[1] for _ in dict_phase.values()].count(True)!=len(dict_phase) ,f'{dict_phase}'
		


		#####pading the action values##### 
		for act_elem in range(config.num_actions):
			# print(f'\nInside the controlsig assign: {act_elem},alpha:{len(alpha)}, beta:{len(beta)}')
			if act_elem in range(config.num_phases ):
				action_set[act_elem] = phase[act_elem] 
			elif act_elem  in range(config.num_phases,  config.num_veh + config.num_phases):
				action_set[act_elem] = alpha[act_elem - config.num_phases] 
			elif act_elem in range(config.num_phases + config.num_veh, config.num_actions):   #> config.num_phases + functions.get_num_of_objects(Vp_set) and alpha_pad_counter ==0:
				action_set[act_elem] = beta[act_elem- (config.num_phases + config.num_veh)] 
			else: assert False, {act_elem}


			# elif act_elem in range(config.num_phases + functions.get_num_of_objects(Vp_set),  config.num_phases + config.num_veh):   #> config.num_phases + functions.get_num_of_objects(Vp_set) and alpha_pad_counter ==0:
			# 	action_set[act_elem] =  alpha[act_elem - config.num_phases] #-190.392
			# elif act_elem in range(config.num_phases + config.num_veh + functions.get_num_of_objects(Vp_set),   config.num_actions):   #> config.num_phases + functions.get_num_of_objects(Vp_set) and alpha_pad_counter ==0:
			# 	action_set[act_elem] = beta[act_elem- (config.num_phases + config.num_veh)]   # -190.392

		# assert action_set.tolist().count(-190.392) == 2*(config.num_veh - functions.get_num_of_objects(Vp_set)),'action overwrite done wrong,'

		assert len(action_set) == (config.num_actions),f'{len(rl_action)},{(config.num_actions)}'
		# print(f'\n action_set np_array in control_sig len:{len(action_set)}, shape:{np.shape(action_set)}')


		lane_itr_var = 0
		for li in config.lanes:
			if len(_veh_set[li]) > 0:
				for ind in range(len(_veh_set[li])):
					_veh_set[li][ind].alpha = alpha[lane_itr_var]
					_veh_set[li][ind].alpha_dict[t_inst] = _veh_set[li][ind].alpha 
					_veh_set[li][ind].beta = beta[lane_itr_var]
					lane_itr_var += 1

	# sim_obj.spawned_veh, alpha, dict_alpha, signal, dict_sig_copy, state_t, obs_t, action_t, explore_flag, phase_id = contr_sig_rl.rl_outputs(time_track, sim_obj.spawned_veh, agent, algo_option, learning_flag)

	if config.output =='Signal': return _veh_set, alpha, dict_alpha, signal, dict_sig, rl_state, np.zeros([1,config.num_features]) ,rl_action, exlporation_flag
	elif config.output =='Phase': return _veh_set, phase, dict_phase_, rl_state, np.zeros([1,config.num_features]) ,rl_action, exlporation_flag, phase_id
	# elif config.output =='Phase': return _veh_set, alpha, dict_alpha, beta, phase, dict_phase_, rl_state, np.zeros([1,config.num_features]) ,rl_action, exlporation_flag, phase_id