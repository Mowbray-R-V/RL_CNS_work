# this file contains definitions of some general functions required
import os
import math
import numpy as np
import copy
from itertools import permutations
import pickle
from collections import deque
import csv
from utility import config
import matplotlib.pyplot as plt
import safe_set_map_bisect







def buffer_inside_check(veh_obj):

	if abs(veh_obj.p_traj[-1]) < abs((veh_obj.v_max*config.dt) + 2*(abs((veh_obj.v_max**2)/veh_obj.u_min))):
		return True
	else: return False



def phase_encoding(veh, lane_info):

	phase_encode = []
	for _ in range(len(config.lanes)):

		if config.phase_ref[_][lane_info] == 'G':
			phase_encode.append(1)
		elif config.phase_ref[_][lane_info] == 'R':
			phase_encode.append(0)
    
	assert len(phase_encode) == 12 and phase_encode.count(1) == 3
	return phase_encode


def phase_obs_transform(state, id_lane = None):

	obs = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  # Initialize a dictionary for lanes
	for lane_info in config.lanes:
		pad = 0
		if len(config.incompdict[lane_info])>0:

			obs[lane_info].extend(state[lane_info])
			if len(obs[lane_info])< config.max_robot_lane*config.state_size:
				assert isinstance(len(obs[lane_info]), int) and not(len(obs[lane_info]) % config.state_size)
				num_robot = int(len(obs[lane_info])/config.state_size)
				assert num_robot < config.max_robot_lane
				if id_lane != None: 
					assert len(id_lane[lane_info]) == num_robot,f'LHS:{ len(id_lane[lane_info])}, RHS:{num_robot}, lane:{lane_info}'    ### redundant check
				pad +=  (config.max_robot_lane - num_robot)*config.state_size
			elif len(obs[lane_info])> config.max_robot_lane*config.state_size: assert False   #redundant check
			elif len(obs[lane_info]) == config.max_robot_lane*config.state_size: pass   #redundant check
			else : assert False

			obs[lane_info] = np.array(obs[lane_info], dtype=float)
			obs[lane_info] = np.pad(obs[lane_info], (0,pad), mode='constant', constant_values=0.0)

			obs[lane_info] =  np.expand_dims(obs[lane_info], axis=0)
			# print(f'obs  before concate inside: shape {np.shape(obs[lane_info])}')
			obs[lane_info] =  np.reshape(obs[lane_info],(config.max_robot_lane*config.state_size)) #obs[lane_info] =  np.reshape(obs[lane_info],(config.max_robot_lane, config.state_size))
			
			# print(f'obs  before concate inside: shape {np.shape(obs[lane_info])}')


	obs_req =  np.concatenate(list(obs.values()), axis= 0)
	# print(f'obs after concatenate: shape {obs_req.shape}')
	assert np.shape(obs_req)[0] == config.num_veh*config.state_size    #assert np.shape(obs_req)[0] == config.num_veh and np.shape(obs_req)[1] == config.state_size
	return obs_req



def phase_state_transform(state, id_lane=None):


	# print(f'*****statetransform_______id_lane:{id_lane}, state:{state}')
	phase_state_list = []
	# phase_state_dict = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  
	phase_state_dict = {i:[] for i in range(config.num_phases)}  
	phase_state_dict_req = {i:[] for i in range(config.num_phases)}  
	# obs_dict_req =  {i:[] for i in range(config.num_phases)}  

	for phase_iter in range(config.num_phases):
		counter = 0
		pad = 0
		# if isinstance(phase_state_dict.get(phase_iter, False), list):
		for lane_info in config.lanes:
			if len(config.incompdict[lane_info])>0:
				if config.phase_ref[phase_iter][lane_info] == 'G':
					# print(f'lane:{lane_info}, phase:{phase_iter}, phase_dict:{phase_state_dict}, state:{state[lane_info]}')
					phase_state_dict[phase_iter].extend(state[lane_info])

					if len(state[lane_info])< config.max_robot_lane*config.state_size:
						assert isinstance(len(state[lane_info]), int) and not(len(state[lane_info]) % config.state_size),f'lane_state:{state[lane_info]}, bool_val{isinstance(len(state[lane_info]), int)}, nr:{len(state[lane_info])}, lane:{lane_info}   '
						num_robot = int(len(state[lane_info])/config.state_size)
						assert num_robot < config.max_robot_lane    ### redundant check
						if id_lane != None: 
							assert len(id_lane[lane_info]) == num_robot,f'LHS:{ len(id_lane[lane_info])}, RHS:{num_robot}, lane:{lane_info}'    ### redundant check
						pad +=  (config.max_robot_lane - num_robot)*config.state_size
					elif len(state[lane_info])> config.max_robot_lane*config.state_size: assert False   #redundant check
					elif len(state[lane_info]) == config.max_robot_lane*config.state_size: pass   #redundant check
					else : assert False
					counter+=1
				elif config.phase_ref[phase_iter][lane_info] == 'R': pass    #redundant check
				else : assert False
		assert counter ==2,f'count _value: {counter}'
		phase_state_dict[phase_iter] = np.array(phase_state_dict[phase_iter], dtype = float)
		phase_state_dict[phase_iter] = np.pad(phase_state_dict[phase_iter], (0, pad), mode='constant', constant_values=0.0)
		# obs_dict_req[phase_iter] =  copy.deepcopy(phase_state_dict[phase_iter])

		assert np.shape(phase_state_dict[phase_iter])[0] == config.state_size*(config.max_robot_lane*2),f'LHS:{np.shape(phase_state_dict[phase_iter])[0]}, RHS:{config.state_size*config.max_robot_lane*2}'
		phase_state_list.extend(phase_state_dict[phase_iter])
		
		# print(f'\n\n successful: phase_iter:{phase_iter}')

		#### editing code input as per phase kernel module requirement
		phase_state_dict_req[phase_iter] = np.reshape(phase_state_dict[phase_iter],((2*config.max_robot_lane), config.state_size ))
		phase_state_dict_req[phase_iter] = np.expand_dims(phase_state_dict_req[phase_iter], axis=0)
		# print(f'******** before np_array operations: shape{np.shape(phase_state_dict_req[phase_iter])}')
		# obs_dict_req[phase_iter] = np.reshape(obs_dict_req[phase_iter],((2*config.max_robot_lane), config.state_size ))

	# print(phase_state_dict_req.values())
	phase_state_req =  np.concatenate(list(phase_state_dict_req.values()), axis= 0)
	phase_state_req = np.expand_dims(phase_state_req, axis=0)   # 1,12,28,30

	# obs_req =  np.concatenate(list(obs_dict_req.values()), axis= 0)
	# print(f'******** afer np_array operations: obs: {np.shape(obs_req)}')


	# exit()
	phase_state_np = np.array(phase_state_list, dtype = float)   ## all state values as 1D array
	assert  np.shape(phase_state_np)[0] == config.state_size*(2*config.max_robot_lane)*12,f'LHS:{np.shape(phase_state_np)[0]}, RHS:{config.state_size*(2*config.max_robot_lane)*12}'

	# print(f'state!!!!!!!!!!!!!!{type(phase_state_list)},{type(phase_state_list[0])},{len(phase_state_list)}')
	# print(f'\n state!!!!!!!!!!!!!!{type(phase_state_np)},{type(phase_state_np[0])},{np.shape(phase_state_np)}')
	# exit()


	return phase_state_list, phase_state_req








# def phase_obs_transform(state, id_lane = None):

# 	# print(f"idlane:{id_lane}")
# 	# exit()

# 	obs = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  # Initialize a dictionary for lanes
# 	for lane_info in config.lanes:
# 		pad = 0
# 		if len(config.incompdict[lane_info])>0:

# 			obs[lane_info].extend(state[lane_info])
# 			if len(obs[lane_info])< config.max_robot_lane*config.state_size:
# 				assert isinstance(len(obs[lane_info]), int) and not(len(obs[lane_info]) % config.state_size)
# 				num_robot = int(len(obs[lane_info])/config.state_size)
# 				assert num_robot < config.max_robot_lane
# 				if id_lane != None: 
# 					assert len(id_lane[lane_info]) == num_robot,f'LHS:{ len(id_lane[lane_info])}, RHS:{num_robot}, lane:{lane_info}'    ### redundant check
# 				pad +=  (config.max_robot_lane - num_robot)*config.state_size
# 			elif len(obs[lane_info])> config.max_robot_lane*config.state_size: assert False   #redundant check
# 			elif len(obs[lane_info]) == config.max_robot_lane*config.state_size: pass   #redundant check
# 			else : assert False

# 			obs[lane_info] = np.array(obs[lane_info], dtype=float)
# 			obs[lane_info] = np.pad(obs[lane_info], (0,pad), mode='constant', constant_values=0.0)

# 			obs[lane_info] =  np.expand_dims(obs[lane_info], axis=0)
# 			# print(f'obs  before concate inside: shape {np.shape(obs[lane_info])}')
# 			obs[lane_info] =  np.reshape(obs[lane_info],(config.max_robot_lane*config.state_size)) #obs[lane_info] =  np.reshape(obs[lane_info],(config.max_robot_lane, config.state_size))
			
# 			# print(f'obs  before concate inside: shape {np.shape(obs[lane_info])}')


# 	obs_req =  np.concatenate(list(obs.values()), axis= 0)
# 	# print(f'obs after concatenate: shape {obs_req.shape}')
# 	assert np.shape(obs_req)[0] == config.num_veh*config.state_size    #assert np.shape(obs_req)[0] == config.num_veh and np.shape(obs_req)[1] == config.state_size
# 	return obs_req




# def phase_state_transform(state, id_lane=None):

# 	phase_state_list = []
# 	# phase_state_dict = {i:[] for i in config.lanes if len(config.incompdict[i])>0}  
# 	phase_state_dict = {i:[] for i in range(config.num_phases)}  
# 	phase_state_dict_req = {i:[] for i in range(config.num_phases)}  
# 	# obs_dict_req =  {i:[] for i in range(config.num_phases)}  

# 	for phase_iter in range(config.num_phases):
# 		counter = 0
# 		pad = 0
# 		# if isinstance(phase_state_dict.get(phase_iter, False), list):
# 		for lane_info in config.lanes:
# 			if len(config.incompdict[lane_info])>0:
# 				if config.phase_ref[phase_iter][lane_info] == 'G':
# 					# print(f'lane:{lane_info}, phase:{phase_iter}, phase_dict:{phase_state_dict}, state:{state[lane_info]}')
# 					phase_state_dict[phase_iter].extend(state[lane_info])

# 					if len(state[lane_info])< config.max_robot_lane*config.state_size:
# 						assert isinstance(len(state[lane_info]), int) and not(len(state[lane_info]) % config.state_size),f'lane_state:{state[lane_info]}, bool_val{isinstance(len(state[lane_info]), int)}, nr:{len(state[lane_info])}, lane:{lane_info}   '
# 						num_robot = int(len(state[lane_info])/config.state_size)
# 						assert num_robot < config.max_robot_lane    ### redundant check
# 						if id_lane != None: 
# 							assert len(id_lane[lane_info]) == num_robot,f'LHS:{ len(id_lane[lane_info])}, RHS:{num_robot}, lane:{lane_info}'    ### redundant check
# 						pad +=  (config.max_robot_lane - num_robot)*config.state_size
# 					elif len(state[lane_info])> config.max_robot_lane*config.state_size: assert False   #redundant check
# 					elif len(state[lane_info]) == config.max_robot_lane*config.state_size: pass   #redundant check
# 					else : assert False
# 					counter+=1
# 				elif config.phase_ref[phase_iter][lane_info] == 'R': pass    #redundant check
# 				else : assert False
# 		assert counter ==2,f'count _value: {counter}'
# 		phase_state_dict[phase_iter] = np.array(phase_state_dict[phase_iter], dtype = float)
# 		phase_state_dict[phase_iter] = np.pad(phase_state_dict[phase_iter], (0, pad), mode='constant', constant_values=0.0)
# 		# obs_dict_req[phase_iter] =  copy.deepcopy(phase_state_dict[phase_iter])

# 		assert np.shape(phase_state_dict[phase_iter])[0] == config.state_size*(config.max_robot_lane*2),f'LHS:{np.shape(phase_state_dict[phase_iter])[0]}, RHS:{config.state_size*config.max_robot_lane*2}'
# 		phase_state_list.extend(phase_state_dict[phase_iter])
		
# 		# print(f'\n\n successful: phase_iter:{phase_iter}')

# 		#### editing code input as per phase kernel module requirement
# 		phase_state_dict_req[phase_iter] = np.reshape(phase_state_dict[phase_iter],((2*config.max_robot_lane), config.state_size ))
# 		phase_state_dict_req[phase_iter] = np.expand_dims(phase_state_dict_req[phase_iter], axis=0)
# 		# print(f'******** before np_array operations: shape{np.shape(phase_state_dict_req[phase_iter])}')
# 		# obs_dict_req[phase_iter] = np.reshape(obs_dict_req[phase_iter],((2*config.max_robot_lane), config.state_size ))

# 	# print(phase_state_dict_req.values())
# 	phase_state_req =  np.concatenate(list(phase_state_dict_req.values()), axis= 0)
# 	phase_state_req = np.expand_dims(phase_state_req, axis=0)   # 1,12,28,30

# 	# obs_req =  np.concatenate(list(obs_dict_req.values()), axis= 0)
# 	# print(f'******** afer np_array operations: obs: {np.shape(obs_req)}')


# 	# exit()
# 	phase_state_np = np.array(phase_state_list, dtype = float)   ## all state values as 1D array
# 	assert  np.shape(phase_state_np)[0] == config.state_size*(2*config.max_robot_lane)*12,f'LHS:{np.shape(phase_state_np)[0]}, RHS:{config.state_size*(2*config.max_robot_lane)*12}'

# 	# print(f'state!!!!!!!!!!!!!!{type(phase_state_list)},{type(phase_state_list[0])},{len(phase_state_list)}')
# 	# print(f'\n state!!!!!!!!!!!!!!{type(phase_state_np)},{type(phase_state_np[0])},{np.shape(phase_state_np)}')
# 	# exit()


# 	return phase_state_list, phase_state_req



def intersection_control(curr_lane_veh, iter, num, time_track, learning_flag, signal):

	if len(curr_lane_veh[iter].t_ser) >0: 

		temp_ind = find_index(curr_lane_veh[iter], time_track)
		assert temp_ind!=None,f'time_track:{curr_lane_veh[iter].t_ser}'
  
		if signal == 'G':
			if (config.L + curr_lane_veh[iter].intsize) >= curr_lane_veh[iter].p_traj[temp_ind] > 0: pass    
			else: 
				assert False, "ERROR - agnet in next ROI requests for future position"
		elif signal == 'R':
			if (config.L + curr_lane_veh[iter].intsize) >= curr_lane_veh[iter].p_traj[temp_ind] > 0: pass    #### EXTRA checks the position 
			else: 
				assert False, "agnet in next ROI still in spawned_set"
	else: assert False,'ERROR - this agent can not be in intersection'

	pre_v = None
	success = None
	v = copy.deepcopy(curr_lane_veh[iter])
	if num >1 and iter >0: 
		pre_v = copy.deepcopy(curr_lane_veh[iter-1])
		pre_temp_ind = find_index(pre_v, time_track)

	if pre_v != None: assert pre_v.id < v.id and pre_v.p_traj[pre_temp_ind] > v.p_traj[temp_ind] ,'pre robot selected wrong'
	v.alpha = 1
	curr_lane_veh[iter], success = safe_set_map_bisect.acc_map(v, pre_v, time_track, learning_flag, flag = 'green', over = 'inter')
 
	return curr_lane_veh[iter]


# sim_obj.spawned_veh[lane], iter, n, time_track, override_lane, lane, signal = 'red'

def lane_control(curr_lane_veh, iter, num, time_track, signal, override_lane, lane, learning_flag):
    
	temp_ind = None

	if len(curr_lane_veh[iter].t_ser)>0: 

		temp_ind = find_index(curr_lane_veh[iter], time_track)
		assert temp_ind!=None, f' time_track:{curr_lane_veh[iter].t_ser}'
		if 0 > curr_lane_veh[iter].p_traj[temp_ind] >= config.int_start[lane]: pass    
		else: 
			print("ERROR in green signal")
			assert False
	else: pass

	pre_v = None
	success = None
	v = copy.deepcopy(curr_lane_veh[iter])
	if num > 1 and iter > 0: 
		pre_v = copy.deepcopy(curr_lane_veh[iter-1])
		pre_temp_ind = find_index(pre_v, time_track)

	if pre_v != None: 
		# print(f'prev_index:{pre_temp_ind}')
		# print(f'prev:{pre_v.p_traj[pre_temp_ind]}')
		# print(f'curr_index:{temp_ind}')
		# print(f'curr:{v.p_traj[temp_ind]}')
		if temp_ind!= None:  assert pre_v.id < v.id and pre_v.p_traj[pre_temp_ind] > v.p_traj[temp_ind],'pre robot selected wrong'
  
	if signal == 'green':
		if lane in override_lane:
			if v.id in override_lane[lane]: 
				v.alpha = 1
				assert False	

		curr_lane_veh[iter], success = safe_set_map_bisect.acc_map(v, pre_v, time_track, learning_flag, flag = signal)  
	
	elif signal =='red':
		curr_lane_veh[iter], success = safe_set_map_bisect.acc_map(v, pre_v, time_track, learning_flag, flag = signal) 
	elif signal =='lane_over':
		v.alpha = 1
		curr_lane_veh[iter], success = safe_set_map_bisect.acc_map(v, pre_v, time_track, learning_flag, flag = 'green') 
  
	else: assert False, signal

	return curr_lane_veh[iter], pre_v, success

	



def  done_pos_check(veh_):

	return veh_.p_traj[-1] > veh_.intsize + config.L and veh_.p_traj[-1] < (veh_.intsize +  config.L - veh_.int_start/2) 



def vehicle_initialise(veh):

	if len(veh.t_ser) < 1:
		veh.t_ser = [veh.sp_t]
		veh.p_traj = [veh.p0]
		veh.v_traj = [veh.v0]
		veh.u_traj = []

	else: pass     
       
	return	veh	


def max_physical_acc(veh_iter, time_int):


	X = []
	temp_ind = find_index(veh_iter, time_int)
	veh_iter.u_traj.append(copy.deepcopy(veh_iter.u_max))
	assert temp_ind!=None
	X.append(veh_iter.p_traj[temp_ind])
	X.append(veh_iter.v_traj[temp_ind])
	x_next = copy.deepcopy(compute_state(X, veh_iter.u_traj[temp_ind],round( config.dt,1),veh_iter.v_max))
	assert x_next[1]<= veh_iter.v_max and  x_next[1]>= veh_iter.v_min
	veh_iter.p_traj.append(copy.deepcopy(x_next[0]))
	veh_iter.v_traj.append(copy.deepcopy(x_next[1]))
	veh_iter.u_safe_min[veh_iter.t_ser[-1]] = veh_iter.u_min
	veh_iter.u_safe_max[veh_iter.t_ser[-1]] = veh_iter.u_max

 
	return veh_iter



def update_plot(rew):
	plt.clf()
	plt.plot(rew, marker='o')
	plt.title('RL Training Rewards')
	plt.xlabel('Time Step')
	plt.ylabel('Reward')
	plt.grid(True)

def webster(arr):
	lp = config.vm / (2*config.u_max) + config.int_bound/config.vm
	sig0 = 1.2
	sfr = 2 * config.vm / (sig0 * config.L)
	Y = 2 * arr/sfr
	C = (1.5 * lp * 2 + 5)/(1-Y)
	return math.ceil(C/2)


def find_index(this_v, this_time): 
	# returns the index of 'this_time' in time series data of vehicle 'this_v'

	#print(f'{round(this_time,1)},{this_v.t_ser},{this_time},{this_v.id}')
	#print("to check whether if inside",round(this_time,1) in this_v.t_ser,this_v.t_ser.index(round(this_time,1)))
	if round(this_time,1) in this_v.t_ser:
		return this_v.t_ser.index(round(this_time,1))
	else: # error!
		#assert not (len(this_v.t_ser)>0) and (this_time <= this_v.t_ser[-1])
		return None


def pres_seper_dist(pre_v, this_v, this_time):
	# returns the present seperation distancebetween vehicle 'this_v' and the vehicle
	# ahead of it on its lane, 'pre_v' at time isntant 'this_time'
	# this is used only in finding initial seperation

	t_ind_this = find_index(this_v, this_time)
	t_ind_pre = find_index(pre_v, this_time)

	if (t_ind_pre == None):
		# print(f"\nlen of prev.u_traj: {len(pre_v.u_traj)}\n")
		if (len(pre_v.u_traj)>0):
			dist0 = - (2 * pre_v.p0) + pre_v.intsize

	else:
		dist0 = (pre_v.p_traj[t_ind_pre]	- (this_v.p0))
		#print(dist0)
		#print(type(dist0))

	if (round(dist0,4) >= 0): # to avoid numenrical errors
		return dist0


def get_follow_dist(pre_v, this_v, this_time):
	# returns the required following distance between vehicle 'this_v' and vehicle 'pre_v'
	# so that rear end safety constraint is satisfied at the time of spawning 'this_v'

	t_ind_this = find_index(this_v, this_time)
	t_ind_pre = find_index(pre_v, this_time)

	if (t_ind_pre == None) and (len(pre_v.p_traj)>0):
		dist0 = 0

	if (t_ind_pre != None) and (t_ind_this == None):
		pre_v0 = pre_v.v_traj[t_ind_pre]
		#dist0 = (((pre_v0)**2) - (this_v.v0**2))/(2*(-3.0))
		dist0 = ( ((pre_v0)**2)/ (2*(pre_v.u_min + 0.001))) - ((this_v.v0**2)/(2*(this_v.u_min + 0.001)))
		
	c0 = max(0, dist0)
	foll_dist1 = 1.45*((pre_v.length) + c0)
	return foll_dist1


def breaking_pos_of_veh_at_time(_veh_obj, _time_duration, _init_time_):
	time_index_curr = find_index(_veh_obj, _init_time_)

	if time_index_curr == None:
		_veh_init_pos = _veh_obj.p0
		_veh_init_vel = _veh_obj.v0

	else:
		_veh_init_pos = _veh_obj.p_traj[time_index_curr]
		_veh_init_vel = _veh_obj.v_traj[time_index_curr]


	return _veh_init_pos + (_veh_init_vel * _time_duration) +  (0.5 * _veh_obj.u_min * (_time_duration ** 2))


def check_init_config(_this_veh, _prev_veh, _t_inst): 
	# returns True if the rear-end-safety is satisfied with initial velocity of 
	# '_this_veh' and velocity of '_prev_veh' at time instant '_t_inst'
	# else, returns False

	if _prev_veh == None:
		return True

	d0 = pres_seper_dist(_prev_veh, _this_veh, _t_inst)
	ed1 = get_follow_dist(_prev_veh, _this_veh, _t_inst)
	
	#print(f"d0: {d0}, ed1: {ed1}")

	# if (ed1 <= d0):
	# 	return True

	# else:
	# 	return False

	t_ind_this = find_index(_this_veh, _t_inst)
	t_ind_pre = find_index(_prev_veh, _t_inst)

	if (t_ind_pre == None) and (len(_prev_veh.p_traj) > 0):
		dist0 = 0

	if (t_ind_pre != None) and (t_ind_this == None):

		# if (_prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - 1.5*_prev_veh.length) >= max(0, ((_prev_veh.v_traj[t_ind_pre]**2) - (_this_veh.v0**2))/(2*_this_veh.u_min)):
		# 	return True

		# else:
		# 	return False

		if (_prev_veh.u_min != _this_veh.u_min):

			_a = 0.5 * (_prev_veh.u_min - _this_veh.u_min)
			_b = (_prev_veh.v_traj[t_ind_pre] - _this_veh.v0)
			_c = _prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - (2*_prev_veh.length)

			_gamma_pre = - _prev_veh.v_traj[t_ind_pre] / _prev_veh.u_min
			_gamma_this = - _this_veh.v0 / _this_veh.u_min

			_p = _gamma_this - (-_b/(2*_a))
			_q = _a*((-_b/(2*_a))**2) + _b*((-_b/(2*_a))) + _c

			if (_c >= 0) and (breaking_pos_of_veh_at_time(_prev_veh, _gamma_pre, _t_inst) - breaking_pos_of_veh_at_time(_this_veh, _gamma_this, _t_inst) >= 2*_prev_veh.length) and \
			((_a <= 0) or (_c >= max(0, _this_veh.v0 - _prev_veh.v_traj[t_ind_pre])*(_this_veh.v0 - _prev_veh.v_traj[t_ind_pre])/(4*_a) )):

				return True

			else:
				return False

		else:
			if (_prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - 2*_prev_veh.length) >= max(0, ((_prev_veh.v_traj[t_ind_pre]**2) - (_this_veh.v0**2))/(2*(_this_veh.u_min))):
				#print(f'LHS:{(_prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - 2*_prev_veh.length)},RHS:{max(0, ((_prev_veh.v_traj[t_ind_pre]**2) - (_this_veh.v0**2))/(2*(_this_veh.u_min)))}')
				#print(f'prev_pos:{_prev_veh.p_traj[t_ind_pre]}, curr_pos:{_this_veh.p0}')
				return True
			else:
				return False


	else:
		return False



# def check_init_config(_this_veh, _prev_veh, _t_inst): 
# 	# returns True if the rear-end-safety is satisfied with initial velocity of 
# 	# '_this_veh' and velocity of '_prev_veh' at time instant '_t_inst'
# 	# else, returns False

# 	if _prev_veh == None:
# 		return True

# 	d0 = pres_seper_dist(_prev_veh, _this_veh, _t_inst)
# 	ed1 = get_follow_dist(_prev_veh, _this_veh, _t_inst)
	
# 	#print(f"d0: {d0}, ed1: {ed1}")

# 	if (ed1 <= d0):
# 		return True

# 	else:
# 		return False


def check_compat(v1, v2):
	# returns -1 if vehicles 'v1' and 'v2' belong to same lane
	# returns 0 if vehicles 'v1' and 'v2' belong to incompatible lanes
	# else returns 1

	if v1.lane == v2.lane:
		return -1

	elif v2.lane in config.incompdict[v1.lane]:
		return 0

	else:
		return 1


def compute_state(X, u, dt, vmax):
	# returns the next state when the current state is '_X',
	# step input '_u' with a discrete time step of '_dt' units

	
	##### integration of linear system with matrix exponential #####

	#r1 = ((_X[0]) + (_X[1]*round(_dt,1)) + ((round(_dt,1)**2)/2)*_u)
	#r2 = (_X[1]) + (round(_dt,1)*_u)

	r2 = (min(max(0, X[1] +  u*round(dt,1)),vmax))
	if u!=0:
		if r2 == vmax: 
			delt = abs((vmax - X[1])/u)
			assert delt <=dt
			r1 = ((X[0]) + (X[1]*delt) + (delt**2/2)*u) + vmax*(dt - delt)
			#print(f'1:{r1}')
		elif r2 == 0:  
			delt = abs(X[1]/u)
			assert delt <=dt
			r1 = ((X[0]) + (X[1]*delt) + ((delt**2)/2)*u)
			#print(f'2:{r1}')
		elif 0<r2<vmax: 
			r1 = ((X[0]) + (X[1]*round(dt,1)) + ((round(dt,1)**2)/2)*u)
			#print(f'3:{r1}')
	elif u==0:
		r1 = (X[0]) + (X[1]*round(dt,1))
		#print(f'4:{r1}')

	return [r1,r2]



	##### integration of linear system with matrix exponential #####
	

def constraint_dynamics(state_opti_var, next_state, opti_class_object):
	# applies one step dynamics constraint for 'opti_class_object'
	# where 'state_opti_var' is the next step state optimization variable
	# and 'next_state' is the computed state update 

	opti_class_object.subject_to(state_opti_var == next_state)


def prov_constraint_vel_max(veh_object, velo_opti_var, posi_opti_var, opti_class_object): 
	# applies constraint on the vehicle velocity optimization variable, 'velo_var'
	# so that it does not enter the intersection

	opti_class_object.subject_to((velo_opti_var**2) <= (2*(veh_object.u_min)*(posi_opti_var + 0.1))) #+ (0.1/(((0.2*veh_object.num_prov_phases) + 1))) )))
	#opti_class_object.subject_to((velo_opti_var**2) <= (2*(veh_object.u_min)*(posi_opti_var+0.2)))
	#opti_class_object.subject_to((velo_opti_var**2) <= (2*(-3.0)*posi_opti_var))


def constraint_vel_bound(veh_object, velo_opti_var, opti_class_object):
	# applies upper and lower bounds on the vehicle velocity optimization variable, 'velo_opti_var'

	opti_class_object.subject_to(opti_class_object.bounded(veh_object.v_min, velo_opti_var, veh_object.v_max))
	#opti_class_object.subject_to(opti_class_object.bounded(0, velo_opti_var, 11.11))


def constraint_acc_bound(veh_object, acc_opti_var, opti_class_object):
	# applies upper and lower bounds on the vehicle acceleration optimization variable, 'acc_opti_var'

	opti_class_object.subject_to(opti_class_object.bounded(veh_object.u_min, acc_opti_var, veh_object.u_max)) 
	#opti_class_object.subject_to(opti_class_object.bounded(-3.0, acc_opti_var, 3.0))

def constraint_init_pos(posi_opti_var, opti_class_object, init_pos):
	# applies initial position constraint for vehicle position optimization variable, 'posi_opti_var'

	opti_class_object.subject_to(posi_opti_var[0,0] == init_pos)

def constraint_init_vel(velo_opti_var, opti_class_object, init_vel): 
	# applies initial velocity constraint for vehicle velocity optimization variable, 'velo_opti_var'

	opti_class_object.subject_to(velo_opti_var[1,0] == init_vel)
	

def prov_constraint_rear_end_safety(veh_object, prev_veh, init_t_inst, posi_var, vel_var, opti_class_object): pass


def coord_constraint_rear_end_safety(veh_object, prev_veh, init_t_inst, posi_var, vel_var, opti_class_object): pass
	

	

def snapdata(veh_object, train_sim_num, _sim, _train_iter_num,snap):
	for lane in config.lanes:
		for _ in len(range(get_num_of_objects(veh_object[lane]))):
			h = {}
			h[veh_object[lane][_].id] = veh_object[lane][_]
			if snap==1:
				dbfile = open(f'../data/captured_snaps/arr_{veh_object[lane][_].arr}/sim_{_sim}/'+str(veh_object[lane][_].id), 'wb')
			pickle.dump(h, dbfile)
			dbfile.close()
	
    # print("stored", veh_object.id)  # Uncomment this line if you want to print a message				






def constraint_waiting_time(_wait_time, pos_var, opti_class_object):
	st = int(math.ceil(((round(_wait_time,1))/round(config.dt,1))))
	#print("waiting time st:", st)
	#return
	if st > 0:
		if pos_var.size2() > st:
			opti_class_object.subject_to(pos_var[st] <= 0)

		else:
			opti_class_object.subject_to(pos_var[-1] <= 0)



def get_num_of_objects(td_list):
	total_num_objects = 0
	for li in config.lanes:
		total_num_objects += len(td_list[li])

	return total_num_objects


def get_set_f(_p_set):
	F = [deque([])for _ in config.lanes]
	for l in config.lanes:
		if len(_p_set[l]) > 0:
			F[l].append(_p_set[l][0])

	return F
			
""" 
def storedata(veh_object, train_sim_num, _sim, _train_iter_num):
    m = {}
    m[veh_object.id] = veh_object
    
    if config.rl_flag and (not config.run_coord_on_captured_snap):
    	if config.used_heuristic == None:
    		dbfile = open(f'../data/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    	else:
    		dbfile = open(f'../data/{config.used_heuristic}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    else:
    	dbfile = open(f'../data/{config.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    pickle.dump(m, dbfile)
    dbfile.close()
    #print("stored", veh_object.id) 
"""



def storedata(veh_object, train_sim_num, _sim, _train_iter_num, version):
	m = {}
	m[veh_object.id] = veh_object

	#if config.rl_flag and (not config.run_coord_on_captured_snap):
	#    if config.used_heuristic == None:
	#        dbfile = open(f'../data/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
	#    else:
	#        dbfile = open(f'../data/{config.used_heuristic}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
	#else:
	#    dbfile = open(f'../data/{config.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')

	#/data_version/version_{int(version)}/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/model_weights_itr_{int(time_track)}"  
	#				os.makedirs(directory, exist_ok=True)
	directory = f'../../data_version/version_{int(version)}/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}' 
	if not os.path.exists(directory):
		os.makedirs(directory)
	dbfile = open(f'../../data_version/version_{int(version)}/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
	pickle.dump(m, dbfile)
	dbfile.close()
    # print("stored", veh_object.id)  # Uncomment this line if you want to print a message

""" 

def snapdata(veh_object, train_sim_num, _sim, _train_iter_num):
    m = {}
    m[veh_object.id] = veh_object
    dbfile = open(f'../data/{config.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
	pickle.dump\
	
	
	dbfile = open(f'../data/{config.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
	dbfile = open(f'../data/{config.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    pickle.dump(m, dbfile)
    dbfile.close()
    # print("stored", veh_object.id)  # Uncomment this line if you want to print a message

 """






def get_feasible_schedules(__prov_set):

	'''print("feasible schedule computation started!")

	veh_set = []

	veh_set_free_lanes = []


	for lii in range(len(config.lanes)):
		ln = config.lanes[lii]

		if len(config.incompdict[ln]) > 0:
			for v in __prov_set[ln]:
				veh_set.append(v)

		else:
			for v in __prov_set[ln]:
				veh_set_free_lanes.append(v)
		
	print("len of veh_set:", len(veh_set))
	print("len of veh in free lanes:", len(veh_set_free_lanes))
	perms = list(permutations(veh_set))
	permsdict = copy.deepcopy(perms)


	for h in range(len(perms)):
		outer_for_loop_flag = 0

		for k in range(1, len(perms[h])):
			i = perms[h][k]
			if outer_for_loop_flag == 1:
				break

			reverse_index_list = list(range(k))
			reverse_index_list.reverse()


			for m in reverse_index_list:
				j = perms[h][m]
				if (j.lane == i.lane) and (j.sp_t >= i.sp_t):
					permsdict[h] = []
					outer_for_loop_flag = 1
					break
		
	feasible_perms = []
	for p in permsdict:
		if len(p) > 0:
			feasible_perms.append(list(p))

	for p in feasible_perms:
		for ve in veh_set_free_lanes:
			p.append(ve)

	for p in feasible_perms:
		x = [v.id for v in p]
		#print(x)

	print("feasible schedules computed!", len(feasible_perms))'''

	def in_order_combinations(*lists):
		lists = list(filter(len, lists))

		if len(lists) == 0:
			yield []

		for lst in lists:
			element = lst.pop()
			for combination in in_order_combinations(*lists):
				yield combination + [element]
			lst.append(element)

	feasible_perms = [_ for _ in in_order_combinations(*__prov_set)]

	# print(f"feasible_perms: {(feasible_perms)}")
	
	return feasible_perms


def check_feasibility(state, inp, veh_obj, prev_veh_obj, init_pos, init_vel, init_time, opti_class_object, debug_flag):

	if debug_flag == 0:

		init_pos_flag = 1
		init_vel_flag = 1
		#### checking initial conditions ####
		if round(opti_class_object.value(state[0,0]),4) != round(init_pos,4):
			print("initial position violated! constraint:", init_pos, "solution value:", opti_class_object.value(state[0,0]))
			init_pos_flag = 0
		
		else:
			pass



		if round(opti_class_object.value(state[1,0]),4) != round(init_vel,4):
			print("initial velocity violated! constraint:", init_vel, "solution value:", opti_class_object.value(state[1,0]))
			init_vel_flag = 0

		else:
			pass


		#### checking velocity and acceleration bounds ####
		bound_flag = 1
		for k in range(state.size2()):
			if (round(opti_class_object.value(state[1,k]),4) > veh_obj.v_max) or (round(opti_class_object.value(state[1,k]),4) < 0):
				print("velocity bound violated!", round(opti_class_object.value(state[1,k]),4))
				bound_flag = 0
				break
			
			if (round(opti_class_object.value(inp[k]),4) > veh_obj.u_max) or (round(opti_class_object.value(inp[k]),4) < veh_obj.u_min):
				print("acceleration bound violated!")
				bound_flag = 0
				break

		if bound_flag == 1:
			pass

		#### checking rear-end safety ####
		rear_end_flag = 1

		if prev_veh_obj != None:
			ind_prev = find_index(prev_veh_obj, init_time)
			min_difference =  -80
			for k in range(state.size2()):
				bool_for_index = ((ind_prev + k) < len(prev_veh_obj.t_ser))
				
				if bool_for_index:
					bool_for_constraint_with_prev_traj = (round(opti_class_object.value(state[0,k]), 4) <= round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))

					min_difference = max(min_difference, (opti_class_object.value(state[0,k])) - (prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )))))

					if (not bool_for_constraint_with_prev_traj):
						print("rear-end-safety constraint violated!1", round(opti_class_object.value(state[0,k]), 4), round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))
						rear_end_flag = 0
						break


				else:
					time_since_last_planned = round(((init_time + (k*config.dt)) - prev_veh_obj.t_ser[-1]), 1) 
					bool_for_constraint_without_prev_traj = (round(opti_class_object.value(state[0,k]), 4) <= round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))

					min_difference = max(min_difference, (opti_class_object.value(state[0,k])) - (((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) ))))

					if (not bool_for_constraint_without_prev_traj):
						print("rear-end-safety constraint violated!2", round(opti_class_object.value(state[0,k]), 4), round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
						rear_end_flag = 0
						break

			with open(f"./data/min_constraint_diff.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[min_difference]])

		if rear_end_flag == 1:
			print("rear-end-safety constraint satisfied")



		#### checking for intersection safety ####

		inter_safety_flag = 1
		min_wait_time_index = int(math.ceil(((round(veh_obj.feat_min_wait_time,1))/round(config.dt,1))))

		for k in range(min_wait_time_index+1):
			if round(opti_class_object.value(state[0,k]),4) > 0:
				print("minimum wait time constratint violated!", min_wait_time_index, k, opti_class_object.value(state[0,k]))
				inter_safety_flag = 0
				break

		if inter_safety_flag == 1:
			pass


		


	if debug_flag == 1:

		init_pos_flag = 1
		init_vel_flag = 1
		#### checking initial conditions ####
		if round(opti_class_object.value(state[0,0]),4) != round(init_pos,4):
			print("initial position violated! constraint:", init_pos, "solution value:", opti_class_object.debug.value(state[0,0]))
			init_pos_flag = 0
		
		else:
			pass


		if round(opti_class_object.value(state[1,0]),4) != round(init_vel,4):
			print("initial velocity violated! constraint:", init_vel, "solution value:", opti_class_object.debug.value(state[1,0]))
			init_vel_flag = 0

		else:
			pass


		#### checking velocity and acceleration bounds ####
		bound_flag = 1
		for k in range(state.size2()):
			if (round(opti_class_object.debug.value(state[1,k]),4) > veh_obj.v_max) or (round(opti_class_object.debug.value(state[1,k]),4) < 0):
				print("velocity bound violated!")
				bound_flag = 0
				break
			
			if (round(opti_class_object.debug.value(inp[k]),4) > config.u_max) or (round(opti_class_object.debug.value(inp[k]),4) < veh_obj.u_min):
				print("acceleration bound violated!")
				bound_flag = 0
				break

		if bound_flag == 1:
			pass

		#### checking rear-end safety ####
		min_difference = -80
		rear_end_flag = 1
		if prev_veh_obj != None:
			ind_prev = find_index(prev_veh_obj, init_time)
			for k in range(state.size2()):
				bool_for_index = ((ind_prev + k) < len(prev_veh_obj.t_ser))
				if bool_for_index:
					bool_for_constraint_with_prev_traj = (round(opti_class_object.debug.value(state[0,k]), 4) <= round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))

					min_difference = max(min_difference, (opti_class_object.debug.value(state[0,k])) - (prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )))))

					if (not bool_for_constraint_with_prev_traj):
						print("rear-end-safety constraint violated1!", round(opti_class_object.debug.value(state[0,k]), 4), round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))
						rear_end_flag = 0


				else:
					time_since_last_planned = round(((init_time + (k*config.dt)) - prev_veh_obj.t_ser[-1]), 1) 
					bool_for_constraint_without_prev_traj = (round(opti_class_object.debug.value(state[0,k]), 4) <= round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
					
					min_difference = max(min_difference, (opti_class_object.debug.value(state[0,k])) - (((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) ))))

					if (not bool_for_constraint_without_prev_traj):
						print("rear-end-safety constraint violated2!", round(opti_class_object.debug.value(state[0,k]), 4), round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
						rear_end_flag = 0

			with open(f"./data/min_constraint_diff_exceeded.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[min_difference]])

		if rear_end_flag == 1:
			pass

		#### checking for intersection safety ####

		inter_safety_flag = 1
		min_wait_time_index = int(math.ceil(((round(veh_obj.feat_min_wait_time,1))/round(config.dt,1))))

		for k in range(min_wait_time_index):
			if round(opti_class_object.debug.value(state[0,k]),4) > 0:
				print("minimum wait time constratint violated!", min_wait_time_index, k, opti_class_object.debug.value(state[0,k]))
				inter_safety_flag = 0
				break

		if inter_safety_flag == 1:
			pass
		

	assert not ((init_pos_flag == 0) or (init_vel_flag == 0) or (bound_flag == 0) or (rear_end_flag == 0) or (inter_safety_flag == 0))




def check_feasibility_Vc_Vs(V_c, V_s, schedu, sol_pos, sol_vel, sol_acc, curr_time, init_c, opti_steps, sch_count, stri):

	init_pos_flag = 0
	init_vel_flag = 0
	vel_bound_flag = 0
	acc_bound_flag = 0
	rear_end_flag = 0
	inter_safety_flag = 0


	for _lan in config.lanes:
		
		### checking initial position constraint ###
		if not init_pos_flag:
			veh_pos_index = 0
			for veh_pos_ in sol_pos[_lan]:
				if not (round(veh_pos_[0], 4) == round(init_c[_lan][veh_pos_index][0], 4)):
					init_pos_flag = 1
					pos_a = round(veh_pos_[0], 4)
					pos_b = round(init_c[_lan][veh_pos_index][0], 4)
					break
				veh_pos_index += 1

		### checking initial velocity constraint ###
		if not init_vel_flag:
			veh_vel_index = 0			
			for veh_vel_ in sol_vel[_lan]:
				if not (round(veh_vel_[0], 4) == round(init_c[_lan][veh_vel_index][1], 4)):
					init_vel_flag = 1
					vel_a = round(veh_vel_[0], 4)
					vel_b = round(init_c[_lan][veh_vel_index][1], 4)
					break
				veh_vel_index += 1



		### checking velocity bounds ###
		if not vel_bound_flag:
			for ind, veh_vel_ in enumerate(sol_vel[_lan]):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[ind]
				for vel_ in veh_vel_:
					if round(vel_, 4) < current_vehicle.v_min:
						vel_bound_flag = 1
						break

					if round(vel_, 4) > current_vehicle.v_max:
						vel_bound_flag = 1
						break

				if vel_bound_flag:
					break

		### checking acceleration bounds ###
		if not acc_bound_flag:
			for ind, veh_acc_ in enumerate(sol_acc[_lan]):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[ind]
				for acc_ in veh_acc_:
					if round(acc_, 4) < current_vehicle.u_min:
						acc_bound_flag = 1
						break

					if round(acc_, 4) > current_vehicle.u_max:
						acc_bound_flag = 1
						break

				if acc_bound_flag:
					break



		### checking rear-end-safety ###
		if not rear_end_flag:
			for veh_index_rear_end in range(len(V_c[_lan])):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[veh_index_rear_end]
				if (veh_index_rear_end == 0) and (len(V_s[_lan]) > 0):
					prev_veh_ = V_s[_lan][-1]
					ind_prev = find_index(prev_veh_, round(curr_time,1))
					for time_index in range(1, len(sol_pos[_lan][veh_index_rear_end])):

						if len(V_s[_lan][-1].t_ser) > (ind_prev + time_index): 
							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[ind_prev + time_index] - (prev_veh_.length)) , 4) > 0.0001):
								rear_end_flag = 1
								print("here1")
								break

							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[ind_prev + time_index] - ((prev_veh_.length) + ((( ((prev_veh_.v_traj[ind_prev + time_index]**2) / (2*prev_veh_.u_min) )- (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) ))))), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here2")
								break

						else:
							time_since_last_planned = round(((ind_prev + time_index + 1 - len(prev_veh_.t_ser))*config.dt), 1)
							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[-1] + (prev_veh_.v_traj[-1]*time_since_last_planned) - (prev_veh_.length)), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here3")
								break

							if (round(sol_pos[_lan][veh_index_rear_end][time_index] -(prev_veh_.p_traj[-1] + (prev_veh_.v_traj[-1]*time_since_last_planned) - ((prev_veh_.length) + ((((prev_veh_.v_traj[-1]**2) / (2*prev_veh_.u_min) )- (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) )))), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here4")
								break

						if rear_end_flag:
							break

				elif ( not (veh_index_rear_end == 0)):

					veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
					previous_vehicle = veh_in_lane[veh_index_rear_end-1]

					for time_index in range(1, len(sol_pos[_lan][veh_index_rear_end])):

						if (round(sol_pos[_lan][veh_index_rear_end][time_index] - ((sol_pos[_lan][veh_index_rear_end-1][time_index] - (previous_vehicle.length))),4) > 0.0001):
							rear_end_flag = 1
							print(sol_pos[_lan][veh_index_rear_end][time_index])

							print("\n")
							print(sol_pos[_lan][veh_index_rear_end])
							print("\n")
							print(sol_pos[_lan][veh_index_rear_end-1][time_index])
							print("\n")
							print(_lan, veh_index_rear_end)
							print("here5")
							break

						if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (sol_pos[_lan][veh_index_rear_end-1][time_index] - ((previous_vehicle.length) + (((( (sol_vel[_lan][veh_index_rear_end-1][time_index]**2) /(2*previous_vehicle.u_min) ) - (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) ))))), 4) > 0.0001):
							rear_end_flag = 1
							print("here6")
							break

				if rear_end_flag:
					break


		if init_pos_flag and init_vel_flag and vel_bound_flag and acc_bound_flag and rear_end_flag:
			break



	### intersection safety ###

	if not inter_safety_flag:
		for time_index in range(opti_steps):
			for _lan in config.lanes:
				if len(config.incompdict[_lan]) > 0:
					for veh_index_int_safety in range(len(V_c[_lan])):
						if (sol_pos[_lan][veh_index_int_safety][time_index] > 0) and (sol_pos[_lan][veh_index_int_safety][time_index] < V_c[_lan][veh_index_int_safety].intsize):
							for incomp_lan in V_c[_lan][veh_index_int_safety].incomp:

								if len(V_s[incomp_lan]) > 0:
									incomp_veh_t_ind = find_index(V_s[incomp_lan][-1], curr_time)

									if (not (incomp_veh_t_ind == None)) and (incomp_veh_t_ind + time_index < len(V_s[incomp_lan][-1].t_ser)):

										if (V_s[incomp_lan][-1].p_traj[incomp_veh_t_ind + time_index] > 0) and (V_s[incomp_lan][-1].p_traj[incomp_veh_t_ind + time_index] < V_s[incomp_lan][-1].intsize):
											inter_safety_flag = 1
											break

						if inter_safety_flag:
							break

					if inter_safety_flag:
						break

			if inter_safety_flag:
				break


	if init_pos_flag:
		print("inital position constraint violated!", sch_count, stri, )

	if init_vel_flag:
		print("inital velocity constraint violated!", sch_count, stri)

	if vel_bound_flag:
		print("velocity bound constraint violated!", sch_count, stri)

	if acc_bound_flag:
		print("acceleration bound constraint violated!", sch_count, stri)

	if rear_end_flag:
		print("rear-end-safety constraint violated!", sch_count, stri)

	if inter_safety_flag:
		print("intersection safety constraint violated!", sch_count, stri)


	if init_pos_flag or init_vel_flag or vel_bound_flag or acc_bound_flag or rear_end_flag or inter_safety_flag:
		print([init_pos_flag, init_vel_flag, vel_bound_flag, acc_bound_flag, rear_end_flag, inter_safety_flag], sch_count, stri)
		if init_pos_flag:
			print(pos_a, pos_b)
		if init_vel_flag:
			print(vel_a, vel_b)

		m = {}
		m['vc'] = V_c
		m['vs'] = V_s
		m['t'] = curr_time
		m['sched'] = schedu
		num_bad_seq_till_now = len(list(os.listdir(f'./data/compare_files/arr_{config.arr_rate}/bad_seqs/')))
		dbfile = open(f"./data/compare_files/arr_{config.arr_rate}/bad_seqs/seq_{num_bad_seq_till_now}", 'wb')
		pickle.dump(m, dbfile)
		dbfile.close()
		


def seq_checker(veh_):

	for li in range(len(config.lanes)):
		l = config.lanes[li]
		if len(veh_[l]) > 1:
			#for ind in range(len(veh_.coord_veh[l])):
			veh_id = [bot_.id for bot_ in veh_[l] ]
			if veh_id != sorted(veh_id) and sum([veh_id.count(_) for _ in veh_id]) !=len(veh_id):
				print("/n",	"sequence wrong") 
				print(game)



		
############### ofor to remove NONE					
""" 
				lane_itr_var = 0
				for li in range(len(config.lanes)):
					l = config.lanes[li]
					if len(sim_obj.coord_veh[l]) > 0:
						for ind in range(len(sim_obj.coord_veh[l])):
							previous_value = None
							#for key in sorted(sim_obj.coord_veh[l][ind].tc_flag_time.keys()):
							if sim_obj.coord_veh[l][ind].tc_flag_time[time_track] is None:
								sim_obj.coord_veh[l][ind].tc_flag_time[time_track] = previous_value
							previous_value = sim_obj.coord_veh[l][ind].tc_flag_time[time_track] 
"""
					


""" 
def del_fol_inside():

	train_iter = 100000 

	train_sim_list = [i for i in range(1,11) ]
	sim_num = [i for i in range(1,11) ]
	for _ in config.arr_rates_to_simulate:
	   	for __ in train_sim_list:
    	    for ___ in sim_num:
        	    files = glob.glob(f"/home/user/mowbrayr/AIM-code/data/arr_{_}/test_homo_stream/train_sim_{__}/train_iter_{train_iter}/pickobj_sim_{___}/*")
                for f in files: os.remove(f)

 """






