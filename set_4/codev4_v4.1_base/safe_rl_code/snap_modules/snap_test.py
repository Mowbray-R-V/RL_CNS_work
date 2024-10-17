import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
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
import safe_set_map_bisect_backup1
import green_phase
import functions
import set_class
import vehicle_snap as vehicle
import contr_sig_rl
import delete_data
from multiprocessing import Pool
import logging
import cProfile
from ddpg_related_class import DDPG as Agent





    
sim_obj = set_class.sets()
lane = 1

curr_traj = [-7, -6.863159326486035, -6.4322983685393895, -5.7344481449830464, -4.955557176790283, -4.2406743945562075, \
    -3.642375564794198, -3.1608059106659336, -2.78042330121786, -2.482980271379993, -2.2524752857915185, -2.0751714564448305, -1.9386671855853992, -1.8339077390970022, -1.7537016879539107, -1.6925010517514936,\
        -1.6457974448444233, -1.6101904038655261, -1.5832031314450716, -1.562716922786456, -1.5471658391311762, -1.5354290545390277, -1.5266351952548372, -1.5200524439649206, -1.5151168382879785, -1.5114156020459992, \
            -1.5086376533285053, -1.506548613737944, -1.504969668021604, -1.5037751645240958, -1.502869758994567, -1.502185021045838, -1.501672860868458, -1.50129668154749, -1.5010245589976468, -1.5008271360053966, \
                -1.5006849733870655, -1.5005829062281975, -1.5005084313503212, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, \
                    -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, \
                        -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, -1.5004769197798626, \
                            -1.4537810978040864, -1.3242947181947509, -1.1453262254010386, -0.9561076244793612]


### 93 to 98
prev_traj = [-0.7502993475830683, -0.6205399003625905, -0.43118927829091847, -0.332612182081661, -0.2640795975317834, -0.2119708486272344]
############# T = 93 ##############
start_curr = -1.5004769197798626
start_prev = -0.7502993475830683
prev_vel = 0
curr_vel = 0
_arr_rate_ = 0.1
arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_}
time_track = 0



veh_prev = vehicle.Vehicle(lane, start_prev,prev_vel,  data_file.vm[lane], lane, data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array_)
veh_curr = vehicle.Vehicle(lane, start_curr ,curr_vel,  data_file.vm[lane], lane, data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array_)

veh_prev.sp_t = time_track
veh_curr.sp_t = time_track

veh_prev.id =1
veh_curr.id = 2
sim_obj.spawned_veh[lane].append(veh_prev)
sim_obj.spawned_veh[lane].append(veh_curr)

  
sim = 1
ss = [data_file.buff_size, 64] # buffer size and sample size
actor_lr = 0.0001
critic_lr = 0.001
p_factor = 0.0001
d_factor = 0
agent = None 
learning_flag = 0
train_iter = 100000
train_sim = 1    
algo_option = "rl_modified_ddswa"






agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)

if (not learning_flag) and (data_file.used_heuristic == None):
    agent.actor_model_.load_weights(f"../data/merged_replay_buffer_with_next_state/train_sim_{train_sim}/trained_weights/actor_weights_itr_{train_iter}")
    max_rep_sim = 1

dict_sig =  {93: {1: 'G', 4: 'R', 10: 'R', 8: 'R', 11: 'R', 7: 'G', 2: 'R', 5: 'R'}, 94: {11: 'G', 4: 'R', 1: 'R', 8: 'R', 2: 'R', 7: 'G', 10: 'R', 5: 'R'}, \
    95: {11: 'R', 4: 'R', 1: 'R', 8: 'R', 2: 'R', 7: 'G', 10: 'R', 5: 'R'}, 96: {11: 'R', 4: 'R', 1: 'R', 8: 'R', 2: 'R', 7: 'G', 10: 'R', 5: 'R'}, \
        97: {11: 'G', 4: 'R', 1: 'R', 8: 'R', 2: 'R', 7: 'G', 10: 'R', 5: 'R'} }
override_lane =  {}
lane_sig_stat = {0:['R',0], 1:['G',0], 2:['R',0], 3:['R',0], 4:['R',0], 5:['R',0], 6:['R',0], 7:['G',0], 8:['R',0], 9:['R',0], 10:['R',0], 11:['R',0]}
   
   
   
while time_track< 5:
   
        
    print(f'sig_status:{dict_sig[93+time_track][lane]}', lane)
    veh_prev.alpha = 1
    veh_curr.alpha = 1
        
   
    n = len(sim_obj.spawned_veh[lane])
    for iter in range(n):
        #sim_obj.spawned_veh[lane][iter].global_sig_val[time_track] = signal
        sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = override_lane # list(override_lane.keys())
        
        

    for _ in override_lane: 
        print("fgdgg",_)
        dict_sig[_]='R'
        for lane__ in data_file.incompdict[_]:
            #print(f'dict:{dict_sig}')
            dict_sig[lane__]='R'    
    
    green_zone = -1*data_file.vm[lane]*data_file.dt + (data_file.vm[lane]**2)/(2*(max(data_file.u_min[lane],-(data_file.vm[lane]/data_file.dt))))  
    print(f'len:{len(data_file.incompdict[lane])}')
    if len(data_file.incompdict[lane])>0:
        override_veh = []
        if dict_sig[93+time_track][lane]=='G':
            print("ABC",n,lane)
            n = len(sim_obj.spawned_veh[lane])
            for iter in range(n):
                print(f'inside green:, id:{sim_obj.spawned_veh[lane][iter].id}, time:{time_track}, size:{n}')
                pre_v = None
                success = None
                v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
                if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
                sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.green_map(v, pre_v, time_track)#, algo_option, learning_flag, 1, sim, train_iter, train_sim)
                print("prev_veh done")
                if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 							
        elif dict_sig[93+time_track][lane]=='R':
            #print("inside red_signal")
            n = len(sim_obj.spawned_veh[lane])
            print("DEF",n,lane)
            for iter in range(n):
                print("DEF--1")
                v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
                success = None
                pre_v = None
                if n >1 and iter >0 : pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
                sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.red_map(v, pre_v,time_track)
                print(f'red_map')                
                if success == False: ##OVERRIDE
                    #print(f'inside RED:, id:{sim_obj.spawned_veh[lane][iter].id}, time:{time_track}, size:{n}')
                    sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect_backup1.green_map(v, pre_v,time_track)
                    print(f'inside RED_green:, id:{sim_obj.spawned_veh[lane][iter].id}, time:{time_track}, size:{n}')
                    #exit()
                    override_veh.append(v.id)
                if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 
            if len(override_veh)>0: override_lane[lane] = override_veh
            #if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 

    assert len(set(override_lane.keys())) == len(override_lane), f'duplicates present in over_ride lane: {override_lane}'
    
    print(f'over:{override_lane}',lane)
    for v in  sim_obj.spawned_veh[lane] : print(f' BEFORE_OVERRIDE time:{time_track}, veh id:{v.id},:{v.p_traj}, prev_vel:{v.v_traj}, prev_u:{v.u_traj}, alpha:{v.alpha}')

    for _ in override_lane:
        dict_sig[93+time_track][_]='R'
        for lane_ in data_file.incompdict[_]:
            dict_sig[93+time_track][lane_]='R'   #### override
            n = len(sim_obj.spawned_veh[lane_])
            for iter in range(n):
                v = copy.deepcopy(sim_obj.spawned_veh[lane_][iter])
                success = None
                pre_v = None
                if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane_][iter-1])
                sim_obj.spawned_veh[lane][iter], success= safe_set_map_bisect_backup1.red_map(v, pre_v,time_track) 
                assert success != False
                if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 
        ############# lane_override trajectories #############
        #for iter in range(len(sim_obj.spawned_veh[_])):
        #        #print(sim_obj.spawned_veh[lane][iter].id, override_lane[_])
        #        if sim_obj.spawned_veh[_][iter].id not in override_lane[_]:
        #            v = copy.deepcopy(sim_obj.spawned_veh[_][iter])
        #            success = None
        #            pre_v = None
        #            if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[_][iter-1])
        #            sim_obj.spawned_veh[_][iter], success= safe_set_map.red_map(v, pre_v,time_track) 
        #            if not learning_flag:functions.storedata(v, train_sim, sim, train_iter)  
        ########### all incompatible control variable estimation ############## """
        
        
    for l in data_file.lanes:
        n_in_green = len(sim_obj.spawned_veh[l])
        #print("override", override_lane,l)
        
        if  l in override_lane: c = len(override_lane[l])
        else: c = 0
        v_ind = 0
        while v_ind < n_in_green:
            green_while_flag = 0
            v = sim_obj.spawned_veh[l][v_ind]
            t_ind = functions.find_index(v, time_track)
            if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length) ):   #- v.int_start)):
                #### if agent crosses remove override #####
                if l in override_lane and v.id in override_lane[l]: c -= 1
                #curr_rew += 10*v.priority
                ## sum([v.priority for _ in data_file.lanes for v in sim_obj.spawned_veh[_]])
                sim_obj.done_veh[l].append(v)
                sim_obj.spawned_veh[l].popleft()
                n_in_green -= 1
            else:
                break
        if c==0 and l in override_lane: del override_lane[l] 


    
        
        
        
        
        
    print("lane:",lane)
    for v in  sim_obj.spawned_veh[lane] : print(f'AFTER_OVERRIDE time:{time_track}, veh id:{v.id},:{v.p_traj}, prev_vel:{v.v_traj}, prev_u:{v.u_traj}, alpha:{v.alpha}')
    time_track = round((time_track + data_file.dt), 1)








