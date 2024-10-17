import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.patches import Ellipse
import time
import numpy as np
import itertools as it
from scipy.integrate import odeint
from numpy import interp
import copy
import pickle
import csv
import os
import data_file
import functions
import vehicle

from multiprocessing import Pool

L = 4.5/3  

len_veh = data_file.L# Length of vehicle

int_bound = data_file.int_bound  # size of intersection
int_start = data_file.int_start[0] # start of intersection system
p_init = int_start  # spawn_position

dt = round(data_file.dt,1)  # step size
T_sc = 30  # scheduling time
T_ps = 20  # prescheduling timeb
T_h = 20
# N = 150  # Number of time steps

tend = 0 + 200  # End of simulation time
t_opti = 3  # optimization intervals
t_O = 3
t_rem = 80
t_rem_inc = 40
branches = [0,1,2,3,4,5,6,7,8,9,10,11]
arr_rate = 0.5

lasttime = 0

W_t = 3
W_comf = 10
W_pos = 1
W_v = 1

unassigned = []
assigned = []
ids = 0
ts = []
V_array = []
full_opti = []

vm = 11.11
um = 3

V_dict = {}
#Vnd = {}
blist = []
klist = []

novinsched = []

s = data_file.intersection_path_length # [data_file.int_bound/(4*np.sqrt(2)), data_file.int_bound, (5*data_file.int_bound)/(4*np.sqrt(2))] 

B = data_file.B

def plot_ellipse(veh_object, veh_id, center_pos_on_lane, lane, tc_flag, dict, sig, pos_status, time, agnet_pos, flag=False):

    veh_plot_len = veh_object.length
    veh_plot_width = B/2
    # print(f'values:{veh_plot_len},{veh_plot_width}, b/2:{B/2 }')
    # exit()
    pos = round(center_pos_on_lane, 3)

    tex_ = ""

    ellipse_pos = None
    ellipse_width = veh_plot_len
    ellipse_height = veh_plot_width
    ellipse_angle = None
    ellipse_colour = None
    text_pos = None
    text_to_print = str(veh_id)
    
                        
    # if veh_id == 0: print(f'\n sig_status:{sig}, id:{veh_id},{pos_status}, time:{time}')
    if sig == 'green' or sig == 'green_sp' or sig == 'green__set' or sig == 'green_inter':  
        el_color = 'g' 
    elif sig == 'red':
        el_color = 'b'
    elif sig == 'lime':
        el_color =  '#bfff00'   
    elif sig == 'orange' or sig == 'orange_inter':
        el_color = '#FFA500' 
    elif sig == None: 
        print(f'error in sig, ID:{veh_id},{agnet_pos}, time:{time}')
        # el_color =  '#bfff00' 
        exit()
    else: 
        print("error, something else than  none")       
        # print(sig)
        exit()



    if lane == 1:
        ellipse_pos = (pos, (7*B/2))
        ellipse_angle = 0
        ellipse_colour = el_color

    elif lane == 4:
        ellipse_pos = (B/2, pos)
        ellipse_angle = 90
        ellipse_colour = el_color
        
    elif lane == 7:
        ellipse_pos = (-(-int_bound + pos), B/2)
        ellipse_angle = 180
        ellipse_colour = el_color

    elif lane == 10:
        ellipse_pos = ((7*B/2), -(pos-int_bound))
        ellipse_angle = -90
        ellipse_colour = el_color
        

    elif lane == 2:
        if pos <= 0:
            ellipse_pos = (pos, (5*B/2))
            ellipse_angle = 0
            ellipse_colour = el_color
            
        elif pos < s[lane%3] + veh_object.length:
            ellipse_pos = (((pos/np.sqrt(2))), (5*B/2)-(pos/np.sqrt(2)))
            ellipse_angle = -45
            ellipse_colour = el_color
            
        else:
            ellipse_pos = ((5*B/2), -(pos - (s[lane%3])))
            ellipse_angle = -90
            ellipse_colour = el_color
            

    elif lane == 8:
        if pos <= 0:
            ellipse_pos = (-(pos-int_bound), (3*B/2))
            ellipse_angle = 180
            ellipse_colour = el_color
            
        elif pos < s[lane%3] + veh_object.length:
            ellipse_pos = (-((pos/np.sqrt(2)-int_bound)), (3*B/2) + (pos/np.sqrt(2)))
            ellipse_angle = 135
            ellipse_colour = el_color
            
        else:
            ellipse_pos = ((3*B/2), (pos - (s[lane%3]-int_bound)))
            ellipse_angle = 90
            ellipse_colour = el_color
            

    elif lane == 5:
        if pos <= 0:
            ellipse_pos = ((3*B/2), (pos))
            ellipse_angle = 90
            ellipse_colour = el_color

        elif pos < s[lane%3] + veh_object.length:
            ellipse_pos = ((3*B/2)+(pos/np.sqrt(2)), ((pos/np.sqrt(2))))
            ellipse_angle = 45
            ellipse_colour = el_color
            

        else:
            ellipse_pos = (pos ,(5*B/2))
            ellipse_angle = 0
            ellipse_colour = el_color
            

    elif lane == 11:
        if pos <= 0:
            ellipse_pos = ((5*B/2), -(pos-int_bound))
            ellipse_angle = -90
            ellipse_colour = el_color
            

        elif pos < s[lane%3] + veh_object.length:
            ellipse_pos = ((5*B/2)-((pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound)))
            ellipse_angle = -135
            ellipse_colour = el_color
            

        else:
            ellipse_pos = ( -(pos - (s[lane%3])) ,(3*B/2))
            ellipse_angle = 180
            ellipse_colour = el_color
            

    else:
        print("error!")
        print(a)
        
        

    ellipse = Ellipse(xy=ellipse_pos, width=ellipse_width, height=ellipse_height, angle=ellipse_angle, color=ellipse_colour)

    plt.gca().add_patch(ellipse)
    number_to_print = 20

    text_pos = (ellipse_pos[0] + ellipse_width / 2, ellipse_pos[1])
    if tc_flag != None:
        plt.text(text_pos[0] + 0.1, text_pos[1], (str(tc_flag)+" "+str(veh_id)), fontsize=12, color='red')
    elif tc_flag == None:     
        plt.text(text_pos[0] + 0.1, text_pos[1], (str(veh_id)), fontsize=12, color='red')
    #plt.text(3.5, 4, str(number_to_print), fontsize=12)
    #plt.text(4, 5, "Simulated", fontsize=20)


    return


def func(heuristic):

    final = []

    arr_ =  0.1
    sim_ = 1

    if heuristic == "RL":
        #../data_version/version_{int(version)}/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/
        file_path = f"../data_version/version_2/arr_0.1/test_homo_stream/train_sim_1/train_iter_500/pickobj_sim_51"
    else:
        file_path = f"../data/{heuristic}/arr_{arr_}/pickobj_sim_{sim_}"

    for c in os.listdir(f"{file_path}"):
        file = open(f"{file_path}/{c}", "rb")
        object_file = pickle.load(file)
        file.close()
        final.append(object_file[int(c)])
        # print("last pos:", object_file[int(c)].p_traj[-1])


    tend = 299

    tfull = np.arange(0, tend, round(1.0, 1))

    xh = np.arange(int_start,-int_start+int_bound,0.1)
    xv = np.arange(0,int_bound,0.1)
    num_veh = 0
    global_dict ={}
    over_dict ={}
    ovr_lane =[]

    #for t in tfull[:1]:
    for t in tfull:
        t = round(t,1)

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111)

        plt.xlim(int_start,-int_start + (int_bound))
        plt.ylim(int_start, -int_start + (int_bound))
        plt.fill_between(xh, 0 , int_bound , facecolor='grey',zorder = 0)
        plt.fill_between(xv, int_start, -int_start + (int_bound) , facecolor='grey',zorder =0)

        plt.text(4.5, 8, f"time: {t}s",fontsize=20)#,color='b')
        plt.text(4, 7, f"Number of robots crossed:",fontsize=20)
        plt.text(4, 6, f"{num_veh} robots",fontsize=20)

        print(f"{t} of {tend} for {heuristic}", end="\r")
        for i in final:
            # if i.lane != 5:
            #     continue
            #print(f'pos:{i.p_traj},id:{i.id}')
            if (i.p_traj[-1] < i.intsize + i.length):
            #    print(f"id: {i.id}, last time: {i.t_ser[-1]}, last pos: {i.p_traj[-1]}, len of t_ser and p_traj: {len(i.t_ser)}, {len(i.p_traj)}, veh.stime: {i.stime}")
                pass #
                #print(f"id: {i.id}, time: {i.t_ser}, last pos: {i.p_traj}, len of t_ser and p_traj: {len(i.t_ser)}, {len(i.p_traj)}, veh.stime: {i.stime}")
                # print(f'inter:{i.intsize}, lane{i.lane}')
                # exit()
                

            try:
                t_ind = functions.find_index(i, t)
                # if i.id == 0: print(f'\n out check : time:{t},temp:{t_ind},{i.p_traj[t_ind]-(i.length/2)}')

                if not (t_ind == None):
                    pass
                    # if i.id==0 and t == 27: 
                    #     print(f'*************, pos{i.p_traj[t_ind]},{i.p_traj[t_ind]-(i.length/2)}')
                    #     exit()


                    plot_ellipse(i, i.id, (i.p_traj[t_ind]-(i.length/2)), i.lane, i.curr_set[t], i.global_sig, i.sig[t], i.pos_status, t, i.p_traj[t_ind]) #i.tc_flag_time[t])
                    if i.global_sig[t] != None: global_dict = i.global_sig[t]
                    if i.ovr_stat[t] != None:ovr_lane = i.ovr_stat[t]
                    if i.over_sig[t] != None: over_dict = i.over_sig[t]
                    # if i.global_sig[t] != None:ovr_lane = i.ovr_stat[t]
                    id = i.id
                    #print(i.global_sig, "signal value :" ,i.global_sig_val)
                    #exit()


                    # if t_ind == 0:
                    #     num_veh +=1

                    if (i.p_traj[t_ind] > i.intsize + i.length) and (i.p_traj[t_ind-1] <= i.intsize + i.length):
                        num_veh +=1

                
            except KeyError as e:
                print(f"Error!: {e}")
                pass
        ##### plot signals ######
        
        #print(f'{type(data_file.phase_dict.values())})
        if global_dict == None or ovr_lane == None or over_dict == None : 
            print(f'id:{id},time:{t}')
            exit()

        plt.text(3, 5.5, f"sig t:{global_dict}",fontsize=12)
        plt.text(3, 5, f"ovr sig t:{over_dict}",fontsize=12)
        plt.text(3, 4.5, f"ovr stat t:{ovr_lane}",fontsize=12)
        #plt.text(4, 5, f"{ovr_lane} OVR ",fontsize=12)

        file_name = None

        if t < 9:
            N = 2
            file_name = str(t).zfill(N + len(str(t+1)))

        elif t < 99:
            N = 1
            file_name = str(t).zfill(N + len(str(t+1)))

        else:
            file_name = str(t)

        savestring = f'./xp_{heuristic}/' + file_name + '.png'
        plt.savefig(savestring)
        fig.clf()
        plt.close(fig)


if __name__ == "__main__":

    list_of_heuristics = [f"RL"] #, f"fifo"] # [f"time_to_react", "fifo", "dist_react_time", "conv_dist_react", "RL"]

    pool = Pool(8)
    pool.map(func, list_of_heuristics)
