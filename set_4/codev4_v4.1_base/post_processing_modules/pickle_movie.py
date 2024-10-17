import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
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

def plot_circle(veh_object, veh_id, center_pos_on_lane, lane, flag=False):

    L = veh_object.length/3
    pos = center_pos_on_lane

    tex_ = ""

    if lane == 1:
        circle = plt.Circle((pos-(L/2), (7*B/2)), L/2, color = 'b')

        if flag:
            tex_ = plt.text(pos-(L/2), (7*B/2), str(veh_id),fontsize=18,color='b')

    elif lane == 4:
        circle = plt.Circle((B/2, (pos-(L/2))) , L/2, color = 'b')
        if flag:
            tex_ = plt.text(B/2, (pos-(L/2)) , str(veh_id),fontsize=18,color='b')

    elif lane == 7:
        circle = plt.Circle((-(-int_bound + pos-(L/2)), B/2), L/2, color = 'b')
        if flag:
            tex_ = plt.text(-(-int_bound + pos-(L/2)), B/2 , str(veh_id),fontsize=18,color='b')

    elif lane == 10:
        circle = plt.Circle(((7*B/2), -(pos-(L/2)-int_bound)) , L/2, color = 'b')
        if flag:
            tex_ = plt.text((7*B/2), -(pos-(L/2)-int_bound), str(veh_id),fontsize=18,color='b')

    # elif lane == 0:
    #     if pos <= 0:
    #         circle = plt.Circle((pos-(L/2), (7*int_bound/8)), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text(pos-(L/2), (7*int_bound/8), str(veh_id),fontsize=12,color='b')

    #     elif pos <= s[lane%3]:
    #         circle = plt.Circle((((pos/np.sqrt(2))), (7*int_bound/8) + (pos/np.sqrt(2))), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text(((pos/np.sqrt(2))), (7*int_bound/8) + (pos/np.sqrt(2)), str(veh_id),fontsize=12,color='b')

    #     else:
    #         circle = plt.Circle(((int_bound/8), (pos - (s[lane%3]) - (L/2) + int_bound)), L/2 , color = 'r')
    #         if flag:
    #             tex_ = plt.text((int_bound/8), (pos - (s[lane%3]) - (L/2) + int_bound), str(veh_id),fontsize=12,color='b')


    elif lane == 2:
        if pos <= 0:
            circle = plt.Circle((pos-(L/2), (5*B/2)), L/2, color = 'g')
            if flag:
                tex_ = plt.text(pos-(L/2), (5*B/2), str(veh_id),fontsize=18,color='b')

        elif pos < s[lane%3]:
            circle = plt.Circle((((pos/np.sqrt(2))), (5*B/2) - (pos/np.sqrt(2)) ), L/2, color = 'g')
            if flag:
                tex_ = plt.text(((pos/np.sqrt(2))), (5*B/2) - (pos/np.sqrt(2)), str(veh_id),fontsize=18,color='b')

        else:
            circle = plt.Circle(((5*B/2), -(pos - (s[lane%3])  - (L/2))), L/2 , color = 'g')
            if flag:
                tex_ = plt.text((5*B/2), -(pos - (s[lane%3])  - (L/2)), str(veh_id),fontsize=18,color='b')

    # elif lane == 6:
    #     if pos <= 0:
    #         circle = plt.Circle((-(pos-(L/2)-int_bound), (int_bound/8)), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text(pos-(L/2), (5*int_bound/8), str(veh_id),fontsize=12,color='b')

    #     elif pos <= s[lane%3]:
    #         circle = plt.Circle(-(pos-(L/2)-int_bound), (int_bound/8), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text(-(pos-(L/2)-int_bound), (int_bound/8), str(veh_id),fontsize=12,color='b')

    #     else:
    #         circle = plt.Circle(((7*int_bound/8), -(pos - (s[lane%3]) - (L/2))), L/2 , color = 'r')
    #         if flag:
    #             tex_ = plt.text((7*int_bound/8), -(pos - (s[lane%3]) - (L/2)), str(veh_id),fontsize=12,color='b')


    elif lane == 8:
        if pos <= 0:
            circle = plt.Circle((-(pos-(L/2)-int_bound), (3*B/2)), L/2, color = 'g')
            if flag:
                tex_ = plt.text(-(pos-(L/2)-int_bound), (3*B/2), str(veh_id),fontsize=18,color='b')

        elif pos < s[lane%3]:
            circle = plt.Circle((-((pos/np.sqrt(2)-int_bound)), (3*B/2) + (pos/np.sqrt(2)) ), L/2, color = 'g')
            if flag:
                tex_ = plt.text(-((pos/np.sqrt(2)-int_bound)), (3*B/2) + (pos/np.sqrt(2)) , str(veh_id),fontsize=18,color='b')

        else:
            circle = plt.Circle(((3*B/2), (pos - (s[lane%3]-int_bound) - (L/2))), L/2 , color = 'g')
            if flag:
                tex_ = plt.text((3*B/2), (pos - (s[lane%3]-int_bound) - (L/2)), str(veh_id),fontsize=18,color='b')


    # elif lane == 3:
    #     if pos <= 0:
    #         circle = plt.Circle(((int_bound/8), (pos-(L/2))), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text((int_bound/8), (pos-(L/2)), str(veh_id),fontsize=12,color='b')


    #     elif pos <= s[lane%3]:
    #         circle = plt.Circle(((int_bound/8) - ( (pos/np.sqrt(2)) ), ((pos/np.sqrt(2)) )), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text((int_bound/8) - ( (pos/np.sqrt(2)) ), ((pos/np.sqrt(2)) ), str(veh_id),fontsize=12,color='b')


    #     else:
    #         circle = plt.Circle(( -(pos - (s[lane%3] ) - L/2) ,(int_bound/8)), L/2 , color = 'r')
    #         tex_ = plt.text(-(pos - (s[lane%3] ) - L/2) ,(int_bound/8), str(veh_id),fontsize=12,color='b')

    # elif lane == 9:
    #     if pos <= 0:
    #         circle = plt.Circle(((7*int_bound/8), -(pos-(L/2)-int_bound)), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text((7*int_bound/8), -(pos-(L/2)-int_bound), str(veh_id),fontsize=12,color='b')


    #     elif pos <= s[lane%3]:
    #         circle = plt.Circle(((7*int_bound/8) + ( (pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound))), L/2, color = 'r')
    #         if flag:
    #             tex_ = plt.text((7*int_bound/8) + ( (pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound)), str(veh_id),fontsize=12,color='b')

    #     else:
    #         circle = plt.Circle(( (pos - (s[lane%3] -int_bound) - L/2) ,(7*int_bound/8)), L/2 , color = 'r')
    #         if flag:
    #             tex_ = plt.text( (pos - (s[lane%3] -int_bound) - L/2) ,(7*int_bound/8), str(veh_id),fontsize=12,color='b')

    elif lane == 5:
        if pos <= 0:
            circle = plt.Circle(((3*B/2), (pos-(L/2))), L/2, color = 'g')
            if flag:
                tex_ = plt.text( (3*B/2), (pos-(L/2)), str(veh_id),fontsize=18,color='b')


        elif pos < s[lane%3]:
            circle = plt.Circle(((3*B/2) + ( (pos/np.sqrt(2)) ), ((pos/np.sqrt(2)) - (L))), L/2, color = 'g')
            if flag:
                tex_ = plt.text( (3*B/2) + ( (pos/np.sqrt(2)) ), ((pos/np.sqrt(2)) ), str((round(pos,2))),fontsize=18,color='b')

        else:
            circle = plt.Circle(( ((pos) - L/2) ,(5*B/2)), L/2 , color = 'g')
            if flag:
                tex_ = plt.text( ((pos) - L/2) ,(5*B/2), str((round(pos,2))),fontsize=18,color='b')

    elif lane == 11:
        if pos <= 0:
            circle = plt.Circle(((5*B/2), -(pos-(L/2)-int_bound)), L/2, color = 'g')
            if flag:
                tex_ = plt.text( (5*B/2), -(pos-(L/2)-int_bound), str(veh_id),fontsize=18,color='b')

        elif pos < s[lane%3]:
            circle = plt.Circle(((5*B/2) - ( (pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound) )), L/2, color = 'g')
            if flag:
                tex_ = plt.text( (5*B/2) - ( (pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound) ), str(veh_id),fontsize=18,color='b')

        else:
            circle = plt.Circle(( -(pos - (s[lane%3]) - L/2) ,(3*B/2)), L/2 , color = 'g')
            if flag:
                tex_ = plt.text( -(pos - (s[lane%3]) - L/2) ,(3*B/2), str(veh_id),fontsize=18,color='b')

    else:
        print("error!")
        print(a)

    plt.gca().add_patch(circle)

    return circle, tex_


'''class Vehicle():
    def __init__(self,arr_rate):
        self.sp_t = np.random.poisson(arr_rate)
        # self.sp_t = .1
        self.p0 = p_init
        self.v0 = np.random.uniform(0, vm-1)
        self.u0 = np.random.uniform(-um+1, um-1)
        # branches = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#        branches = [1, 4, 7, 10]
        self.lane = np.random.choice(branches)
        # self.lane = 2
        self.p_traj = [self.p0]
        self.v_traj = [self.v0]
        self.u_traj = [self.u0]
        self.t_ser = [self.sp_t]
        self.ptime = 10000
        self.stime = 10000
        self.id = None
        self.finptraj = []
        self.finvtraj = []
        self.finutraj = []
        self.bindex = 0
        self.exittime = 10000
        self.ps_end = 0
        self.schedindex = 0
        colors = ['b', 'r', 'g', 'k', 'm', 'c', 'y']
        self.col = np.random.choice(colors)
        self.priority_index = 0
        self.w_f = 0
        self.incv = 0
        self.incomp = compdict[self.lane]
        self.intsize = s[self.lane%3]
        self.clrtime = 1000000
        self.actcost = 0
        self.comfcost = 0
        self.poscost = 0
        self.cpcost = 0
        self.demand = 0'''


def func(heuristic):

    final = []

    if heuristic == "RL":
        file_path = f"../data_version/version_2/arr_0.1/test_homo_stream/train_sim_1/train_iter_500/pickobj_sim_1"

    else:
        file_path = f"../data/{heuristic}/arr_0.1/pickobj_sim_6"

    for c in os.listdir(f"{file_path}"):
        file = open(f"{file_path}/{c}", "rb")
        object_file = pickle.load(file)
        file.close()
        final.append(object_file[int(c)])
        # print("last pos:", object_file[int(c)].p_traj[-1])


    tend = 500

    tfull = np.arange(0, tend, round(1.0, 1))

    xh = np.arange(int_start,-int_start+int_bound,0.1)
    xv = np.arange(0,int_bound,0.1)
    num_veh = 0

    for t in tfull:
        t = round(t,1)

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111)

        plt.xlim(int_start,-int_start + (int_bound))
        plt.ylim(int_start, -int_start + (int_bound))
        plt.fill_between(xh, 0 , int_bound , facecolor='grey',zorder = 0)
        plt.fill_between(xv, int_start, -int_start + (int_bound) , facecolor='grey',zorder =0)

        plt.text(4.5, 8, f"time: {t}s",fontsize=20)#,color='b')
        plt.text(4, 7, f"Average arrival rate:",fontsize=20)
        plt.text(4, 6, f"{'%0.4f' % (num_veh/(8*max(t, 1)))} robot/lane/s",fontsize=20)

        print(f"{t} of {tend} for {heuristic}", end="\r")
        for i in final:

            try:
                t_ind = functions.find_index(i, t)

                if not (t_ind == None):
                    pass

                    c, text_ = plot_circle(i, i.id, i.p_traj[t_ind], i.lane)


                    if t_ind == 0:
                        num_veh +=1

                        # if (i.p_traj[t_ind] > i.intsize) and (i.p_traj[t_ind-1] <= i.intsize):
                        #     num_veh +=1

                
            except KeyError:
                pass

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

    list_of_heuristics = [f"RL"] # [f"time_to_react", "fifo", "dist_react_time", "conv_dist_react", "RL"]

    pool = Pool(8)
    pool.map(func, list_of_heuristics)