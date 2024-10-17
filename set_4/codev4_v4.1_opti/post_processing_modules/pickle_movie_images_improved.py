import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.transforms import Affine2D
import matplotlib as mpl
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

# mpl.rc('font',family='Times New Roman')

from multiprocessing import Pool

L = 4.5/3  

len_veh = data_file.L# Length of vehicle

int_bound = data_file.int_bound  # size of intersection
int_start = data_file.int_start[0] # start of intersection system
p_init = int_start  # spawn_position

dt = round(data_file.dt,1)  # step size
# T_sc = 30  # scheduling time
# T_ps = 20  # prescheduling timeb
# T_h = 20
# # N = 150  # Number of time steps

# tend = 0 + 200  # End of simulation time
# t_opti = 3  # optimization intervals
# t_O = 3
# t_rem = 80
# t_rem_inc = 40
# branches = [0,1,2,3,4,5,6,7,8,9,10,11]
# arr_rate = 0.5

# lasttime = 0

# W_t = 3
# W_comf = 10
# W_pos = 1
# W_v = 1

# unassigned = []
# assigned = []
# ids = 0
# ts = []
# V_array = []
# full_opti = []

# vm = 11.11
# um = 3

# V_dict = {}
# #Vnd = {}
# blist = []
# klist = []

# novinsched = []

s = data_file.intersection_path_length # [data_file.int_bound/(4*np.sqrt(2)), data_file.int_bound, (5*data_file.int_bound)/(4*np.sqrt(2))] 

B = data_file.B

L = data_file.L

def plot_ellipse(veh_object, veh_id, center_pos_on_lane, lane, flag=False):

    veh_plot_len = veh_object.length/2
    veh_plot_width = B/4
    circle_rad = B/4
    pos = round(center_pos_on_lane, 2)

    tex_ = ""

    ellipse_pos = None
    ellipse_width = veh_plot_len
    ellipse_height = veh_plot_width
    ellipse_angle = None
    ellipse_colour = None
    text_pos = None
    text_to_print = str(veh_id)

    circle_pos = None
    circle_offset = veh_plot_len/2


    wheel_base_half_length = veh_plot_len/2.5
    wheel_base_half_width = veh_plot_width/2.5

    wheel_1_pos = None
    wheel_2_pos = None
    wheel_3_pos = None
    wheel_4_pos = None

    wheel_angle = None
    wheel_width = veh_plot_len/5
    wheel_height = wheel_width/2
    wheel_colour = 'k'

    wheel_x_offset = None
    wheel_y_offset = None



    if lane == 1:
        ellipse_pos = (pos, (7*B/2))
        ellipse_angle = 0
        ellipse_colour = 'b'

        circle_pos = np.asarray(ellipse_pos) + np.asarray([-circle_offset, 0])

        wheel_x_offset = wheel_base_half_length
        wheel_y_offset = wheel_base_half_width

        


    elif lane == 4:
        ellipse_pos = (B/2, pos)
        ellipse_angle = 90
        ellipse_colour = 'b'

        circle_pos = np.asarray(ellipse_pos) + np.asarray([0, -circle_offset])

        wheel_x_offset = -wheel_base_half_width
        wheel_y_offset = wheel_base_half_length
        
        
    elif lane == 7:
        ellipse_pos = (-(-int_bound + pos), B/2)
        ellipse_angle = 180
        ellipse_colour = 'b'

        circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset, 0])

        wheel_x_offset = -wheel_base_half_length
        wheel_y_offset = -wheel_base_half_width

        

    elif lane == 10:
        ellipse_pos = ((7*B/2), -(pos-int_bound))
        ellipse_angle = -90
        ellipse_colour = 'b'

        circle_pos = np.asarray(ellipse_pos) + np.asarray([0, circle_offset])

        wheel_x_offset = wheel_base_half_width
        wheel_y_offset = -wheel_base_half_length
        
        

    elif lane == 2:
        if pos <= 0:
            ellipse_pos = (pos, (5*B/2))
            ellipse_angle = 0
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([-circle_offset, 0])

            wheel_x_offset = wheel_base_half_length
            wheel_y_offset = wheel_base_half_width


            
            
        elif pos < s[lane%3]:
            ellipse_pos = (((pos/np.sqrt(2))), (5*B/2)-(pos/np.sqrt(2)))
            ellipse_angle = -45
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset/np.sqrt(2), circle_offset/np.sqrt(2)])


            wheel_x_offset = wheel_base_half_length/np.sqrt(2)
            wheel_y_offset = wheel_base_half_width/np.sqrt(2)

            
            
        else:
            ellipse_pos = ((5*B/2), -(pos - (s[lane%3])))
            ellipse_angle = -90
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([0, circle_offset])

            wheel_x_offset = wheel_base_half_width
            wheel_y_offset = -wheel_base_half_length
            
            

    elif lane == 8:
        if pos <= 0:
            ellipse_pos = (-(pos-int_bound), (3*B/2))
            ellipse_angle = 180
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset, 0])

            wheel_x_offset = -wheel_base_half_length
            wheel_y_offset = -wheel_base_half_width

            
            
        elif pos < s[lane%3]:
            ellipse_pos = (-((pos/np.sqrt(2)-int_bound)), (3*B/2) + (pos/np.sqrt(2)))
            ellipse_angle = 135
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset/np.sqrt(2), -circle_offset/np.sqrt(2)])

            wheel_x_offset = -wheel_base_half_length/np.sqrt(2)
            wheel_y_offset = -wheel_base_half_width/np.sqrt(2)

            
            
        else:
            ellipse_pos = ((3*B/2), (pos - (s[lane%3]-int_bound)))
            ellipse_angle = 90
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([0, -circle_offset])

            wheel_x_offset = -wheel_base_half_width
            wheel_y_offset = wheel_base_half_length
            
            

    elif lane == 5:
        if pos <= 0:
            ellipse_pos = ((3*B/2), (pos))
            ellipse_angle = 90
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([0, -circle_offset])

            wheel_x_offset = -wheel_base_half_width
            wheel_y_offset = wheel_base_half_length

            

        elif pos < s[lane%3]:
            ellipse_pos = ((3*B/2)+(pos/np.sqrt(2)), ((pos/np.sqrt(2))))
            ellipse_angle = 45
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([-circle_offset/np.sqrt(2), -circle_offset/np.sqrt(2)])

            wheel_x_offset = -wheel_base_half_length/np.sqrt(2)
            wheel_y_offset = wheel_base_half_width/np.sqrt(2)

            
            

        else:
            ellipse_pos = (pos ,(5*B/2))
            ellipse_angle = 0
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([-circle_offset, 0])

            wheel_x_offset = wheel_base_half_length
            wheel_y_offset = wheel_base_half_width

            
            

    elif lane == 11:
        if pos <= 0:
            ellipse_pos = ((5*B/2), -(pos-int_bound))
            ellipse_angle = -90
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([0, circle_offset])

            wheel_x_offset = wheel_base_half_width
            wheel_y_offset = -wheel_base_half_length

            
            

        elif pos < s[lane%3]:
            ellipse_pos = ((5*B/2)-((pos/np.sqrt(2)) ), -((pos/np.sqrt(2)-int_bound)))
            ellipse_angle = -135
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset/np.sqrt(2), circle_offset/np.sqrt(2)])

            wheel_x_offset = wheel_base_half_length/np.sqrt(2)
            wheel_y_offset = -wheel_base_half_width/np.sqrt(2)

            
            

        else:
            ellipse_pos = ( -(pos - (s[lane%3])) ,(3*B/2))
            ellipse_angle = 180
            ellipse_colour = 'g'

            circle_pos = np.asarray(ellipse_pos) + np.asarray([circle_offset, 0])

            wheel_x_offset = -wheel_base_half_length
            wheel_y_offset = -wheel_base_half_width


            
            

    else:
        print("error!")
        print(a)


    wheel_base_half_length = veh_plot_len/2.5

    # wheel_1_pos = np.asarray(ellipse_pos) + np.asarray([wheel_x_offset, wheel_y_offset])
    # wheel_2_pos = np.asarray(ellipse_pos) + np.asarray([wheel_x_offset, -wheel_y_offset])
    # wheel_3_pos = np.asarray(ellipse_pos) + np.asarray([-wheel_x_offset, -wheel_y_offset])
    # wheel_4_pos = np.asarray(ellipse_pos) + np.asarray([-wheel_x_offset, wheel_y_offset])

    wheel_angle = ellipse_angle
    wheel_colour = 'k'
    
    # trans = Affine2D().rotate_around(ellipse_pos[0], ellipse_pos[0], np.deg2rad(wheel_angle))

    ellipse = Ellipse(xy=ellipse_pos, width=ellipse_width, height=ellipse_height, angle=ellipse_angle, color=ellipse_colour)

    ellipse.set_edgecolor('k')

    # print(ellipse.get_edgecolor())

    # circle = Circle(xy=circle_pos, radius=circle_rad,  color=ellipse_colour)

    # wheel_1 = Rectangle(xy=wheel_1_pos, width=wheel_width, height=wheel_height, color=wheel_colour)
    # wheel_1.set_transform(trans + plt.gca().transData)

    # wheel_2 = Rectangle(xy=wheel_2_pos, width=wheel_width, height=wheel_height, color=wheel_colour)
    # wheel_2.set_transform(trans + plt.gca().transData)

    # wheel_3 = Rectangle(xy=wheel_3_pos, width=wheel_width, height=wheel_height, color=wheel_colour)
    # wheel_3.set_transform(trans + plt.gca().transData)

    # wheel_4 = Rectangle(xy=wheel_4_pos, width=wheel_width, height=wheel_height, color=wheel_colour)
    # wheel_4.set_transform(trans + plt.gca().transData)

    plt.gca().add_patch(ellipse)

    number_to_print = 20

    
    text_pos = (ellipse_pos[0] + ellipse_width / 2, ellipse_pos[1])
    plt.text(text_pos[0] + 0.2, text_pos[1], str(number_to_print), fontsize=12, color='red')
    #plt.text(3.5, 4, str(number_to_print), fontsize=12)
    plt.text(4, 5, "Simulated", fontsize=20)






    # plt.gca().add_patch(circle)

    # plt.gca().add_patch(wheel_1)
    # plt.gca().add_patch(wheel_2)
    # plt.gca().add_patch(wheel_3)
    # plt.gca().add_patch(wheel_4)

    return



def func(args_inp):

    heuristic = args_inp[0]
    comparison = args_inp[1]
    arr_ = args_inp[2]
    sim_ = args_inp[3]

    final = []

    interpolation_res = 0.5

    tend = 300

    tfull = np.arange(0, tend, round(interpolation_res, 1))

    dict_arr_rate = {}

    arr_rate_pos = {}

    for lane in data_file.lanes:
        dict_arr_rate[lane] = {}

        for t in range(18): # tfull:
            dict_arr_rate[lane][t] = None

    for lane in data_file.lanes:
        if lane == 1:
            arr_rate_pos[lane] = (-8.5, 2.25)

        elif lane == 2:
            arr_rate_pos[lane] = (-8.5, 1.55)

        elif lane == 4:
            arr_rate_pos[lane] = (0, -8)

        elif lane == 5:
            arr_rate_pos[lane] = (1, -8)

        elif lane == 7:
            arr_rate_pos[lane] = (10, 0.15)

        elif lane == 8:
            arr_rate_pos[lane] = (10, 1)

        elif lane == 10:
            arr_rate_pos[lane] = (1.5, 10.25)

        elif lane == 11:
            arr_rate_pos[lane] = (2.5, 10.25)

    heuristic_input_folder = {'fifo':'fifo', 'cdt':'conv_dist_react', 'pdt':'dist_react_time', 'ttr': 'time_to_react'}

    if heuristic == "RL":
        if arr_ == None:
            # file_path = f"../data/arr_{arr_}/test_homo_stream/train_sim_1/train_iter_5000/pickobj_sim_{sim_}"
            file_path = f"../data/pickobj_{comparison}"

        else:
            file_path = f"../data/arr_{arr_}/test_homo_stream/train_sim_1/train_iter_100000/pickobj_sim_{sim_}"

    else:
        file_path = f"../data/{heuristic_input_folder[heuristic]}/arr_{arr_}/pickobj_sim_{sim_}"


    num_of_veh = 0
    for c in os.listdir(f"{file_path}"):
        file = open(f"{file_path}/{c}", "rb")
        object_file = pickle.load(file)
        file.close()
        object_file[int(c)].interpolated_ptraj = []
        object_file[int(c)].interpolated_vtraj = []
        object_file[int(c)].interpolated_utraj = []
        object_file[int(c)].interpolated_tser = []

        if dict_arr_rate[object_file[int(c)].lane][int(object_file[int(c)].sp_t / 30)] == None:
            # print(f"arr_rate of {c} before update: {dict_arr_rate[object_file[int(c)].lane][int(object_file[int(c)].sp_t / 30)]}")
            dict_arr_rate[object_file[int(c)].lane][int(object_file[int(c)].sp_t / 30)] = round(object_file[int(c)].arr, 2)
            # print(f"arr_rate of {c} after update: {dict_arr_rate[object_file[int(c)].lane][int(object_file[int(c)].sp_t / 30)]}\n")

        num_of_veh += 1

        # if object_file[int(c)].lane == 1:
        #     print(f"num_of_veh:{num_of_veh}\tarr_rate of robot {c}: {round(object_file[int(c)].arr, 2)}\tarr_rate in dict: {dict_arr_rate[object_file[int(c)].lane][int(object_file[int(c)].sp_t / 30)]}\tspawning time: {round(object_file[int(c)].sp_t, 1)}\tk_sp-t:{int(object_file[int(c)].sp_t / 30)}\tlane:{object_file[int(c)].lane}")
        
        for t_ind, t_ in enumerate(object_file[int(c)].t_ser):
            object_file[int(c)].interpolated_tser.append(object_file[int(c)].t_ser[t_ind])
            object_file[int(c)].interpolated_ptraj.append(object_file[int(c)].p_traj[t_ind])
            object_file[int(c)].interpolated_vtraj.append(object_file[int(c)].v_traj[t_ind])
            object_file[int(c)].interpolated_utraj.append(object_file[int(c)].u_traj[t_ind])
            
            for interpol_t_ind in range(1, int(data_file.dt/interpolation_res)):
                interpolated_time = round(object_file[int(c)].t_ser[t_ind] + (interpol_t_ind*interpolation_res), 1)
                interpolated_pos = object_file[int(c)].p_traj[t_ind] + (object_file[int(c)].v_traj[t_ind]*(interpol_t_ind*interpolation_res)) + (0.5 * object_file[int(c)].u_traj[t_ind] * ((interpol_t_ind*interpolation_res)**2))
                interpolated_vel = object_file[int(c)].v_traj[t_ind] + (object_file[int(c)].u_traj[t_ind]*(interpol_t_ind*interpolation_res))
                interpolated_acc = object_file[int(c)].u_traj[t_ind]

                object_file[int(c)].interpolated_tser.append(interpolated_time)
                object_file[int(c)].interpolated_ptraj.append(interpolated_pos)
                object_file[int(c)].interpolated_vtraj.append(interpolated_vel)
                object_file[int(c)].interpolated_utraj.append(interpolated_acc)


        while (object_file[int(c)].interpolated_ptraj[-1] <=  (2*object_file[int(c)].intsize + 2*object_file[int(c)].length + (-2*object_file[int(c)].p0))):

            object_file[int(c)].interpolated_ptraj.append(object_file[int(c)].interpolated_ptraj[-1] + (object_file[int(c)].interpolated_vtraj[-1]*(interpolation_res)))
            object_file[int(c)].interpolated_vtraj.append(object_file[int(c)].interpolated_vtraj[-1])
            object_file[int(c)].interpolated_utraj.append(0)
            object_file[int(c)].interpolated_tser.append(round(object_file[int(c)].interpolated_tser[-1] + interpolation_res, 1))




        final.append(object_file[int(c)])
        # print("last pos:", object_file[int(c)].p_traj[-1])

    xh = np.arange(int_start,-int_start+int_bound+0.1,0.1)
    xv = np.arange(0,int_bound,0.1)
    num_veh_crossed = 0
    num_veh_entered = 0

    file_num = 0

    # for lane in data_file.lanes:
    #     if lane not in [0, 3, 6, 9]:
    #         print(f"lane:{lane}\tarrival_rates:{[dict_arr_rate[lane][_] for _ in dict_arr_rate[lane].keys()]}")


    for t in tfull:
        t = round(t,1)

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111)

        plt.xlim(int_start,-int_start + (int_bound))
        plt.ylim(int_start, -int_start + (int_bound))
        plt.fill_between(xh, 0 , int_bound , facecolor='grey',zorder = 0)
        plt.fill_between(xv, int_start, -int_start + (int_bound) , facecolor='grey',zorder =0)

        # plt.plot([0, 0, int_bound, int_bound], [0, int_bound, int_bound, 0], color='lightsteelblue')

        plt.plot([int_start, 0], [int_bound/2, int_bound/2], color='lightsteelblue')

        plt.plot([int_start, 0], [int_bound/4, int_bound/4], color='lightsteelblue', linestyle='dashed')

        plt.plot([int_start, 0], [3*int_bound/4, 3*int_bound/4], color='lightsteelblue', linestyle='dashed')



        plt.plot([int_bound/2, int_bound/2], [int_start, 0], color='lightsteelblue')

        plt.plot([int_bound/4, int_bound/4], [int_start, 0], color='lightsteelblue', linestyle='dashed')

        plt.plot([3*int_bound/4, 3*int_bound/4], [int_start, 0], color='lightsteelblue', linestyle='dashed')



        plt.plot([int_bound/2, int_bound/2], [-int_start+int_bound, int_bound], color='lightsteelblue')

        plt.plot([int_bound/4, int_bound/4], [-int_start+int_bound, int_bound], color='lightsteelblue', linestyle='dashed')

        plt.plot([3*int_bound/4, 3*int_bound/4], [-int_start+int_bound, int_bound], color='lightsteelblue', linestyle='dashed')



        plt.plot([-int_start+int_bound, int_bound], [int_bound/2, int_bound/2], color='lightsteelblue')

        plt.plot([-int_start+int_bound, int_bound], [int_bound/4, int_bound/4], color='lightsteelblue', linestyle='dashed')

        plt.plot([-int_start+int_bound, int_bound], [3*int_bound/4, 3*int_bound/4], color='lightsteelblue', linestyle='dashed')


        plt.text(5, 7, f"time: {t}s",fontsize=20)#,color='b')
        plt.text(3.5, 6, f"Number of robots crossed: {num_veh_crossed}",fontsize=20)
        #plt.text(6, 6, f"{num_veh}",fontsize=20)

        if not (comparison == f"comparison_algos"):
            plt.text(-6, 5, f"Different arrival rates on\ndifferent lanes\n(non-homogeneous traffic)\nvarying with time (shown at\nstart of lane --robot/lane/s)", fontsize=20)

        else:

            true_arr_rate = 0 if (num_veh_entered == 0) else round(num_veh_entered/8/t, 2)

            plt.text(-6, 7, f"Simulated arrival rate:", fontsize=20)
            plt.text(-5.5, 6.5, f"{round(arr_, 2)} robot/lane/s", fontsize=20)

            plt.text(-5.5, 5.5, f"True arrival rate:", fontsize=20)
            plt.text(-5.25, 5, f"{round(true_arr_rate, 4)} robot/lane/s", fontsize=20)

            # plt.text(-7, 5, f"True arrival rate: {round(true_arr_rate, 4)} robot/lane/s", fontsize=20)


        for lane in data_file.lanes:
            if lane in [0, 3, 6, 9]:
                continue

            if not (comparison == f"comparison_algos"):
                if dict_arr_rate[lane][int(t/30)] == None:
                    plt.text(arr_rate_pos[lane][0], arr_rate_pos[lane][1], f"0",fontsize=20)

                else:
                    plt.text(arr_rate_pos[lane][0], arr_rate_pos[lane][1], f"{dict_arr_rate[lane][int(t/30)]}",fontsize=20)

        print(f"{t} of {tend} for {heuristic}", end="\r")
        for i in final:
            # if i.lane != 5:
            #     continue

            try:
                t_ind = i.interpolated_tser.index(round(t, 1)) if (t in i.interpolated_tser) else None

                if not (t_ind == None):
                    pass

                    plot_ellipse(i, i.id, (i.interpolated_ptraj[t_ind]-(i.length/2)), i.lane)
                    
                    if t_ind == 0:
                        num_veh_entered +=1

                    if (i.interpolated_ptraj[t_ind] > i.intsize) and (i.interpolated_ptraj[t_ind-1] <= i.intsize):
                        num_veh_crossed +=1

                
            except KeyError as e:
                print(f"Error!: {e}")
                pass

        file_num += 1

        # if t < 9:
        #     N = 2
        #     file_name = str(int(t/interpolation_res)).zfill(N + len(str(int(t/interpolation_res)+1)))

        # elif t < 99:
        #     N = 1
        #     file_name = str(int(t/interpolation_res)).zfill(N + len(str(int(t/interpolation_res)+1)))

        # else:
        #     file_name = str(int(t/interpolation_res))

        if file_num <= 9:
            file_name = f"00{int(file_num)}"

        elif file_num <= 99:
            file_name = f"0{int(file_num)}"

        else:
            file_name = f"{int(file_num)}"


        if comparison == f"comparison_algos":
            save_file_path = f"./comparison/{comparison}/arr_{arr_}/{heuristic}/{file_name}.png"

        else:
            save_file_path = f"./comparison/RL_{comparison}/{file_name}.png"

        save_file_path = f'./xp_{heuristic}/' + file_name + '.png'
        plt.savefig(save_file_path)
        fig.clf()
        plt.close(fig)


if __name__ == "__main__":

    list_of_heuristics = [f"RL"] #, f"fifo", f"ttr", f"cdt", f"pdt"] #, f"fifo"] # [f"time_to_react", f"fifo", f"dist_react_time", f"conv_dist_react", f"RL"]

    list_of_comparisons = [f"_"] #["comparison_algos"] # "changing_arr_rate",  

    # func(list_of_heuristics[-1])

    args = []

    for heu in list_of_heuristics:
        for comp in list_of_comparisons:
            if (heu == f"fifo") and ((comp == f"non-homo") or ((comp == f"changing_arr_rate"))):
                continue


            if (comp == f"non-homo") or (comp == f"changing_arr_rate"):
                args.append([heu, comp, None, None])

            else:
                for arr, sim in zip([0.1], [1]): # [0.11], [5]
                    args.append([heu, comp, arr, sim])

    pool = Pool(8)
    pool.map(func, args)