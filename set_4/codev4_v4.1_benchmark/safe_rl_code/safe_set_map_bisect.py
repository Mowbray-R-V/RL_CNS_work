from utility import config,functions
import copy
import logging
import time
import numpy as np
 

def state_esti(veh, u_allow_min, u_allow_max, temp_ind, t_init, feas_stat, sig_traj, d, learning_flag):   

    X = []
    if u_allow_max != None and u_allow_min != None:
    
        if u_allow_max>veh.u_max or u_allow_min<veh.u_min or u_allow_max<veh.u_min or u_allow_min>veh.u_max: 
            print("in state esti","u_max",u_allow_max,"u_min",u_allow_min)
            assert False, 'safe set exceeds the physical limits'
        
        ################## override ###############
        ####veh.alpha = 0.5#np.random.random(1)[0]   #1
        assert 0<=veh.alpha <=1,f"alpha value: {veh.alpha}"
        assert 0<=veh.beta <=1,f"beta value: {veh.beta}"

        if veh.lane in veh.ovr_stat[t_init]:
            if veh.id in veh.ovr_stat[t_init][veh.lane]:
                assert veh.alpha == 1
                # assert False,'this block should not run'
                
        ###beta addition ####
        u_allow_min_beta =  (veh.u_min - u_allow_max)*veh.beta + u_allow_max

        assert u_allow_max >= u_allow_min and u_allow_max >= u_allow_min_beta

        veh.u_traj.append(copy.deepcopy((veh.alpha*(u_allow_max - u_allow_min_beta)) + u_allow_min_beta))   #veh.u_traj.append(copy.deepcopy((veh.alpha*(u_allow_max - u_allow_min)) + u_allow_min))

        assert veh.u_min <= ((veh.alpha * (u_allow_max - u_allow_min_beta)) + u_allow_min_beta) <= veh.u_max

        ##### linear mapping #####
        X.append(veh.p_traj[temp_ind])
        X.append(veh.v_traj[temp_ind])
        x_next = functions.compute_state(X, veh.u_traj[temp_ind],round( config.dt,1),veh.v_max)

        if not learning_flag and config.file_analysis:
            f = open("RL_actions.txt", "a")
            f.write(f'\n alpha:{veh.alpha}, beta:{veh.beta}, u_max_safe:{u_allow_max}, u_min_beta:{u_allow_min_beta}, u*:{veh.u_traj[-1]}')
            f.close()

        if x_next[1]>veh.v_max or x_next[1]<veh.v_min:

            assert False, f'clipping not working, veh_vmax:{veh.v_max}, veh_vmin :{veh.v_min} pos:{x_next[0]}, vel:{x_next[1]}, pos_vec:{veh.p_traj}, vel_vec:{veh.v_traj}, sig:{sig_traj}' 
            p_next1 =  x_next[0]
            v_next1 =  x_next[1]
            if round(p_next1, 4) > 0: assert sig_traj=='G' or (sig_traj=='R' and feas_stat == False), f"u_min: {u_allow_min}, u_max: {u_allow_max}, alpha: {veh.alpha}, veh.u: {(veh.alpha * (u_allow_max - u_allow_min) )+ u_allow_min},\
                ptraj: {veh.p_traj}\nvtraj: {veh.v_traj}, utraj: {veh.u_traj},set: {sig_traj}, ID:{veh.id}, sig_traj :{sig_traj}, feas_state:{feas_stat}, p_next:{p_next1}, v_pred:{ x_next[1]}, v_esti:{v_next1}, D_val:{d}, sig_traj:{veh.global_sig} "
            veh.p_traj.append(copy.deepcopy(p_next1))
            veh.v_traj.append(copy.deepcopy(v_next1))
            veh.t_ser.append(round((t_init + (round(config.dt,1))), 1))

        elif x_next[1]<=veh.v_max and  x_next[1]>=veh.v_min:   ### recent log: made it or
            if round(x_next[0], 4) > 0: assert sig_traj=='G' or (sig_traj=='R' and feas_stat == False),f" stat:{feas_stat} sig:{sig_traj}   u_min: {u_allow_min}, u_max: {u_allow_max}, alpha: {veh.alpha}, veh.u: {(veh.alpha * (u_allow_max - u_allow_min) )+ u_allow_min},\
                ptraj: {veh.p_traj}\n vtraj: {veh.v_traj}, utraj: {veh.u_traj}, pos_next:{x_next[0]},v_esti:{x_next[1]}, D_val:{d}  set: {set}, ID:{veh.id}, sig_traj:{veh.global_sig} "

            veh.p_traj.append(copy.deepcopy(x_next[0]))
            veh.v_traj.append(copy.deepcopy(x_next[1]))
            veh.t_ser.append(round((t_init + (config.dt)), 1))

        veh.u_safe_min[veh.t_ser[-2]] = u_allow_min
        veh.u_safe_max[veh.t_ser[-2]] = u_allow_max
        
        # veh.finptraj[veh.t_ser[-1]] = veh.p_traj[-1]
        # veh.finvtraj[veh.t_ser[-1]] = veh.v_traj[-1]
        # veh.finutraj[veh.t_ser[-2]] = veh.u_traj[-1]   


        #print(f't_ser:{veh.t_ser}')
        temp_ind = functions.find_index(veh, t_init)
        #print(f"cuur id: {veh.id}, current time: {t_init}")
        #print(f"cur pos: {veh.p_traj}, vel: {veh.v_traj},acc: {veh.u_traj}, t: {veh.t_ser}")

        assert (veh.t_ser[temp_ind+1])== t_init + (round(config.dt,1))
        
    elif  u_allow_max == None and u_allow_min == None: pass
    else: assert False        
    
    # if veh.id==1 or veh.id==3:
    #     f = open("debug.txt", "a")
    #     f.write(f'\n---set, ID:{veh.id} , umin:{u_allow_min}, umax:{u_allow_max}, alpha:{veh.alpha}, sig:{sig_traj} \n \
    #             pos:{veh.p_traj}, vel:{veh.v_traj},tser:{veh.t_ser}')
    #     f.close()
    
    return veh, feas_stat    


def acc_clip(veh, u_allow_min, temp_ind):
    
    if veh.v_traj[temp_ind] ==0:
        u_allow_min  = 0
    elif 0 < veh.v_traj[temp_ind] : pass
    elif veh.v_traj[temp_ind] < 0: assert False
    else: assert False, f'{veh.v_traj[temp_ind]}'
    # elif 0 < veh.v_traj[temp_ind] <= 0.1:
    #     u_allow_min  = -0.1

    return u_allow_min
 

def IEP_safety_def(veh, x_prime):
    
    status = None
    assert x_prime[1]<= veh.v_max and  x_prime[1]>= veh.v_min
    # assert x_prime[0]<= 0,f'x:{x_prime[0]}, pos_tra:{veh.p_traj}, LHS:{x_prime[1] ** 2}, RHS:{(2*x_prime[0]*veh.u_min)}' 
    
    # assert x_prime[1]>=0,f"IEP_ERROR,  vel:{ x_prime[1]}"
    if (x_prime[1] ** 2) <= (2*x_prime[0]*veh.u_min): 
        status = True
    elif (x_prime[1] ** 2) > (2*x_prime[0]*veh.u_min): 
        status = False
    else: 
        assert False,"IEP ERROR"
    
    return status



def RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt):
    
    ### position check in inter
    ### differnec small
    ### check if really veh before
    ###
    
    assert pre_ind_nxt!=None,"error in RES input"
    assert x_prime[1]<= veh.v_max and  x_prime[1]>= veh.v_min
    # assert x_prime[0]<= (veh.length + veh.intsize),f'pos:{x_prime[0]}'
    
    status = None
    if (_pre_v.p_traj[pre_ind_nxt] -  x_prime[0]) >= (config.L + max(0,(((_pre_v.v_traj[pre_ind_nxt] ** 2) - (x_prime[1] ** 2))/(2*veh.u_min)))): 
        status  = True
    elif (_pre_v.p_traj[pre_ind_nxt] -  x_prime[0] ) < (config.L + max(0,(((_pre_v.v_traj[pre_ind_nxt] ** 2) - (x_prime[1] ** 2))/(2*veh.u_min)))): 
        status = False
    else: 
        assert False,"RES_ERROR"
     
    return status           


def bisection(_pre_v, veh, temp_ind, const_active, pre_ind_nxt=None):
    
    
    u_prime = copy.deepcopy(veh.u_max)
    u1 = copy.deepcopy(veh.u_min)
    u2 = copy.deepcopy(veh.u_max)
    X = []
    X.append(veh.p_traj[temp_ind])
    X.append(veh.v_traj[temp_ind])
    x_prime = copy.deepcopy(functions.compute_state(X, u_prime, round(config.dt,1), veh.v_max))
    iter_status = None  ####False

    # assert const_active==1 or const_active==2 or const_active==3,'ERROR' 
    if const_active == 1: 
        #if RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt): 
        #    iter_status = True
        iter_status = RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt)
    elif const_active == 2: 
        # if IEP_safety_def(veh, x_prime):
        #     iter_status = True
        iter_status = IEP_safety_def(veh, x_prime)
    elif const_active == 3: 
        # if RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt) and IEP_safety_def(veh, x_prime): 
        #     iter_status = True   
        iter_status = RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt) and IEP_safety_def(veh, x_prime)
    else: assert False 
    
    assert iter_status in {True, False}  #### remove later
    
    if  iter_status:   
        u2 = copy.deepcopy(u_prime)
        u_allow_max = copy.deepcopy(u2)
    elif not iter_status: 
        u_prime = copy.deepcopy((u2 + u1)/2)
        while round(abs(u1-u2),config.delt)>0:
            X = []
            X.append(veh.p_traj[temp_ind])
            X.append(veh.v_traj[temp_ind])
            x_prime = copy.deepcopy(functions.compute_state(X, u_prime, round( config.dt,1), veh.v_max))
            
            iter_status = None    #####False
            if const_active == 1: 
                # if RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt): 
                #     iter_status = True
                iter_status = RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt)
            elif const_active == 2: 
                # if IEP_safety_def(veh, x_prime): 
                #     iter_status = True
                iter_status = IEP_safety_def(veh, x_prime)
            elif const_active == 3: 
                # if RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt) and IEP_safety_def(veh, x_prime): 
                #     iter_status = True   
                iter_status = RES_safety_def(_pre_v, veh, x_prime, pre_ind_nxt) and IEP_safety_def(veh, x_prime)
            else: assert False          

            assert iter_status in {True, False}    #### remove later

            if iter_status: 
                u1 =  copy.deepcopy(u_prime) 
            elif not iter_status:  
                u2 =  copy.deepcopy(u_prime) 
            else: assert False    

            u_prime = copy.deepcopy((u2 + u1)/2)
            assert u1<=u2
            
        u_allow_max = copy.deepcopy(min(u1,u2))
        
    else: assert False    
    
    return u_allow_max  



def acc_map(veh, _pre_v, t_init, learning_flag, flag= None, over = None):


    if flag == 'green': 
    
        if over == None: veh.sig[t_init] = 'green'  #'green__set'
        elif over == 'override': 
            veh.sig[t_init] = 'orange'
            assert False
        elif over == 'inter': assert  veh.alpha == 1  
        else: 
            assert False,'ERROR in over value for sig assignmnet'
            
        feas_stat = True
        d = None
        note_flag = 'G'
        
        
    elif flag == 'red':   

        veh.sig[t_init] = 'red'
        feas_stat = None #True
        d = None
        note_flag = 'R'
        
    else: assert False

        
    if _pre_v!=None:

        pre_ind = functions.find_index(_pre_v, t_init)
        pre_ind_nxt = functions.find_index(_pre_v, (t_init+round(config.dt,1)))
        assert pre_ind!= None,'previous veh not present'
        assert pre_ind_nxt!= None, f'previous veh not present next time, time: {(t_init+round( config.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'


        if ((_pre_v.p_traj[pre_ind] > (_pre_v.intsize + _pre_v.length))) and ((_pre_v.p_traj[pre_ind_nxt] > (_pre_v.intsize + _pre_v.length ))):
            _pre_v = None
            # print("\n*********The next step of previous vehicle is in done_set*********")


    if len(veh.t_ser) < 1:
        veh.t_ser = [veh.sp_t]
        veh.p_traj = [veh.p0]
        veh.v_traj = [veh.v0]
        veh.u_traj = []
        
    if len(veh.p_traj) >=1:

        temp_ind = functions.find_index(veh, t_init)
        assert temp_ind!= None,'current veh not present'
        assert veh.p_traj[temp_ind] <= config.L + veh.intsize,"veh position in intersection"
        
        veh.t_ser = veh.t_ser[:temp_ind+1]
        veh.p_traj = veh.p_traj[:temp_ind+1]
        veh.v_traj = veh.v_traj[:temp_ind+1]
        veh.u_traj = veh.u_traj[:temp_ind]
        
        X = []
        X.append(veh.p_traj[temp_ind])
        X.append(veh.v_traj[temp_ind])
        x_prime_z = copy.deepcopy(functions.compute_state(X, veh.u_min, round( config.dt,1), veh.v_max))
        
        if _pre_v == None:

            if flag == 'green': 
                u_allow_max = copy.deepcopy(veh.u_max)
                u_allow_min = copy.deepcopy(veh.u_min)
                assert u_allow_max > u_allow_min  ####assert u_allow_max >= u_allow_min
                u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
                
            elif flag == 'red':
            
                ############### IEP constraints ##################
                if IEP_safety_def(veh, x_prime_z):
                    
                    feas_stat = True
                    u_allow_max = copy.deepcopy(bisection(_pre_v, veh, temp_ind, 2))
                    u_allow_min = copy.deepcopy(veh.u_min)
                    u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
                    assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
                    if (u_allow_max - u_allow_min) <= 10**-6: u_allow_max = copy.deepcopy(u_allow_min)

                elif not IEP_safety_def(veh, x_prime_z):
                    feas_stat = False
                    u_allow_max = None
                    u_allow_min = None     
                    
                else: assert False
            else: assert False
        
        elif _pre_v != None:
            
            if flag == 'green':   
                
                assert RES_safety_def(_pre_v, veh, x_prime_z, pre_ind_nxt), \
                    f'GREEN SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \n \
                            prev-pos:{_pre_v.p_traj}, vel: {_pre_v.v_traj}, curr-pos;{veh.p_traj}, vel: {veh.v_traj},\n \
                            curr_id:{veh.id},prev_id:{_pre_v.id}, timne:{t_init}'

                u_allow_max = copy.deepcopy(bisection(_pre_v, veh, temp_ind, 1, pre_ind_nxt))
                u_allow_min = copy.deepcopy(veh.u_min)
                u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
                assert u_allow_max >= u_allow_min,f'max:{u_allow_max},min:{u_allow_min}'
                
                if (u_allow_max - u_allow_min) <= 10**-6:
                    u_allow_max = copy.deepcopy(u_allow_min)
                    
            elif flag == 'red':
                
                ######## RES + IEP #############
                assert RES_safety_def(_pre_v, veh, x_prime_z, pre_ind_nxt), \
                    f'GREEN SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \n \
                            prev-pos:{_pre_v.p_traj}, vel: {_pre_v.v_traj}, curr-pos;{veh.p_traj}, vel: {veh.v_traj},\n \
                            curr_id:{veh.id},prev_id:{_pre_v.id}, time:{t_init}'
     
                if IEP_safety_def(veh, x_prime_z):
                    feas_stat = True
                    u_allow_max = copy.deepcopy(bisection(_pre_v, veh,temp_ind, 3, pre_ind_nxt))
                    u_allow_min = copy.deepcopy(veh.u_min) 
                    u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
                    assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
                    if (u_allow_max - u_allow_min) <= 10**-6: 
                        u_allow_max = copy.deepcopy(u_allow_min)


                elif not IEP_safety_def(veh, x_prime_z):
                    feas_stat = False
                    u_allow_max = None
                    u_allow_min = None         

                else: assert False    
                
            else: assert False,f'flag:{flag}'    
            
            # if (u_allow_max - u_allow_min) <= 10**-6:
            #         u_allow_max = copy.deepcopy(u_allow_min)
            
        else: assert False                

    assert feas_stat!=None
    
    return state_esti(veh, u_allow_min, u_allow_max, temp_ind, t_init, feas_stat, note_flag , d, learning_flag)      
         
            








# def green_map(veh, _pre_v, t_init, over = None):

#     if over == None: veh.sig[t_init] = 'green__set'
#     elif over == 'override': 
#         veh.sig[t_init] = 'orange'
#         assert False
#     else: 
#         assert False,'ERROR in over value for sig assignmnet'
        
        
#     feas_stat = True
#     d = None
    
#     if _pre_v!=None:

#         pre_ind = functions.find_index(_pre_v, t_init)
#         pre_ind_nxt = functions.find_index(_pre_v, (t_init+round(config.dt,1)))
#         assert pre_ind!= None,'previous veh not present'
#         assert pre_ind_nxt!= None, f'previous veh not present next time, time: {(t_init+round( config.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'

#         # if ((pre_ind == None) or (_pre_v.p_traj[pre_ind] > 0)) and ((pre_ind_nxt == None) or (_pre_v.p_traj[pre_ind_nxt] > 0)):
#         #     _pre_v = None

#         ### made to just lenght of the lane ####
#         # if ((pre_ind == None) or (_pre_v.p_traj[pre_ind] > (_pre_v.intsize + _pre_v.length))) and \
#         #     ((pre_ind_nxt == None) or (_pre_v.p_traj[pre_ind_nxt] > (_pre_v.intsize + _pre_v.length ))):
#         #     _pre_v = None
#         #     print("\n*********The next step of previous vehicle is in done_set*********")


#     if len(veh.t_ser) < 1:
#         veh.t_ser = [veh.sp_t]
#         veh.p_traj = [veh.p0]
#         veh.v_traj = [veh.v0]
#         veh.u_traj = []

#     if len(veh.p_traj) >=1:

#         temp_ind = functions.find_index(veh, t_init)
#         assert temp_ind!= None,'current veh not present'
#         assert veh.p_traj[temp_ind] <= config.L + veh.intsize,"veh position in intersection"
        

#         veh.t_ser = veh.t_ser[:temp_ind+1]
#         veh.p_traj = veh.p_traj[:temp_ind+1]
#         veh.v_traj = veh.v_traj[:temp_ind+1]
#         veh.u_traj = veh.u_traj[:temp_ind]
        
#         if _pre_v == None:

#             u_allow_max = copy.deepcopy(veh.u_max)
#             u_allow_min = copy.deepcopy(veh.u_min)
#             assert u_allow_max > u_allow_min  ####assert u_allow_max >= u_allow_min
#             u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
        
#         elif _pre_v != None:
            
#                 # pre_ind = functions.find_index(_pre_v, t_init)
#                 # pre_ind_nxt = functions.find_index(_pre_v, (t_init+round(config.dt,1)))
#                 # assert pre_ind!= None,'previous veh not present'
#                 # assert pre_ind_nxt!= None, f'previous veh not present next time, time: {(t_init+round( config.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'
                
#                 X = []
#                 X.append(veh.p_traj[temp_ind])
#                 X.append(veh.v_traj[temp_ind])
#                 x_prime_z = copy.deepcopy(functions.compute_state(X, veh.u_min, round(config.dt,1), veh.v_max))
                
#                 assert RES_safety_def(_pre_v, veh, x_prime_z, pre_ind_nxt), \
#                     f'GREEN SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \n \
#                             prev-pos:{_pre_v.p_traj}, vel: {_pre_v.v_traj}, curr-pos;{veh.p_traj}, vel: {veh.v_traj},\n \
#                             curr_id:{veh.id},prev_id:{_pre_v.id}, timne:{t_init}'
                

#                 u_allow_max = copy.deepcopy(bisection(_pre_v, veh, temp_ind, 1, pre_ind_nxt))
#                 u_allow_min = copy.deepcopy(veh.u_min)
#                 u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
#                 assert u_allow_max >= u_allow_min,f'max:{u_allow_max},min:{u_allow_min}'
                
#                 if (u_allow_max - u_allow_min) <= 10**-6:
#                     u_allow_max = copy.deepcopy(u_allow_min)
                        
#         else: assert False               


#     return state_esti(veh, u_allow_min, u_allow_max, temp_ind, t_init, feas_stat, 'G', d)




# def red_map(veh, _pre_v, t_init):

#     veh.sig[t_init] = 'red'
#     feas_stat = None #True
#     d = None
    
#     if _pre_v!=None:
        
#         pre_ind = functions.find_index(_pre_v, t_init)
#         pre_ind_nxt = functions.find_index(_pre_v, (t_init+round( config.dt,1)))
#         assert pre_ind!= None,'current veh not present'
#         assert pre_ind_nxt!= None, f'current veh not present, time: {(t_init+round( config.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'


#         # if ((pre_ind == None) or (_pre_v.p_traj[pre_ind] > (0))) and \
#         #       ((pre_ind_nxt == None) or (_pre_v.p_traj[pre_ind_nxt] > (0))):
#         #     _pre_v = None
            
#     if len(veh.t_ser) < 1:
#         veh.t_ser = [veh.sp_t]
#         veh.p_traj = [veh.p0]
#         veh.v_traj = [veh.v0]
#         veh.u_traj = []

#     if len(veh.p_traj) >=1:
        
#         temp_ind = functions.find_index(veh, t_init)
#         assert temp_ind!= None,'current veh not present'
#         assert veh.p_traj[temp_ind] <= config.L + veh.intsize,"veh position in intersection"
        

#         veh.t_ser = veh.t_ser[:temp_ind+1]
#         veh.p_traj = veh.p_traj[:temp_ind+1]
#         veh.v_traj = veh.v_traj[:temp_ind+1]
#         veh.u_traj = veh.u_traj[:temp_ind]


#         X = []
#         X.append(veh.p_traj[temp_ind])
#         X.append(veh.v_traj[temp_ind])
#         x_prime_z = copy.deepcopy(functions.compute_state(X, veh.u_min, round( config.dt,1), veh.v_max))

#         if _pre_v == None:
            
#             ############### IEP constraints ##################
#             #assert _pre_v.p_traj[pre_ind_nxt] -  x_prime_z[0] >= config.L + max(0,((_pre_v.v_traj[pre_ind_nxt]**2 - x_prime_z[1]**2 )/(2*veh.u_min))), f'prev:{_pre_v.p_traj} '    # RES
#             if IEP_safety_def(veh, x_prime_z):
                
#                 feas_stat = True
#                 u_allow_max = copy.deepcopy(bisection(_pre_v, veh, temp_ind, 2))
#                 u_allow_min = copy.deepcopy(veh.u_min)
#                 u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
#                 assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
#                 if (u_allow_max - u_allow_min) <= 10**-4: u_allow_max = copy.deepcopy(u_allow_min)

#             elif not IEP_safety_def(veh, x_prime_z):
#                 feas_stat = False
#                 u_allow_max = None
#                 u_allow_min = None     
                
#             else: assert False
                
#         elif _pre_v != None:

#             # pre_ind = functions.find_index(_pre_v, t_init)
#             # pre_ind_nxt = functions.find_index(_pre_v, (t_init+round( config.dt,1)))
#             # assert pre_ind!= None,'current veh not present'
#             # assert pre_ind_nxt!= None, f'current veh not present, time: {(t_init+round( config.dt,1))}, \nt_ser: {_pre_v.t_ser}, p_traj: {_pre_v.p_traj}'
#             ######## RES + IEP #############
#             assert RES_safety_def(_pre_v, veh, x_prime_z, pre_ind_nxt), \
#                 f'GREEN SIGNAL: \n prev_pos:{_pre_v.p_traj[pre_ind_nxt] }, prev_vel:{_pre_v.v_traj[pre_ind_nxt]},  curr_pos:{ x_prime_z[0]},curr_vel:{ x_prime_z[1]}, \n \
#                          prev-pos:{_pre_v.p_traj}, vel: {_pre_v.v_traj}, curr-pos;{veh.p_traj}, vel: {veh.v_traj},\n \
#                         curr_id:{veh.id},prev_id:{_pre_v.id}, time:{t_init}'
     

#             if IEP_safety_def(veh, x_prime_z):
#                 feas_stat = True
#                 u_allow_max = copy.deepcopy(bisection(_pre_v, veh,temp_ind, 3, pre_ind_nxt))
#                 u_allow_min = copy.deepcopy(veh.u_min) 
#                 u_allow_min = copy.deepcopy(acc_clip(veh, u_allow_min, temp_ind))
#                 assert u_allow_max >= u_allow_min,f'max:{u_allow_max} , min;{u_allow_min}'
#                 if (u_allow_max - u_allow_min) <= 10**-6: u_allow_max = copy.deepcopy(u_allow_min)

#             elif not IEP_safety_def(veh, x_prime_z):
#                 feas_stat = False
#                 u_allow_max = None
#                 u_allow_min = None         

#             else: assert False    

#         else: assert False

#     assert feas_stat!=None
    
#     return state_esti(veh, u_allow_min, u_allow_max, temp_ind, t_init, feas_stat, 'R', d)      
























