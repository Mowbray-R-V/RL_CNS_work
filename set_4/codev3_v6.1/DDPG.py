# pylint: disable=unused-import
import csv
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import  GRU, Dense
import tensorflow as tf
import numpy as np
import time
import copy 
import matplotlib.pyplot as plt
import data_file
import math
from pprint import pprint 
import functions
from contextlib import redirect_stdout
import os
# from tensorflow.keras.utils import plot_model




# graph_flag = False #True # 
# tf.config.experimental_run_functions_eagerly(graph_flag)
# if graph_flag== True: tf.print(f'\n\n********************************* Graph flag status************:  "{graph_flag}\n\n')

action_list = []
action_sig_list = []
gradient_list = []

def float_precision(value, precision):
    float128_value = np.float128(value)
    truncated_value = np.round(float128_value, precision)
    result = float(truncated_value)
    return result



   
class NNModel(tf.keras.Model): 
    def __init__(self, lat_size): 
        super().__init__()
        # self.RNN = SeqModule(lat_size)
        self.lat_size = lat_size
        self.actor_model = ActorModel()
        self.critic_model = CriticModel(self.lat_size)
  
    @tf.function    
    def call(self, control_state = None, phase_state = None, action = None, act_or_cri = 'act'):
        # assert features.shape[0] == data_file.rl_ddpg_samp_size or features.shape[0] == 1, f'wrong dimensions, input_batch: {features.shape[0],features.shape}'
        # seq, latent = self.RNN(input)
        if act_or_cri == 'act': 
            # mask = (input!= 0).any(axis=-1)
            # print(f'model inputsize:{phase_state.shape}')
            return  self.actor_model(control_state, phase_state)
        elif act_or_cri == 'critic': 
            return   self.critic_model(control_state, action)
        else: assert False



class UnitVectorLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UnitVectorLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute the norm of each row
        norm = tf.norm(inputs, axis=1, keepdims=True)
        norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
        # Divide inputs by their norm to get unit vectors
        unit_vectors = inputs / norm
        return unit_vectors


class CNN(layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, activation=None):
        super(CNN, self).__init__()
        self.act = activation
        self.conv = layers.Conv1D(kernel_num, kernel_size=kernel_size, strides=strides, activation=self.act)
        self.kernel = kernel_num

    def call(self, input):
        y = tf.reshape(input, [input.shape[0], input.shape[1] * input.shape[2], 1])
        y = self.conv(y)
        return y   

class ActorModel(tf.keras.Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.robot_dense3 = Dense(18)#, activation='relu')
        self.robot_dense4 = Dense(36)#, activation='relu')
        self.robot_dense = Dense(9)#, activation='linear')
        self.robot_dense2 = Dense(1)#,activation='tanh')
        self.unit = UnitVectorLayer()
        # self.reshape_con = layers.Reshape(target_shape=(data_file.num_veh, data_file.state_size))
        # self.repeat = layers.RepeatVector(data_file.num_veh)
        self.conact  = layers.Concatenate(axis = -1)

        self.flat = layers.Flatten()
        # self.reshape_obs = layers.Reshape(target_shape=(data_file.num_veh, data_file.state_size))

        ###########original model##################
        # self.layer_conv2004 = CNN(2004, data_file.num_veh*data_file.state_size, data_file.num_veh*data_file.state_size, activation="relu")
        # self.layer_conv1200 = CNN(1200, 2004, 2004, activation="relu")
        # self.layer_conv800 = CNN(800, 1200, 1200, activation="relu")
        # self.layer_conv400 = CNN(400, 800, 800)
        # self.layer_conv200 = CNN(200, 400, 400, activation="relu")
        # self.layer_conv120 = CNN(120, 200, 200)
        # self.layer_conv60 = CNN(60, 120, 120)

        # self.layer_conv48 = CNN(48, 102, 102, activation="relu")
        # self.layer_conv24 = CNN(24, 48, 48)
        # self.layer_conv12 = CNN(12, 24, 24, activation="relu")
        # self.layer_conv6 = CNN(6, 12, 12)
        # self.layer_conv2 = CNN(2, 6, 6)
        ###########original model##################


        self.layer_conv2004p = CNN(2004, data_file.num_veh*data_file.state_size, data_file.num_veh*data_file.state_size, activation="relu")
        self.layer_conv120p = CNN(120, 2004, 2004, activation="relu")
        self.layer_conv60p = CNN(60, 120, 120)
        self.layer_conv48p = CNN(48, 102, 102, activation="relu")
        self.layer_conv24p = CNN(24, 48, 48)
        self.layer_conv2p = CNN(2, 24, 24)




    @tf.function
    def call(self, control_state, phase_state, mask=None):

        # tf.print(f'phase_shape:{phase_state.shape}, type:{type(phase_state)}')
        # tf.print(f'control_shape:{control_state.shape}, type:{type(control_state)}')



        phase_scores = []
        ######phase_module#########
        for i in range(phase_state.shape[1]):
            phase_input = phase_state[:, i, :, :]  # Shape: (batch_size, num_robots, num_features)

            phase_mask = tf.reduce_any(phase_input != 0, axis=-1)  # Shape: (batch_size, num_robots)

            # Apply the dense layer to each robot's features
            rob_sc = self.robot_dense4(phase_input)
            # print(rob_sc)
            robot_scores = self.robot_dense2(self.robot_dense(self.robot_dense3(self.robot_dense4(phase_input))))  # Shape: (batch_size, num_robots, 1)
            # print('robot_scores',robot_scores.shape)  
            masked_scores = tf.where(tf.expand_dims(phase_mask, axis=-1), robot_scores, tf.zeros_like(robot_scores))
            # print('masked_scores',masked_scores.shape)
            phase_score = tf.reduce_sum(masked_scores, axis=1)  # Shape: (batch_size, 1)
            phase_scores.append(phase_score)

        phase_scores = tf.stack(phase_scores, axis=1)  # Shape: (batch_size, num_phases, 1)
        phase_scores = tf.squeeze(phase_scores,axis=-1)
        phase_scores_norm = self.unit(phase_scores)
        # tf.print(f'phase_scores_norm:{phase_scores_norm.shape}')
        phase_scores_norm_dim = tf.expand_dims(phase_scores_norm, axis=-2)#### check compute graph
        phase_scores_num = tf.repeat(phase_scores_norm_dim,repeats=data_file.num_veh, axis=1)   #### check compute graph
        ######phase_module#########


       ######control_module#########
        control_state_1 = tf.expand_dims(control_state, axis=-2)#### check compute graph
        # tf.print(f'control_sttae_1:{control_state_1.shape}')
        control_state_num = tf.repeat(control_state_1,repeats=data_file.num_veh, axis=1)   #### check compute graph
        # tf.print(f'control_shape_num:{control_state_num.shape}')

        ###########original model##################
        # control_state_num = self.layer_conv60(self.layer_conv120(self.layer_conv200(self.layer_conv400(self.layer_conv800(self.layer_conv1200(self.layer_conv2004(control_state_num)))))))
        ###########original model##################


        control_state_num = self.layer_conv60p(self.layer_conv120p(self.layer_conv2004p(control_state_num)))


        # print(f'control_state_num_cnn:{control_state_num.shape}')

        obs  =    tf.expand_dims(control_state, axis=-2) #### check compute graph
        # tf.print(f'obs:{obs.shape}')
        obs_state  =  tf.reshape(control_state, [control_state.shape[0],data_file.num_veh, data_file.state_size])
        # tf.print(f'obs:{obs_state.shape}')
        state = self.conact([control_state_num, obs_state,phase_scores_num ])
        # tf.print(f'state:{state.shape}')

        ###########original model##################
        # control = self.layer_conv2(self.layer_conv6(self.layer_conv12(self.layer_conv24(self.layer_conv48(state)))))
        ###########original model##################

        control = self.layer_conv2p(self.layer_conv24p(self.layer_conv48p(state)))
        # print(f'control:{control.shape}')
        # exit()
        ######control_module#########


        phase_out  = self.flat(phase_scores_norm)  
        # tf.print(f'phase_scores_norm:{phase_scores_norm.shape}')
        control_out  = self.flat(control)  
        # tf.print(f'control_out:{control_out.shape}')
        outputs = self.conact([phase_out, control_out])
        # print(f'------actions:{outputs.shape}')
        return outputs


class CustomLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-6, mask_value=None, center=True, scale=True):
        super(CustomLayerNormalization, self).__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.mask_value = mask_value
        self.center = center
        self.scale = scale

    #def build(self, input_shape):
    #    self.gamma_1 = self.add_weight(name='gamma', shape=input_shape[self.axis:], initializer='ones', trainable=True)
    #    self.beta = self.add_weight(name='beta', shape=input_shape[self.axis:], initializer='zeros', trainable=True)

    @tf.function
    def call(self, inputs, mask):
        mask_1 = tf.cast(mask, dtype=tf.float32)
        #print(f'***************mask_1:{mask_1.shape},{mask_1}')
        inputs = tf.squeeze(inputs)
        masked_sum = tf.reduce_sum(inputs * mask_1, axis=self.axis, keepdims=True)
        #masked_count = tf.math.count_nonzero(mask_1)
        masked_count = tf.reduce_sum(mask_1, axis=self.axis, keepdims=True)
        #masked_count = tf.reduce_sum(mask_1, axis=1, keepdims=True)
        masked_mean = masked_sum / masked_count
        masked_diff = (inputs - masked_mean) * mask_1
        masked_variance = tf.reduce_sum(masked_diff ** 2, axis=self.axis, keepdims=True) / masked_count
        normalized_inputs = (inputs - masked_mean) / tf.sqrt(masked_variance + self.epsilon)

        return normalized_inputs

class CriticModel(tf.keras.Model):
    def __init__(self, lat_size):
        super(CriticModel, self).__init__()
        self.output_concat = layers.Concatenate(axis=-1)



        ###########original model##################
        # self.layer_conv2004 = CNN(2004, data_file.num_veh*data_file.state_size, data_file.num_veh*data_file.state_size, activation="relu")
        # self.layer_conv1200 = CNN(1200, 2004, 2004, activation="relu")
        # self.layer_conv800 = CNN(800, 1200, 1200, activation="relu")
        # self.layer_conv400 = CNN(400, 800, 800)
        # self.layer_conv250 = CNN(250, 400, 400)

        # self.layer_conv200 = CNN(200, 486, 486, activation="relu")
        # self.layer_conv120 = CNN(120, 200, 200, activation="relu")
        # self.layer_conv60 = CNN(60, 120, 120)
        # self.layer_conv30 = CNN(30, 60, 60, activation="relu")
        # self.layer_conv18 = CNN(18, 30, 30, activation="relu")
        # self.layer_conv9 = CNN(9, 18, 18)
        # self.layer_conv1 = CNN(1, 9, 9)
        ###########original model##################



        self.layer_conv2004p = CNN(2004, data_file.num_veh*data_file.state_size, data_file.num_veh*data_file.state_size, activation="relu")
        self.layer_conv800p = CNN(800, 2004, 2004)
        self.layer_conv400p = CNN(400, 800, 800)

        self.layer_conv200p = CNN(200, 486, 486, activation="relu")
        self.layer_conv120p = CNN(120, 200, 200, activation="relu")
        self.layer_conv60p = CNN(60, 120, 120)
        self.layer_conv30p = CNN(9, 60, 60, activation="relu")
        self.layer_conv18p = CNN(1, 9, 9, activation="relu")



        # self.gru_latent = GRU(lat_size)
        # self.gru_action = GRU(lat_size)
        # self.lay_norm = CustomLayerNormalization()
        # self.action_mask = layers.Masking(mask_value=-190.392)
        # # self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        # self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        # self.fc3 = tf.keras.layers.Dense(32)#, activation='relu')
        # self.fc4 = tf.keras.layers.Dense(16, activation='relu')
        # # self.fc5 = tf.keras.layers.Dense(8)#, activation='relu')
        # self.fc6 = tf.keras.layers.Dense(1)#, activation='relu')

    @tf.function
    def call(self, state_inp, action):

        # print(f'***************state:{state_inp.shape}')
        # print(f'***************action:{action.shape}')

        state   =    tf.expand_dims(state_inp, axis=-2) #### check compute graph
        # tf.print(f'state:{state.shape}')

        ###########original model##################
        # state_mod = self.layer_conv250(self.layer_conv400(self.layer_conv800(self.layer_conv1200(self.layer_conv2004(state)))))
        ###########original model##################

        state_mod = self.layer_conv400p(self.layer_conv800p(self.layer_conv2004p(state)))


        # tf.print(f'state_mod:{state_mod.shape}')
        action_mod  =  tf.expand_dims(action, axis=-2) #### check compute graph
        # tf.print(f'action_mod:{action_mod.shape}')
        concat = self.output_concat([state_mod, action_mod])
        # tf.print(f'concat:{concat.shape}')
        ###########original model##################
        # q_val = self.layer_conv1(self.layer_conv9(self.layer_conv18(self.layer_conv30(self.layer_conv60p(self.layer_conv120p(self.layer_conv200p(concat)))))))
        ###########original model##################

        q_val = self.layer_conv18p(self.layer_conv30p(self.layer_conv60p(self.layer_conv120p(self.layer_conv200p(concat)))))
        
        # print(f'q_val:{q_val.shape}')

        q_val =  tf.squeeze(q_val)
        # tf.print(f'q_val:{q_val.shape}')
        return q_val



class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):


        self.num_states =  data_file.state_size*data_file.num_veh   ##### need to edit
        self.num_obs = data_file.num_features
        self.num_actions = data_file.num_actions
        self.noise_std_dev = noise_std_dev
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
        self.model = NNModel(32)
        self.target_model = NNModel(32)
        
        self.target_model.set_weights(self.model.get_weights())   

        # state = (1,112*30),  phase_state = (1,12,28,30),  
        sample_batch = 1
        build_ouput_act = self.model(control_state = tf.random.uniform((sample_batch,data_file.num_veh*data_file.state_size)), phase_state=tf.random.uniform((sample_batch, data_file.num_phases,(2*data_file.max_robot_lane),data_file.state_size)), act_or_cri='act')
        # exit()
        build_ouput_crtic = self.model(control_state=tf.random.uniform((sample_batch,data_file.num_veh*data_file.state_size)), phase_state=None, action = tf.random.uniform((sample_batch,self.num_actions)),act_or_cri='critic')
        # exit()

        self.model.summary()
        self.model.actor_model.summary()
        self.model.critic_model.summary()


        with open('model_summary.txt', 'w+') as f:
            with redirect_stdout(f):
                self.model.summary()
                self.model.actor_model.summary()
                self.model.critic_model.summary()

        #####plot_needs_upadte####
        # if not os.path.isfile('Model_Diagram.png'):
        # plot_model(self.model, to_file='Model_Diagram.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        #####plot_needs_upadte####
        

        # build_ouput = self.model(control_state = None, phase_state=tf.random.uniform((1,data_file.num_phases,(2*data_file.max_robot_lane),data_file.state_size)), act_or_cri='act')
        # build_ouput_1 = self.model(control_state=tf.random.uniform((1,data_file.num_veh ,data_file.state_size)), phase_state=None, action = tf.random.uniform((1,self.num_actions)),act_or_cri='critic')

        self.critic_lr = cri_lr
        self.actor_lr = act_lr

        self.critic_optimizer_ = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer_ = tf.keras.optimizers.Adam(self.actor_lr)

        self.gamma_ = disc_factor
        self.tau_ = polyak_factor

        self.buff_size = buff_size
        self.samp_size = samp_size
        
        self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, observe_size=self.num_obs, action_size=self.num_actions,  buff_model = self.model, buff_model_target = self.target_model, gamma=self.gamma_, tau=self.tau_, cri_optimizer=self.critic_optimizer_, act_optimizer=self.actor_optimizer_)


    @tf.function
    def update_target(self, target_weights, weights, tau):  
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


    def sigmoid_func(self, np_array):
        temp_array = []
        for t in np_array:
            trunc_val = t # float_precision(t, 8)   #np.float128(t) #
            temp_array.append(1 / (1 + np.exp(- trunc_val)))
            #assert temp_array[-1]<0 and temp_array[-1]>0,f'ori value:{t},trun val :{trunc_val}, sig:{1 / (1 + np.exp(- trunc_val))}'
        return np.asarray(temp_array)


    def policy(self, state_control, state_phase, time, noise_object, num_veh=None):

        # print(f'*************control: shape:{np.shape(state_control)}')
        state_control_mod  = np.expand_dims(state_control, axis  = 0)
        # print(f'*************control: shape:{np.shape(state_control_mod)}')
        # print(f'*************control: type:{np.dtype(state_control_mod[0][0])},shape:{np.shape(state_control_mod)}')
        # print(f'*************phase: type:{np.dtype(state_phase[0][0][1][1])},shape:{np.shape(state_phase)}')
        # print(f'*************control: shape:{np.shape(state_phase)}')


        #### NAN check########
        # a = np.empty((1, 12, 28, 30))
        # a[0][0][0][0] = np.nan
        state_has_nan = [np.isnan(state_phase[0]).any()]
        assert all ([not x for x in state_has_nan]) , f'phase_state-bad : phase_state: {state_phase}'
        state_has_nan_cont = [np.isnan(state_control[0]).any()]
        assert all ([not x for x in state_has_nan_cont]),f'control_state-bad : control_state: {state_has_nan_cont}'
        #### NAN check ########

        ##### random input testing #####
        # input_control = tf.random.uniform((1,3360))     ## change the dimensions as needed
        # input_phase = tf.random.uniform((1,12,28,30))     ## change the dimensions as needed
        # sampled_actions = tf.squeeze(self.model(control_state = input_control, phase_state = input_phase, act_or_cri = 'act'))
        ##### random input testing #####
        # exit()``


        # state = (1,112*30),  phase_state = (1,12,28,30),  
        state_con = tf.convert_to_tensor(state_control_mod, dtype=tf.float32)
        state_ph = tf.convert_to_tensor(state_phase, dtype=tf.float32)
        # print(f'*************control: type:{type(state_con)},shape:{tf.shape(state_con)}')
        # print(f'*************phase: type:{type(state_ph)},shape:{tf.shape(state_ph)}')

        sampled_actions = tf.squeeze(self.model(control_state = state_con, phase_state = state_ph, act_or_cri = 'act'))
        # print(f'action:{sampled_actions.shape}')


        if noise_object is not None:
            noise = noise_object()
            noise = np.float32(noise)
            sampled_actions = sampled_actions.numpy() + noise #np.maximum(noise, 0)

        legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases:(self.num_actions)])
        legal_action_ph = sampled_actions[:data_file.num_phases] 
        legal_action_set = np.append(legal_action_ph, legal_action)

        
        f = open("debug.txt", "a")
        f.write(f'action_after_sigmoid:{legal_action}')
        f.close()
        
        assert all([ _>=0 and _<=1  for _ in legal_action_set[data_file.num_phases:(self.num_actions)]]),f'alpha value not in range :{legal_action_set[data_file.num_phases:(self.num_actions)]}, \
            signal:{legal_action_set[:data_file.num_phases]} \n\n, state_contr:{state_control}'

        return [np.squeeze(legal_action_set)]


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=5e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.t = 0
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        self.t += self.dt
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.random.normal(size=self.mean.shape) * (1/self.t)
        )
        self.x_prev = x
        
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=700000, batch_size=64, state_size=12, observe_size=None, action_size=5, buff_model = None, buff_model_target = None, gamma=0.99, tau=0.001, cri_optimizer=None, act_optimizer=None):
        # Number of "experiences" to store at max


        ####################### Tensorflow buffer available   ####################### 
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = observe_size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.model_buff = buff_model
        self.model_target_buff =  buff_model_target       
        self.gamma = gamma
        self.tau = tau
        self.critic_optimizer = cri_optimizer
        self.actor_optimizer = act_optimizer

        self.state_buffer = np.zeros((self.buffer_capacity,1), dtype=object)
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity,1), dtype=object)
        ####################### Tensorflow buffer available   ####################### 


        # self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        # self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        # self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))

    def remember(self, obs_tuple):


        index = self.buffer_counter % self.buffer_capacity

        # print(f'BUFFER_____prev_state:{obs_tuple[0]},prev_act:{obs_tuple[1]},prev_rew:{obs_tuple[2]},curr_state:{obs_tuple[3]}')
        # print(f'prev_state:{type(prev_state)},prev_act:{type(prev_act)},prev_rew:{type(prev_rew)},curr_state:{type(curr_state)}')
        
        # assert len(obs_tuple[0])==1, f'obs_tuple:{len(obs_tuple[0])},state_size:{self.state_size}'
        assert len(obs_tuple[1]) == self.action_size ,f'replay assignment error state:'
        # assert len(obs_tuple[3])==1, f'replay assignment error'
             
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1



    @tf.function
    def update(self, control_state, phase_state, action_batch, reward_batch, next_control_state_batch, next_phase_state_batch):

        # weights_crtic_loss = [self.model_buff.RNN.trainable_variables,self.model_buff.critic_model.trainable_variables]
        # weights_crtic_loss = [_ for __ in range(len(weights_crtic_loss)) for _ in weights_crtic_loss[__] ]
        # weights_actor_loss = [self.model_buff.RNN.trainable_variables,self.model_buff.actor_model.trainable_variables]
        # weights_actor_loss = [_ for __ in range(len(weights_actor_loss)) for _ in weights_actor_loss[__] ]

        # with tf.GradientTape() as tape:
        #     target_actions = self.model_target_buff(next_state_batch, act_or_cri='act')
        #     y = reward_batch + self.gamma * self.model_target_buff(next_state_batch, None, target_actions, act_or_cri='critic')
        #     critic_value = self.model_buff(state_batch, None, action_batch, act_or_cri='critic')
        #     critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value),1)
        # critic_grad = tape.gradient(critic_loss,weights_crtic_loss)  
        # self.critic_optimizer.apply_gradients(
        #     zip(critic_grad, weights_crtic_loss) 
        # )
        
        # learn_init_time = time.time()

        with tf.GradientTape() as tape:
            target_actions = self.model_target_buff(next_control_state_batch, next_phase_state_batch, act_or_cri='act')
            y = reward_batch + self.gamma * self.model_target_buff(next_control_state_batch, next_phase_state_batch, target_actions, act_or_cri='critic')
            critic_value = self.model_buff(control_state, phase_state, action_batch, act_or_cri='critic')
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value),1)
        critic_grad = tape.gradient(critic_loss, self.model_buff.critic_model.trainable_variables)   #   weights_crtic_loss)  
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.model_buff.critic_model.trainable_variables)   #  weights_crtic_loss) 
        )
        # print(f"[update.py]: critic time: {round(time.time() - learn_init_time, 3)}")
        
        # with tf.GradientTape() as tape:
        #     #actions = self.actor_model([state_batch, obs_batch], training=True)
        #     actions = self.model_buff(state_batch,act_or_cri='act' )
        #     critic_value = self.model_buff(state_batch, None, actions, act_or_cri='critic')
        #     actor_loss = -tf.math.reduce_mean(critic_value)
        # actor_grad = tape.gradient(actor_loss, weights_actor_loss)
        # self.actor_optimizer.apply_gradients(
        #     zip(actor_grad, weights_actor_loss)  
        # )



        # learn_init_time = time.time()

        with tf.GradientTape() as tape:
            actions = self.model_buff(control_state, phase_state, act_or_cri='act' )
            critic_value = self.model_buff(control_state, phase_state, actions, act_or_cri='critic')
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.model_buff.actor_model.trainable_variables)   #weights_actor_loss)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.model_buff.actor_model.trainable_variables)   #weights_actor_loss)  
        )

        # print(f"[update.py]: actor time: {round(time.time() - learn_init_time, 3)}")

        # exit()



    @tf.function
    def learn(self):


        record_range = min(self.buffer_counter, self.buffer_capacity) # Get index range/sampling range
        batch_indices = np.random.choice(record_range, self.batch_size)# Randomly sample a tuple for each batch

        # print(f'\n\n state extrct values-- contol:{np.shape(self.state_buffer[batch_indices])},{type(self.state_buffer[batch_indices][0])}')
        phase_state_mod = np.zeros((np.shape(self.state_buffer[batch_indices])[0], data_file.num_phases, (data_file.max_robot_lane*2), data_file.state_size))
        control_state_mod = np.zeros((np.shape(self.state_buffer[batch_indices])[0], data_file.num_veh*data_file.state_size))

        nxt_phase_state_mod = np.zeros((np.shape(self.next_state_buffer[batch_indices])[0], data_file.num_phases, (data_file.max_robot_lane*2), data_file.state_size))
        nxt_control_state_mod = np.zeros((np.shape(self.next_state_buffer[batch_indices])[0], data_file.num_veh*data_file.state_size))



        # print(f'*********phase_state_mod: {phase_state_mod.shape}, {phase_state_mod[0].shape}')
        # print(f'*********control_state_mod: {control_state_mod.shape}, {control_state_mod[0].shape}')

        for iter in range(np.shape(self.state_buffer[batch_indices])[0]): 
            _, phase_state_mod[iter]  = copy.deepcopy(functions.phase_state_transform(self.state_buffer[batch_indices][0][0]))
            control_state_mod[iter]  = copy.deepcopy(functions.phase_obs_transform(self.state_buffer[batch_indices][0][0]))
            _, nxt_phase_state_mod[iter]  = copy.deepcopy(functions.phase_state_transform(self.next_state_buffer[batch_indices][0][0]))
            nxt_control_state_mod[iter]  = copy.deepcopy(functions.phase_obs_transform(self.next_state_buffer[batch_indices][0][0]))

        # print(f'*********phase_state_mod: {phase_state_mod}, {phase_state_mod[0].shape}')
        # print(f'*********control_state_mod11: {control_state_mod.shape}, {control_state_mod[0].shape}')


        # state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        # next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        control_state = tf.convert_to_tensor(control_state_mod, dtype=tf.float32)
        phase_state = tf.convert_to_tensor(phase_state_mod, dtype=tf.float32)
        next_control_state_batch = tf.convert_to_tensor(nxt_control_state_mod, dtype=tf.float32)
        next_phase_state_batch = tf.convert_to_tensor(nxt_phase_state_mod, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        # exit()


        # # tensor = tf.constant([1.0, float('nan'), 2.0, float('nan'), 3.0])
        # # control_state_mask = tf.math.is_nan(control_state)
        # control_state_mask = tf.reduce_all(tf.math.is_nan(control_state))
        # # phase_state_mask = tf.math.is_nan(phase_state)
        # # next_control_state_batch_mask = tf.math.is_nan(next_control_state_batch)
        # # next_phase_state_batch_mask = tf.math.is_nan(next_phase_state_batch)
        # tf.Assert(control_state_mask, [tf.math.is_nan(control_state)])

        # exit()
        # tf.debugging.assert_equal(control_state_mask, True, message="There are False values in the tensor")
        # assert all ([not x for x in control_state_mask]) , f'phase_state-bad : phase_state: {control_state_mask}'
        # assert all ([not x for x in phase_state_mask]) , f'phase_state-bad : phase_state: {phase_state_mask}'
        # assert all ([not x for x in next_control_state_batch_mask]) , f'phase_state-bad : phase_state: {next_control_state_batch_mask}'
        # assert all ([not x for x in next_phase_state_batch_mask]) , f'phase_state-bad : phase_state: {next_phase_state_batch_mask}'
        # assert all ([not x for x in state_has_nan]) , f'phase_state-bad : phase_state: {state_phase}'

        # learn_init_time = time.time()
        self.update(control_state, phase_state, action_batch, reward_batch, next_control_state_batch, next_phase_state_batch)
        # print(f"[update.py]: learning time: {round(time.time() - learn_init_time, 3)}")

