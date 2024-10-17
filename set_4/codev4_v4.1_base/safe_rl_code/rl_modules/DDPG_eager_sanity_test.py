# import gym
import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
#from tensorflow.keras.utils import 
from tensorflow.keras.layers import  GRU
#from tensorflow.keras.layers import Bidirectional, GRU
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import data_file
import math
#import tensorflow.compat.v1 as tf


#tf.config.run_functions_eagerly(True)
def float_precision(value, precision):
    #print(fvdgdgghf)
    float128_value = np.float128(value)
    truncated_value = np.round(float128_value, precision)
    result = float(truncated_value)
    return result

class SeqModule(layers.Layer):
    def __init__(self, lat_size):
        super(SeqModule, self).__init__()
        #self.num_state =  (data_file.num_features+(data_file.num_lanes*4))*data_file.num_veh 
        self.flat = layers.Flatten()
        self.reshape = layers.Reshape(target_shape=(1, ( (data_file.num_features+(data_file.num_lanes*4))*data_file.num_veh  )))
        self.reshape_1 = layers.Reshape(target_shape=(data_file.num_veh, (data_file.num_features + (data_file.num_lanes * 4))))
        #self.reshape_1 = layers.Reshape(target_shape=(data_file.num_veh, (data_file.num_features * data_file.num_lanes * 4)))
        self.gru = layers.Bidirectional(GRU(lat_size, return_sequences=True, return_state=True, activation='relu'))
        self.state_mask = layers.Masking(mask_value=0.0)
        self.FC1 = tf.keras.layers.Dense(16, activation='relu')


    def build(self, input_shape):
        super().build(input_shape)

    #@tf.function  
    def call(self, inputs):
        inputs = self.reshape(inputs)
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = self.reshape_1(inputs)
        masked_inputs = self.state_mask(inputs)
        mask = self.state_mask.compute_mask(inputs)
        seq_output, forward_state, backward_state = self.gru(inputs, mask=mask)
        #print(f'---------state:{seq_output.shape}')
        #print(f'maks:*********{mask},{mask.shape}')
        #print(f'd,khjgfgf{[_ for _ in tf.math.equal(mask,True)[0] if _]}')
        seq_output = self.FC1(seq_output)
        #print(f'---------state:{seq_output.shape}')
        num_agents =  len([el for el in tf.math.equal(mask,True)[0] if el])
        #print(f'dfghjkl{num_agents}')
        
        ###############################################
        ##################
        #sanity check for mixed states required 
        ###################
        ###############################################

        if tf.math.floormod(num_agents,2) == 0: latent = seq_output[:,int(num_agents/2),:]
        elif tf.math.floormod(num_agents,2) != 0: latent = seq_output[:,int((num_agents+1)/2),:]
        #print(f'hn/2_from_middle:{latent.shape} ')
        return seq_output, latent

class NNModel(tf.keras.Model): 
    def __init__(self, lat_size): 
        super().__init__()
        self.RNN = SeqModule(lat_size)
        self.lat_size = lat_size
        self.actor_model = ActorModel()
        self.critic_model = CriticModel(self.lat_size)
        self.FC1 = tf.keras.layers.Dense(16, activation='relu')
        #self.FC2 = tf.keras.layers.Dense(16, activation='relu')


    #def build(self, input_shape):
    #    self.RNN.build(input_shape)
    #    self.actor_model.build([(None,112,128),(None,128)])
    #    self.critic_model.build([(None,128),(None,124)])
  
    #@tf.function    
    def call(self, input, action=None, act_or_cri='act'):
        assert input.shape[0] == data_file.rl_ddpg_samp_size or input.shape[0] == 1, f'wrong dimensions, input_batch: {input.shape[0]}'
        #tf.print(f'{input.shape}')
        seq, latent = self.RNN(input)
        #print(f'***************{latent.shape}')    
        #print(f'***************{seq.shape}') 
        if act_or_cri == 'act': 
            y_hat = self.actor_model(seq, latent)
            return y_hat
        elif act_or_cri == 'critic': 
            return self.critic_model(latent, action)

class CNN(layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides):
        super(CNN, self).__init__()
        self.conv = layers.Conv1D(kernel_num, kernel_size=kernel_size, strides=strides, activation='relu')
        self.kernel = kernel_num

    #@tf.function
    def call(self, input):
        y = tf.reshape(input, [input.shape[0], input.shape[1] * input.shape[2], 1])
        y = self.conv(y)
        return y   

class ActorModel(tf.keras.Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        
        self.layer_conv12 = CNN(12, 16, 16)
        self.layer_conv24 = CNN(24, 44, 44)
        self.layer_conv12_24 = CNN(12, 24, 24)
        self.layer_conv6 = CNN(6, 12, 12)
        self.layer_conv1 = CNN(1, 6, 6)
        self.output_concat = layers.Concatenate()
        self.flat = layers.Flatten()

    #@tf.function
    def call(self, obs, latent):
        #print(f'latent: {latent.shape}')
        #print(f'obs{obs.shape}')
        latent = tf.expand_dims(latent, axis=1)
        #print(f'lat_1:{latent.shape}')
        signal =self.layer_conv12(latent)
        #print(f'sig{signal.shape}')
        signal_cop =tf.repeat(signal, repeats=data_file.num_veh, axis=1)
        #print(f'sig_cop{signal_cop.shape}')
        state = tf.repeat(latent, repeats=data_file.num_veh, axis=1)
        #print(f'state: {state.shape}')
        input = self.output_concat([state, obs, signal_cop])
        #print(f'input to control{input.shape}') 
        control = self.layer_conv1(self.layer_conv6(self.layer_conv12_24(self.layer_conv24(input))))
        #print(f'********control{control.shape}') 
        control = self.flat(control)
        #print(f'********control{control.shape}') 
        signal = self.flat(signal)
        #print(f'*****sig {signal.shape}')
        outputs = self.output_concat([signal, control])
        #print(f'*****output {outputs.shape}')
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

    #@tf.function
    def call(self, inputs, mask):
        mask_1 = tf.cast(mask, dtype=tf.float32)
        #print(f'***************mask_1:{mask_1.shape},{mask_1}')
        #exit()
        inputs = tf.squeeze(inputs)
        masked_sum = tf.reduce_sum(inputs * mask_1, axis=self.axis, keepdims=True)
        #masked_count = tf.math.count_nonzero(mask_1)
        masked_count = tf.reduce_sum(mask_1, axis=self.axis, keepdims=True)
        #masked_count = tf.reduce_sum(mask_1, axis=1, keepdims=True)
        masked_mean = masked_sum / masked_count
        masked_diff = (inputs - masked_mean) * mask_1
        masked_variance = tf.reduce_sum(masked_diff ** 2, axis=self.axis, keepdims=True) / masked_count
        normalized_inputs = (inputs - masked_mean) / tf.sqrt(masked_variance + self.epsilon)

    #    if self.scale:
    #        normalized_inputs *= self.gamma_1
    #    if self.center:
    #        normalized_inputs += self.beta

        return normalized_inputs


class CriticModel(tf.keras.Model):
    def __init__(self, lat_size):
        super(CriticModel, self).__init__()
        self.output_concat = layers.Concatenate()
        self.lay_norm = CustomLayerNormalization()
        self.gru_action = GRU(lat_size, activation='relu')
        self.action_mask = layers.Masking(mask_value=-190.392)
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(64, activation='relu')
        self.fc4 = tf.keras.layers.Dense(1)

    #@tf.function
    def call(self, latent_state, action):
        action = tf.expand_dims(action, axis=-1)
        masked_inputs = self.action_mask(action)
        #print(f'***************mask_action:{masked_inputs.shape},{masked_inputs}')
        mask = self.action_mask.compute_mask(action)
        #print(f'***************mask_action:{mask.shape},{mask}')
        
        
        #exit()
        assert mask.shape[0] == action.shape[0], f'{mask.shape[0]}, {action.shape[0]}'
        #print(f'****norm_inp{type(action)},{action.dtype}')

        norm_action = self.lay_norm(action, mask)
        norm_action = tf.expand_dims(norm_action, axis=-1)
        latent_act = self.gru_action(norm_action, mask=mask)
        concat = self.output_concat([latent_state, latent_act])
        Q_val = self.fc1(concat)
        Q_val = self.fc2(Q_val)
        Q_val = self.fc3(Q_val)
        Q_val = self.fc4(Q_val)
        return Q_val

    
class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):
        #num_veh = 11
        #self.num_states = data_file.num_features 
        self.num_states =  (data_file.num_features+(data_file.num_lanes*4))*data_file.num_veh    #(data_file.num_features*data_file.num_lanes*4)*data_file.num_veh
        self.num_obs = data_file.num_features
        self.num_actions = data_file.num_veh + data_file.num_phases
        self.noise_std_dev = noise_std_dev

        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))

        self.model = NNModel(16)
        self.target_model = NNModel(16)
        
        
        # Making the weights equal initially
        self.target_model.set_weights(self.model.get_weights()) ####********
        
        _ = self.model(tf.random.uniform((1, self.num_states)), act_or_cri='act')##
        
        
        _ = self.model(tf.random.uniform((1, self.num_states)), tf.random.uniform((1, self.num_actions)),act_or_cri='critic')
        #print(_)
        #exit()
        #print(_.summary())
        
        
        

        #self.target_model.actor_model.set_weights(self.actor_model_.get_weights()) 
        #self.target_model.critic_model.set_weights(self.actor_model_.get_weights()) 
        
        self.critic_lr = cri_lr
        self.actor_lr = act_lr

        self.critic_optimizer_ = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer_ = tf.keras.optimizers.Adam(self.actor_lr)

        self.gamma_ = disc_factor
        self.tau_ = polyak_factor

        self.buff_size = buff_size
        self.samp_size = samp_size
        
        self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, observe_size=self.num_obs, action_size=self.num_actions,  buff_model = self.model, buff_model_target = self.target_model, gamma=self.gamma_, tau=self.tau_, cri_optimizer=self.critic_optimizer_, act_optimizer=self.actor_optimizer_)

    #@tf.function
    def update_target(self, target_weights, weights, tau):  ####********
        #print(f'***{type(weight)}, {type(target_weights)}')
        #print(f'{len(weights)},{len(target_weights)}')
        #print(f'weights:{weights}, \n\ntar_weights:{target_weights}')
        #print(f'weights:{weights[0].dtype}, \ntar_weights:{target_weights[0].dtype}')
        for (a, b) in zip(target_weights, weights):
            #print(f'*********{a.shape},{type(a)},{b.shape},{b.dtype}*********')
            #exit()
            a.assign(b * tau + a * (1 - tau))

    
    
    def sigmoid_func(self, np_array):
        temp_array = []
        for t in np_array:
            trunc_val = t # float_precision(t, 8)   #np.float128(t) #
            temp_array.append(1 / (1 + np.exp(- trunc_val)))
            #assert temp_array[-1]<0 and temp_array[-1]>0,f'ori value:{t},trun val :{trunc_val}, sig:{1 / (1 + np.exp(- trunc_val))}'
        return np.asarray(temp_array)

    def policy(self, state, noise_object, num_veh=None):
        
        #print(f'SATRT policy**********************************')

        
        # print(f"=======state: {state.shape}, observation: {observation.shape}")
        #state = state.astype(np.float64)
        #print(state)
        #observation = observation.astype(np.float64)

        observation = []
        state_has_nan = np.isnan(state).any()
        #observation_has_nan = np.isnan(observation).any()
        #assert not state_has_nan and not observation_has_nan, f'input values-bad : state: {state}, obs: {observation}'
        assert not state_has_nan , f'input values-bad : state: {state}, obs: {observation}'

        ##########################################
        ##### random input testing #####
        input_tensor = tf.random.normal((64, 1680))
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        ##########################################
        #print(f' state:{state.shape}')

        sampled_actions = tf.squeeze(self.model(state, act_or_cri='act'))
        #assert all([ _>=0 and _<=1 or _<=0 and _>=-1  for _ in sampled_actions[:data_file.num_veh]]),f'alpha value not in range :{sampled_actions[:data_file.num_veh]}, \
        #    signal:{sampled_actions[data_file.num_veh:]}, state:{state}, obs:{observation}'

        if noise_object is not None:
            noise = noise_object()
            noise = np.float32(noise)
            sampled_actions = sampled_actions.numpy() + noise #np.maximum(noise, 0)

        
        #print(f'phase:{sampled_actions[:data_file.num_phases]} phase_len:{len(sampled_actions[:data_file.num_phases])}, alpha:{sampled_actions[data_file.num_phases: (data_file.num_veh+data_file.num_phases)]}, alpha_len:{len(sampled_actions[data_file.num_phases: (data_file.num_veh+data_file.num_phases)])}')
        legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases: (data_file.num_veh+data_file.num_phases)])
        legal_action_ph = sampled_actions[:data_file.num_phases] 
        #print(f'{type(legal_action)},{type(legal_action_ph)}')
        #print("----------------------")
        #print(f'{(legal_action)},{np.size(legal_action)},{(legal_action_ph)},{np.size(legal_action_ph)}')

        
        #legal_actions = np.append(legal_action_ph, legal_action)#[:data_file.num_veh]
        legal_action_set = np.append(legal_action_ph, legal_action)
        #print(f'leact:{legal_action_set}, {np.size(legal_action_set)}')

        assert all([ _>=0 and _<=1  for _ in legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)]]),f'alpha value not in range :{legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)]}, \
            signal:{legal_action_set[:data_file.num_phases]}'

        #print(f'phase: {legal_action_set[:data_file.num_phases]} phase_len:{len(legal_action_set[:data_file.num_phases])}, alpha:{legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)]}, alpha_len:{len(legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)])}')
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
        # Store x into x_prev
        # Makes next noise dependent on current one
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
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = observe_size
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.model_buff = buff_model
        self.model_target_buff =  buff_model_target       
        #self.actor_model = buff_model.__init_subclass__.actor
        #self.critic_model = critic_m
        #self.target_actor = tar_act_m
        #self.target_critic = tar_cri_m
        self.gamma = gamma
        self.tau = tau
        self.critic_optimizer = cri_optimizer
        self.actor_optimizer = act_optimizer

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        #print("statebuff_size",self.buffer_capacity)
        self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        #print("statebuff_size",len(self.state_buffer))
        #self.observe_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        #self.next_observe_buffer = np.zeros((self.buffer_capacity, self.state_size))
        #print("inside buff_class")
        

    # Takes (s,a,r,s') obervation tuple as input
    def remember(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        
        #print(f'state:{obs_tuple[0]}, len:{(np.size(obs_tuple[0]))}')
        assert len(obs_tuple[0])==self.state_size 
        assert len(obs_tuple[1])==self.action_size,f'replay assignment error state:'
        assert len(obs_tuple[3])==self.state_size,f'replay assignment error'
            
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        
        
        
        #print("statebuff_size_in_rem",len(self.buffer_capacity))
        #self.observe_buffer[index] = obs_tuple[1]
        #self.next_observe_buffer[index] = obs_tuple[5]
        #print("inside remember", self.state_buffer[index], "input:", obs_tuple[0])

        self.buffer_counter += 1

    #@tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        
        with tf.GradientTape() as tape:
            target_actions = self.model_target_buff(next_state_batch, act_or_cri='act')
            #print(f'target_actions:{target_actions},{target_actions.shape}')
            y = reward_batch + self.gamma * self.model_target_buff(next_state_batch, target_actions, act_or_cri='critic')
            #print(f'y:{y},{y.shape}')
            critic_value = self.model_buff(state_batch, action_batch, act_or_cri='critic')
            #print(f'critic_value:{critic_value},{critic_value.shape}')
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value),1)

        critic_grad = tape.gradient(critic_loss, self.model_buff.critic_model.trainable_variables)   #.critic_model
        
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.model_buff.critic_model.trainable_variables) 
        )

        with tf.GradientTape() as tape:
            #tf.print("state:", tf.size(state_batch) ," obser", tf.size(obs_batch))
            
            #actions = self.actor_model([state_batch, obs_batch], training=True)
            actions = self.model_buff(state_batch,act_or_cri='act' )
            critic_value = self.model_buff(state_batch , actions, act_or_cri='critic')
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.model_buff.actor_model.trainable_variables)

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.model_buff.actor_model.trainable_variables)   ####********
        )



    # We compute the loss and update parameters
    #@tf.function
    def learn(self):
        

        record_range = min(self.buffer_counter, self.buffer_capacity) # Get index range/sampling range
        batch_indices = np.random.choice(record_range, self.batch_size)# Randomly sample a tuple for each batch
        #print(f'batch_index:{batch_indices},record_range:{record_range}, buff_cap:{self.buffer_capacity}, buff_coun:{self.buffer_counter}, batch_size:{self.batch_size}')   
  
        #tf.print("ddpgup-1")
      
        ### batching process ###
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        #obs_batch = tf.convert_to_tensor(self.observe_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        #reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        #next_obs_batch = tf.convert_to_tensor(self.next_observe_buffer[batch_indices], dtype=tf.float32)
        #print("shaping process", tf.shape(state_batch),"buff:",self.state_buffer[1])
        
        #self.update(state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch)
        #tf.print("ddpgup-2")
        init_time_learn = time.time()
        self.update(state_batch,action_batch, reward_batch, next_state_batch)
        tf.print(f"[IN DDPG]: learning time: {round(time.time() - init_time_learn, 3)}")