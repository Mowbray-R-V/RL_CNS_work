# pylint: disable=unused-import
import csv
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import  GRU,Dense
import tensorflow as tf
import numpy as np
import time
import copy 
import matplotlib.pyplot as plt
import data_file
import math
from pprint import pprint 


# tf.config.experimental_run_functions_eagerly(True)

action_list = []
action_sig_list = []
gradient_list = []
def float_precision(value, precision):
    float128_value = np.float128(value)
    truncated_value = np.round(float128_value, precision)
    result = float(truncated_value)
    return result

def extract_phases(flattened_state):
        # Define phase combinations
        phase_combinations = [
            (1, 2), (1, 7), (1, 5), (1, 10),
            (2, 8), (4, 10), (4, 5), (4, 8),
            (5, 11), (7, 8), (7,11), (10, 11)
        ]

        phases = []
        features=[]
        # i=0
        # masks = []
        # print('flattend',len(flattened_state))
        for flat in flattened_state:
            
            phase_2=[]
            feature_1=[]
            
            for lane1, lane2 in phase_combinations:
                if len(flat)==12:
                    phase_1 = np.concatenate((flat[lane1], flat[lane2]))
                    # print('flat[lane1]',flat[lane1])
                elif len(flat)==1 and len(flat[0])==12:
                    phase_1 = np.concatenate((flat[0][lane1], flat[0][lane2]))
                    # print('flat[lane1]',flat[0][lane1])
                # Pad the phase to ensure the same length (if needed)
                current_length = len(phase_1) // 18 #data_file.num_features
                max_vehicles_per_phase = 14 * 2
                if current_length < max_vehicles_per_phase:
                    padding_needed = (max_vehicles_per_phase - current_length) * 18 
                    phase_1 = np.pad(phase_1, (0, padding_needed), mode='constant', constant_values=0)


                phase_2.append(phase_1.reshape(-1, 18))#data_file.num_features))

            if len(flat)==12:
                for key in flat.keys() :
                    feature_1=np.concatenate((feature_1,flat[key]))
            elif len(flat)==1 and len(flat[0])==12:
                for key in flat[0].keys() :
                    feature_1=np.concatenate((feature_1,flat[0][key]))
            # print('features',features)
            if len(feature_1)<112*18:
                # print('len',len(feature_1))
                feature_1=np.pad(feature_1,(0,112*18 - len(feature_1)),mode='constant',constant_values=0.0)
            # print('len',len(feature_1))

            phases.append(np.asarray(phase_2))

            # print('phase_2',np.asarray(phase_2).shape)
            # print('f2',np.asarray(feature_1).reshape(-1,18).shape)
            features.append(np.asarray(feature_1).reshape(-1,18))
        return np.asarray(phases),np.asarray(features)#, np.asarray(masks)


class NNModel(tf.keras.Model): 
    def __init__(self, lat_size): 
        super().__init__()
        # self.RNN = SeqModule(lat_size)
        self.lat_size = lat_size
        self.actor_model = ActorModel()
        self.critic_model = CriticModel(self.lat_size)
  
    @tf.function    
    def call(self,features,input=None, action=None, act_or_cri='act'):
        assert features.shape[0] == data_file.rl_ddpg_samp_size or features.shape[0] == 1, f'wrong dimensions, input_batch: {features.shape[0],features.shape}'
        # seq, latent = self.RNN(input)
        if act_or_cri == 'act': 
            # mask = (input!= 0).any(axis=-1)
            y_hat = self.actor_model(input, features)
            return y_hat
        
        elif act_or_cri == 'critic': 
            y_c =  self.critic_model(features, action)
            return y_c
            exit()
        else: print(FFFFFFFF)

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

class ActorModel(tf.keras.Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.robot_dense3=Dense(18)#, activation='relu')
        self.robot_dense4=Dense(36)#, activation='relu')
        self.robot_dense = Dense(9)#, activation='linear')
        self.robot_dense2=Dense(1)#,activation='tanh')
        self.unit=UnitVectorLayer()
    @tf.function
    def call(self, inputs,features,mask=None):
        phase_scores = []
        for i in range(inputs.shape[1]):
            phase_input = inputs[:, i, :, :]  # Shape: (batch_size, num_robots, num_features)

            phase_mask = tf.reduce_any(phase_input != 0, axis=-1)  # Shape: (batch_size, num_robots)

            # Apply the dense layer to each robot's features
            robot_scores = self.robot_dense2(self.robot_dense(self.robot_dense3(self.robot_dense4(phase_input))))  # Shape: (batch_size, num_robots, 1)
            # print('robot_scores',robot_scores.shape)  
            masked_scores = tf.where(tf.expand_dims(phase_mask, axis=-1), robot_scores, tf.zeros_like(robot_scores))
            # print('masked_scores',masked_scores.shape)
            phase_score = tf.reduce_sum(masked_scores, axis=1)  # Shape: (batch_size, 1)
            phase_scores.append(phase_score)

        phase_scores = tf.stack(phase_scores, axis=1)  # Shape: (batch_size, num_phases, 1)
        phase_scores=tf.squeeze(phase_scores,axis=-1)
        phase_scores_norm=self.unit(phase_scores)
        return phase_scores_norm


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

        return normalized_inputs

class CriticModel(tf.keras.Model):
    def __init__(self, lat_size):
        super(CriticModel, self).__init__()
        self.output_concat = layers.Concatenate()
        self.lay_norm = CustomLayerNormalization()
        self.gru_action = GRU(lat_size)
        self.gru_latent = GRU(lat_size)
        self.action_mask = layers.Masking(mask_value=-190.392)
        # self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32)#, activation='relu')
        self.fc4 = tf.keras.layers.Dense(16, activation='relu')
        # self.fc5 = tf.keras.layers.Dense(8)#, activation='relu')
        self.fc6 = tf.keras.layers.Dense(1)#, activation='relu')

    @tf.function
    def call(self, latent_state, action):
        action = tf.expand_dims(action, axis=-1)
        masked_inputs = self.action_mask(action)
        #print(f'***************mask_action:{masked_inputs.shape},{masked_inputs}')
        mask = self.action_mask.compute_mask(action)
        #print(f'***************mask_action:{mask.shape},{mask}')
        assert mask.shape[0] == action.shape[0], f'{mask.shape[0]}, {action.shape[0]}'
        norm_action = self.lay_norm(action, mask)
        # print(f'****norm_action{norm_action.shape}')
        norm_action = tf.expand_dims(norm_action, axis=-1)
        # print(f'****norm_action{norm_action.shape}')
        latent_act = self.gru_action(norm_action, mask=mask)


        mask_latent = tf.reduce_any(tf.not_equal(latent_state, 0), axis=-1)

        latent_state=self.gru_latent(latent_state, mask=mask_latent)

        # print(f'****latent_act{latent_act.shape}')
        # print(f'****latent_state{latent_state.shape}')
        concat = self.output_concat([latent_state, latent_act])
        #print(f'****concat{concat.shape},{concat}')
        Q_val = self.fc2(concat)
        #print(f'****Q_val{Q_val.shape},{Q_val}')
        Q_val = self.fc3(Q_val)
        Q_val = self.fc4(Q_val)
        Q_val = self.fc6(Q_val)
        # print(f'****Q_val{Q_val.shape},{Q_val}')
    
        return Q_val




class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):

        self.num_states =  (data_file.num_features+(data_file.num_lanes*4))*data_file.num_veh    #(data_file.num_features*data_file.num_lanes*4)*data_file.num_veh
        self.num_obs = data_file.num_features
        self.num_actions = data_file.num_phases
        self.noise_std_dev = noise_std_dev
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
        self.model = NNModel(32)
        self.target_model = NNModel(32)
        
        self.target_model.set_weights(self.model.get_weights()) ####********        
        buid_ouput = self.model(tf.random.uniform((1,112,18)), input=tf.random.uniform((1,12,28,18)), act_or_cri='act')##
        buid_ouput_1 = self.model(features=tf.random.uniform((1,112,18)), action = tf.random.uniform((1,124)),act_or_cri='critic')
         
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




    def policy(self, state, noise_object, num_veh=None):

        observation = []

        def has_non_empty_list(d):
            for value in d.values():
                if isinstance(value, list) and len(value) > 0:
                    return True
            return False
        
        assert has_non_empty_list(state[0]) , f'input values-bad : state: {state}, obs: {observation}'


        ##########################################
        ##### random input testing #####
        input_tensor = tf.random.normal((64, 1680))
        state,feature=extract_phases(state)

        if len(feature.shape)==2:
            feature=np.expand_dims(feature,axis=0)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        feature=tf.convert_to_tensor(feature, dtype=tf.float32)
        ##########################################
        
        
        sampled_actions = tf.squeeze(self.model(feature,input=state, act_or_cri='act'))

        if noise_object is not None:
            noise = noise_object()
            noise = np.float32(noise)
            
            #print(f'action:{sampled_actions},{noise}')
            sampled_actions = sampled_actions.numpy() + noise #np.maximum(noise, 0)

        
        legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases: (data_file.num_veh+data_file.num_phases)])
        legal_action_ph = sampled_actions[:data_file.num_phases] 
        legal_action_set = np.append(legal_action_ph, legal_action)
        
        
        assert all([ _>=0 and _<=1  for _ in legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)]]),f'alpha value not in range :{legal_action_set[data_file.num_phases:(data_file.num_veh+data_file.num_phases)]}, \
            signal:{legal_action_set[:data_file.num_phases]}'

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

        self.state_buffer = np.zeros((self.buffer_capacity,1),dtype=object)
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity,1),dtype=object)

    def remember(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        
        assert len(obs_tuple[0])==1, f'obs_tuple:{len(obs_tuple[0])},state_size:{self.state_size}'
        assert len(obs_tuple[1]) == self.action_size,f'replay assignment error state:'
        assert len(obs_tuple[3])==1,f'replay assignment error'
             
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch,action_batch, reward_batch, next_state_batch):
        
        weights_crtic_loss = [self.model_buff.critic_model.trainable_variables]
        weights_crtic_loss =[_ for __ in range(len(weights_crtic_loss)) for _ in weights_crtic_loss[__] ]
        weights_actor_loss = [self.model_buff.actor_model.trainable_variables]
        weights_actor_loss =[_ for __ in range(len(weights_actor_loss)) for _ in weights_actor_loss[__] ]
       
        with tf.GradientTape() as tape:
            target_actions = self.model_target_buff(next_features,input=next_state_batch, act_or_cri='act')
            y = reward_batch + self.gamma * self.model_target_buff(next_features, action=target_actions, act_or_cri='critic')
            critic_value = self.model_buff(features, action=action_batch, act_or_cri='critic')
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value),1)
        critic_grad = tape.gradient(critic_loss,weights_crtic_loss)  
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, weights_crtic_loss) 
        )
        
        with tf.GradientTape() as tape:
            #actions = self.actor_model([state_batch, obs_batch], training=True)
            actions = self.model_buff(features,input=state_batch,act_or_cri='act' )
            critic_value = self.model_buff(features ,action= actions, act_or_cri='critic')
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, weights_actor_loss)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, weights_actor_loss)  
        )

    @tf.function
    def learn(self):

        record_range = min(self.buffer_counter, self.buffer_capacity) # Get index range/sampling range
        batch_indices = np.random.choice(record_range, self.batch_size)# Randomly sample a tuple for each batch

        # state_batch,features=extract_phases(self.state_buffer[batch_indices])
        # exit()
        # next_state_batch,next_features=extract_phases(self.next_state_buffer[batch_indices])

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        # features=tf.convert_to_tensor(features, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        # next_features=tf.convert_to_tensor(next_features, dtype=tf.float32)
        #learn_init_time = time.time()
        self.update(state_batch,action_batch, reward_batch, next_state_batch)
        #print(f"[update.py]: learning time: {round(time.time() - learn_init_time, 3)}")
