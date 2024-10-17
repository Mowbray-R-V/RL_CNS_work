# pylint: disable=unused-import
import csv
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import  GRU
import tensorflow as tf
import numpy as np
import time
import copy 
import matplotlib.pyplot as plt
import data_file
import math
from pprint import pprint 


tf.config.experimental_run_functions_eagerly(True)

action_list = []
action_sig_list = []
gradient_list = []
def float_precision(value, precision):
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
        self.gru = layers.Bidirectional(GRU(lat_size, return_sequences=True, return_state=True))
        self.state_mask = layers.Masking(mask_value=0.0)
        self.FC1 = tf.keras.layers.Dense(64, activation='relu')
        self.FC2 = tf.keras.layers.Dense(48, activation=None)
        self.FC3 = tf.keras.layers.Dense(32, activation=None)
        self.embed_concat = layers.Concatenate()
        
    def build(self, input_shape):
        super().build(input_shape)

    @tf.function  
    def call(self, inputs):
        
        inputs = self.reshape(inputs)
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = self.reshape_1(inputs)
        #print(f'inpt:{inputs.shape}')
        #print(f'inpt:{inputs}')s
        masked_inputs = self.state_mask(inputs)
        mask = self.state_mask.compute_mask(inputs)
        #print(f'mask:{mask.shape}**********')
        
        seq_output, forward_state, backward_state = self.gru(inputs, mask=mask)
        #print(f' \n forward: {forward_state.shape}, back:{backward_state.shape},{seq_output.shape}')
        seq_output = self.FC3(self.FC2(self.FC1(seq_output)))
        #print(f' \n forward: {forward_state.shape}, back:{backward_state.shape},{seq_output.shape}')
        #embed_state = seq_output[:,data_file.num_veh-1,:]     
        
        #print(f'for_{forward_state.shape},back: {backward_state.shape}')
        embed_state = self.FC3(self.FC2(self.FC1(tf.concat([forward_state, backward_state], axis=-1))))
        #print(f'embed state:{embed_state.shape}')
        return seq_output, embed_state, mask

class NNModel(tf.keras.Model): 
    def __init__(self, lat_size): 
        super().__init__()
        self.RNN = SeqModule(lat_size)
        self.lat_size = lat_size
        self.actor_model = ActorModel(self.lat_size)
        self.critic_model = CriticModel(self.lat_size)


    #def build(self, input_shape):
    #    self.RNN.build(input_shape)
    #    self.actor_model.build([(None,112,128),(None,128)])
    #    self.critic_model.build([(None,128),(None,124)])

    @tf.function    
    def call(self, input,time_, action=None,act_or_cri = 'act'): 
        assert input.shape[0] == data_file.rl_ddpg_samp_size or input.shape[0] == 1, f'wrong dimensions, input_batch: {input.shape[0]}'
        seq, latent, mask = self.RNN(input)
        #print(f'mask:{mask.shape},{input.shape}****************')
        #print(act_or_cri)
        if act_or_cri == 'act': 
            return self.actor_model(seq, latent, mask, time_)
        elif act_or_cri == 'critic': 
            #print(f'action_input from critic:{action.shape}')
            return  self.critic_model(latent, action)
        else: print(FFFFFFFF)

class CNN(layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, activation_inp=None):
        super(CNN, self).__init__()
        self.conv = layers.Conv1D(kernel_num, kernel_size=kernel_size, strides=strides, activation=activation_inp)
        self.kernel = kernel_num

    @tf.function
    def call(self, input):
        y = tf.reshape(input, [input.shape[0], input.shape[1] * input.shape[2], 1])
        y = self.conv(y)
        return y   

class ActorModel(tf.keras.Model):
    def __init__(self,lat_size):
        super(ActorModel, self).__init__()
        
        self.layer_conv48 = CNN(48, 76, 76, activation_inp='relu')
        self.layer_conv24 = CNN(24, 48, 48)
        self.layer_conv12 = CNN(12, 24, 24, activation_inp=None)
        self.layer_conv6 = CNN(6, 12, 12, activation_inp=None)
        self.layer_conv1 = CNN(1, 6, 6)
        self.output_concat = layers.Concatenate()
        self.flat = layers.Flatten()
        self.phase_dl_1 = tf.keras.layers.Dense(64, activation='relu')
        self.phase_dl_2 = tf.keras.layers.Dense(64, activation='relu')
        self.phase_dl_3 = tf.keras.layers.Dense(16, activation=None)
        self.phase_dl_4 = tf.keras.layers.Dense(12, activation=None)
        
        self.action_out = tf.keras.layers.Dense(1)
        self.gru_action = layers.Bidirectional(GRU(lat_size, return_sequences=True, return_state=False))
        self.gru_action_1 = GRU(1, return_sequences=True, return_state=True)
        

    @tf.function
    def call(self, obs, latent, mask,time_):


        with open('hidden_values_record.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time'])
            writer.writerow({time_})
            # writer.writerow({time.time()})
            # writer.writerow(['mask'])   
            # writer.writerow({int(tf.reduce_sum(tf.cast(mask, dtype=tf.float32)[0]))})  
            # writer.writerow(['state Values'])  
            # writer.writerow(input[0][:int(tf.reduce_sum(tf.cast(mask, dtype=tf.float32)[0]))*15].numpy().tolist())  
            writer.writerow(['latent Values'])  
            writer.writerow(latent[0].numpy().tolist())
            writer.writerow(['obs Values'])  
            writer.writerow(obs[0][0].numpy().tolist())

        filename = 'hidden_values.csv'
        if time_<94:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['state Values'])  
                writer.writerow(latent[0].numpy().tolist())
                writer.writerow(['obs Values'])  
                writer.writerow(obs[0][0].numpy().tolist())


        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  
            prev_state = [float(value) for value in next(reader)]
            header = next(reader)  
            prev_obs = [float(value) for value in next(reader)]
            # print(f'prev_state:{len(prev_state)},{type(prev_state[1])},{prev_state[1]},{prev_state[2]}')
            # print(f'prev_state:{len(prev_obs)},{type(prev_obs[1])},{prev_obs[1]},{prev_obs[2]}')
        
        
        diff_obs = []
        diff_state = []
        #print(f'************:{obs[0][2].numpy().tolist()}')        
        #print(f'************:{prev_obs}')  
        
        for iter1 in range(len(obs[0][0].numpy().tolist())):
            diff_obs.append(obs[0][0].numpy().tolist()[iter1] - prev_obs[iter1])

        # print(f'************:{len(diff_obs)}')        
        # print(f'original_{len(prev_obs)},{len(obs[0][2].numpy().tolist())}\n')
        
        
        for iter1 in range(len(latent[0].numpy().tolist())):
            diff_state.append(latent[0].numpy().tolist()[iter1] - prev_state[iter1])
                
                
        #diff_obs = [iter_1 - iter_2   for iter_1 in obs[0][2].numpy().tolist() for iter_2  in prev_obs]  
        # diff_state = [latent[0].numpy().tolist()[iter_1] - prev_state[iter_1] for iter_1 in range(len(latent[0].numpy().tolist()))] 
        
        
        # print(f'prev_value_file_{len(prev_obs)},{len(prev_state)}\n')
        # print(f'diff_{len(diff_state)},{len(diff_obs)}\n')
        # print(f'current_{latent.shape},{obs.shape}\n')
        
        
        
        
        l2_obs = tf.norm(diff_obs) 
        lmax_obs = tf.reduce_max(diff_obs)
        l2_state = tf.norm(diff_state)
        lmax_state = tf.reduce_max(diff_state)
        
        f = open("debug.txt", "a")
        f.write(f'state_l2:{l2_state}, state_max:{lmax_state},obs_l2:{l2_obs}, obs_max:{lmax_obs}\n')
        f.close()
        
        #if time_<19:
            #print(f'state:{latent[0].numpy().tolist()},{type(latent[0].numpy().tolist())}')
        if time_>=94:
            #print(f'*****************time:{time}')
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['state Values'])  
                writer.writerow(latent[0].numpy().tolist())
                writer.writerow(['obs Values'])  
                writer.writerow(obs[0][0].numpy().tolist())

        
        latent = tf.expand_dims(latent, axis=1)
        #tf.print(f'lat_1:{latent.shape}')
        signal =  self.phase_dl_4(self.phase_dl_3(self.phase_dl_2(self.phase_dl_1(latent))))
        # 32 to 64 relu,  64 to 64 relu,  64 to 16, 16 to 12   
        #tf.print(f'sig{signal.shape}')
        signal_cop =tf.repeat(signal, repeats=data_file.num_veh, axis=1)
        #tf.print(f'sig_cop{signal_cop.shape}')
        state = tf.repeat(latent, repeats=data_file.num_veh, axis=1)
        #tf.print(f'state: {state.shape}')
        input = self.output_concat([state, obs, signal_cop])
        #tf.print(f'input to control{input.shape}') 


        action_seq,action_dim = self.gru_action_1((self.gru_action(input, mask=mask)), mask=mask)
        #print(f'actiion_seq{action_seq.shape}') 
        
        control = self.flat(action_seq)
        
        f = open("debug.txt", "a")
        f.write(f'action_before_sigmoid:{control[0][0]}\n')
        f.close()
        
    

        #print(f'control :{control.shape}')
        signal = self.flat(signal)
        #print(f'signal :{signal.shape}')
        outputs = self.output_concat([signal, control])
        #print(f'output :{outputs.shape}')
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
        self.action_mask = layers.Masking(mask_value=-190.392)
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(64, activation=None)
        self.fc4 = tf.keras.layers.Dense(16, activation=None)
        self.fc5 = tf.keras.layers.Dense(8, activation=None)
        self.fc6 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, latent_state, action):
        #print("*****************inside critic")
        #print(f'***************action:{action.shape},{action}')
        action = tf.expand_dims(action, axis=-1)
        masked_inputs = self.action_mask(action)
        #print(f'***************mask_action:{masked_inputs.shape},{masked_inputs}')
        mask = self.action_mask.compute_mask(action)
        #print(f'***************mask_action:{mask.shape},{mask}')
        assert mask.shape[0] == action.shape[0], f'{mask.shape[0]}, {action.shape[0]}'
        norm_action = self.lay_norm(action, mask)
        #print(f'****norm_action{norm_action.shape},{norm_action}')
        norm_action = tf.expand_dims(norm_action, axis=-1)
        #print(f'****norm_action{norm_action.shape},{norm_action}')
        latent_act = self.gru_action(norm_action, mask=mask)
        #print(f'****latent_act{latent_act.shape},{latent_act}')
        concat = self.output_concat([latent_state, latent_act])
        #print(f'****concat{concat.shape},{concat}')
        Q_val = self.fc4(concat)
        #print(f'****Q_val{Q_val.shape},{Q_val}')
        Q_val = self.fc5(Q_val)
        Q_val = self.fc6(Q_val)
        #print(f'****Q_val{Q_val.shape},{Q_val}')
        #exit()
        return Q_val


class DDPG:

    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0, polyak_factor=0, buff_size=1000, samp_size=64):
        self.num_states =  (data_file.num_features+(data_file.num_lanes*4))*data_file.num_veh    #(data_file.num_features*data_file.num_lanes*4)*data_file.num_veh
        self.num_obs = data_file.num_features
        self.num_actions = data_file.num_veh + data_file.num_phases
        self.noise_std_dev = noise_std_dev
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
        self.model = NNModel(32)
        self.target_model = NNModel(32)
        
        self.target_model.set_weights(self.model.get_weights()) ####********        
        buid_ouput = self.model(tf.random.uniform((64, self.num_states)),1, act_or_cri= 'act')##
        buid_ouput_1 = self.model(tf.random.uniform((64, self.num_states)), 1,tf.random.uniform((64, self.num_actions)), act_or_cri= 'critic')
        
        f = open("hidden_values.csv", "w+")
        f.close()
        
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

    def policy(self, state, time, noise_object, num_veh=None):

        observation = []
        state_has_nan = np.isnan(state).any()
        #observation_has_nan = np.isnan(observation).any()
        #assert not state_has_nan and not observation_has_nan, f'input values-bad : state: {state}, obs: {observation}'
        assert not state_has_nan , f'input values-bad : state: {state}, obs: {observation}'

        # if time ==23 or time==40:
        #     print(f"state.shape: {state.shape}")
        #     pprint(f'state values of robot 0: {state[0][:num_veh*15]}')


        ##########################################
        ##### random input testing #####
        input_tensor = tf.random.normal((64, 1680))
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        ##########################################
        

        
        #print(f'state:{state.shape}')
        sampled_actions = tf.squeeze(self.model(state, time, act_or_cri = 'act'))

        if noise_object is not None:
            noise = noise_object()
            noise = np.float32(noise)
            
            #print(f'action:{sampled_actions},{noise}')
            sampled_actions = sampled_actions.numpy() + noise #np.maximum(noise, 0)

        
        legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases: (data_file.num_veh+data_file.num_phases)])
        legal_action_ph = sampled_actions[:data_file.num_phases] 
        legal_action_set = np.append(legal_action_ph, legal_action)
        
        
        f = open("debug.txt", "a")
        f.write(f'action_after_sigmoid:{legal_action[0]}')
        f.close()
        
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
        # Number of "experiences" to store at max
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

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))

    def remember(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        
        assert len(obs_tuple[0])==self.state_size 
        assert len(obs_tuple[1])==self.action_size,f'replay assignment error state:'
        assert len(obs_tuple[3])==self.state_size,f'replay assignment error'
            
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        
        weights_crtic_loss = [self.model_buff.RNN.trainable_variables,self.model_buff.critic_model.trainable_variables]
        weights_crtic_loss =[_ for __ in range(len(weights_crtic_loss)) for _ in weights_crtic_loss[__] ]
        weights_actor_loss = [self.model_buff.RNN.trainable_variables,self.model_buff.actor_model.trainable_variables]
        weights_actor_loss =[_ for __ in range(len(weights_actor_loss)) for _ in weights_actor_loss[__] ]

        with tf.GradientTape() as tape:
            target_actions = self.model_target_buff(next_state_batch, act_or_cri='act')
            y = reward_batch + self.gamma * self.model_target_buff(next_state_batch, target_actions, act_or_cri='critic')
            critic_value = self.model_buff(state_batch, action_batch, act_or_cri='critic')
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value),1)
        critic_grad = tape.gradient(critic_loss,weights_crtic_loss)  
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, weights_crtic_loss) 
        )
        
        with tf.GradientTape() as tape:
            #actions = self.actor_model([state_batch, obs_batch], training=True)
            actions = self.model_buff(state_batch,act_or_cri='act' )
            critic_value = self.model_buff(state_batch , actions, act_or_cri='critic')
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, weights_actor_loss)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, weights_actor_loss)  
        )

    @tf.function
    def learn(self):

        record_range = min(self.buffer_counter, self.buffer_capacity) # Get index range/sampling range
        batch_indices = np.random.choice(record_range, self.batch_size)# Randomly sample a tuple for each batch
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        #learn_init_time = time.time()
        self.update(state_batch,action_batch, reward_batch, next_state_batch)
        #print(f"[update.py]: learning time: {round(time.time() - learn_init_time, 3)}")
