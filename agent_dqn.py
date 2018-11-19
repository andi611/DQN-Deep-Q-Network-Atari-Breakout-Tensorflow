# -*- coding: utf-8 -*-
"""*******************************************************************************"""
#	FileName     [ agent_dqn.py ]
#	Synopsis     [ Implementation of a Reinforcement Agent using Deep Q Network ]
#	Author       [ Ting-Wei (Andy) Liu ]
#	Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*******************************************************************************"""


###############
# IMPORTATION #
###############
import os
import pickle
import numpy as np
import tensorflow as tf


class Agent_DQN(object):
	def __init__(self, env, args):
		######################
		# INITIALIZING AGENT #
		######################
		
		# general settings
		tf.reset_default_graph()
		self.env = env
		self.get_path()

		# model settings
		self.double_q = True
		self.n_actions = self.env.action_space.n
		self.feature_shape = self.env.observation_space.shape
		self.n_features = np.prod(np.array(self.feature_shape)) # n_features = 1 * 84 * 84 * 4 = 25767
		self.memory_size = 10000
		self.memory = np.zeros((self.memory_size, 3 + self.n_features*2)) # initialize zero memory [s, a, r, done, s_]
		self.reward_his = []

		# training settings
		self.n_steps = 7e6
		self.learn_every_n_step = 4
		self.save_every_n_episode = 100
		self.update_network_every_n_step = 1000
		self.start_learning_after_n_step = 10000
		self.gamma = 0.99 # reward_decay
		self.batch_size = 32
		self.epsilon_min = 0.07 # e_greedy percentage
		self.epsilon = 1.0
		self.epsilon_decrement = (self.epsilon - self.epsilon_min) / (1000000) # (x) -> x steps before epsilon reaches min
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.99)

		# model
		self.build_model() # consist of [target_net, evaluate_net]
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
		with tf.variable_scope('soft_replacement'):
			self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		# tf initialization
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=3)

		load = bool(0)
		if args.test_dqn or load:
			print('Loading trained model: ' + self.model_path)
			if load: self.reward_his = pickle.load(open(self.reward_his_path, 'rb'))
			try:
				self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=self.model_path))
			except:
				self.saver.restore(self.sess, save_path=os.path.join(self.model_path, 'model_dqn-25581'))


	def get_path(self):
		########################
		# GET PATH FOR STORAGE #
		########################
		directory = './model'
		try:
			if not os.path.exists(directory):
				os.makedirs(directory)
		except:
			print('Failed to create result directory.')
			directory = './'
		self.reward_his_path = os.path.join(directory, 'reward_his_dqn.pkl')
		self.model_path = directory


	def build_model(self):
		########################
		# BUILD DEEP Q NETWORK #
		########################
		# ------------------ all inputs ------------------ #
		n_features_tensor = [None] + [dim for dim in self.feature_shape] # [None, 84, 84, 4]
		#--eval net--#
		self.s = tf.placeholder(tf.float32, n_features_tensor, name='s')  # input State
		#--target net--#
		self.s_ = tf.placeholder(tf.float32, n_features_tensor, name='s_')  # input Next State
		#--for loss--#
		self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

		# ------------------ initializers ------------------ #
		from tensorflow.python.ops.init_ops import VarianceScaling
		def lecun_normal(seed=None):
			return VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)
		def lecun_uniform(seed=None):
			return VarianceScaling(scale=1., mode='fan_in', distribution='uniform', seed=seed)
		w_initializer = lecun_normal()
		b_initializer = tf.zeros_initializer()

		# ------------------ build evaluate_net ------------------ #
		with tf.variable_scope('eval_net'):
			e_conv1 = tf.layers.conv2d(
					inputs=self.s,
					filters=32,
					kernel_size=[8, 8],
					strides=(4, 4),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='e_conv1')
			e_conv2 = tf.layers.conv2d(
					inputs=e_conv1,
					filters=64,
					kernel_size=[4, 4],
					strides=(2, 2),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='e_conv2')
			e_conv3 = tf.layers.conv2d(
					inputs=e_conv2,
					filters=64,
					kernel_size=[3, 3],
					strides=(1, 1),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='e_conv3')
			e_flat = tf.contrib.layers.flatten(e_conv3)
			e_dense1 = tf.layers.dense(
					inputs=e_flat,
					units=512, 
					activation=tf.nn.relu,
					kernel_initializer=w_initializer,
					bias_initializer=b_initializer, 
					name='e_dense1')
			self.q_new = tf.layers.dense(
					inputs=e_dense1, 
					units=self.n_actions,
					activation=None,
					kernel_initializer=w_initializer,
					bias_initializer=b_initializer,
					name='q_new') # q_new shape: (batch_size, n_actions)

		# ------------------ build target_net ------------------ #
		with tf.variable_scope('target_net'):
			t_conv1 = tf.layers.conv2d(
					inputs=self.s_,
					filters=32,
					kernel_size=[8, 8],
					strides=(4, 4),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='t_conv1')
			t_conv2 = tf.layers.conv2d(
					inputs=t_conv1,
					filters=64,
					kernel_size=[4, 4],
					strides=(2, 2),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='t_conv2')
			t_conv3 = tf.layers.conv2d(
					inputs=t_conv2,
					filters=64,
					kernel_size=[3, 3],
					strides=(1, 1),
					padding="same",
					activation=tf.nn.relu6,
					kernel_initializer=w_initializer,
					name='t_conv3')
			t_flat = tf.contrib.layers.flatten(t_conv3)
			t_dense1 = tf.layers.dense(
					inputs=t_flat,
					units=512, 
					activation=tf.nn.relu,
					kernel_initializer=w_initializer,
					bias_initializer=b_initializer, 
					name='t_dense1')
			self.q_old = tf.layers.dense(
					inputs=t_dense1, 
					units=self.n_actions,
					activation=None,
					kernel_initializer=w_initializer,
					bias_initializer=b_initializer,
					name='q_old')

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_new, name='TD_error'))

		with tf.variable_scope('train'):
			self.train_op = self.optimizer.minimize(self.loss)


	def store_transition(self, s, a, r, d, s_):
		#######################
		# STORE REPLAY MEMORY #
		#######################
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((np.reshape(s, [-1]), [a, r, int(d)], np.reshape(s_, [-1]))) # stack arrays in sequence horizontally (column wise)
		
		index = self.memory_counter % self.memory_size # replace the old memory with new memory
		self.memory[index, :] = transition
		self.memory_counter += 1


	def learn(self):
		######################
		# LEARNING PROCEDURE #
		######################
		if self.memory_counter > self.memory_size: sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False) # sample batch memory from all memory
		else: sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[sample_index, :]
		n_features_tensor = [self.batch_size] + [dim for dim in self.feature_shape] # [batch_size, 84, 84, 4]

		#-----------------------------------------------------------------------------------#
		states = np.reshape(batch_memory[:, :self.n_features], newshape=n_features_tensor)
		actions = batch_memory[:, self.n_features].astype(int) # action batch
		rewards = batch_memory[:, self.n_features + 1] # reward batch
		done = batch_memory[:, self.n_features + 2] # done batch
		states_ = np.reshape(batch_memory[:, -self.n_features:], newshape=n_features_tensor)
		#-----------------------------------------------------------------------------------#
		
		q_new, q_old = self.sess.run([self.q_new, self.q_old], feed_dict={ self.s: states, self.s_: states_ })
		q_target = q_new.copy()
		batch_index = np.arange(self.batch_size, dtype=np.int32) # [0, 1, 2, ... batch_size]

		#--------------------------------#
		if self.double_q:
			q_new4next = self.sess.run(self.q_new, feed_dict={ self.s: states_ }) # next observation for double Q
			max_act4next = np.argmax(q_new4next, axis=1)        # the action that brings the highest value is evaluated by q_new
			selected_q_old = q_old[batch_index, max_act4next]  # Double DQN, select q_old depending on above actions
		else:
			selected_q_old = np.max(q_old, axis=1)    # the natural DQN
		#--------------------------------#

		q_target[batch_index, actions] = rewards + (1-done) * self.gamma * selected_q_old # change q_target w.r.t q_new's action

		_, loss = self.sess.run([self.train_op, self.loss], feed_dict={ self.s: states, self.q_target: q_target })
		return loss


	def train(self):
		######################
		# TRAINING ALGORITHM #
		######################
		episode = 0
		step = 0
		loss = 9.9999
		rwd_avg_max = 0

		while step < self.n_steps:
			
			observation = self.env.reset() # initial observation
			done = False
			episode_reward = 0.0

			while not done:	
			#-------------------------------------------------------------------------------------------------------------#	
				action = self.make_action(observation, test=False) # choose action based on observation	
				observation_, reward, done, info = self.env.step(action) # take action and get next observation and reward
				episode_reward += reward


				self.store_transition(observation, action, reward, done, observation_)
				if (step > self.start_learning_after_n_step) and (step % self.learn_every_n_step == 0):
					loss = self.learn()
				if (step > self.start_learning_after_n_step) and (step % self.update_network_every_n_step == 0):
					self.sess.run(self.target_replace_op)


				print('Step: %i,  Episode: %i,  Action: %i,  Reward: %.0f,  Epsilon: %.5f, Loss: %.5f' % (step, episode, action, reward, self.epsilon, loss), end='\r')
				self.epsilon = self.epsilon - self.epsilon_decrement if self.epsilon > self.epsilon_min else self.epsilon_min # decreasing epsilon
				observation = observation_
				step += 1
			#-------------------------------------------------------------------------------------------------------------#	

			# save
			episode += 1
			self.reward_his.append(episode_reward)
			if step < 1000000:
				if (episode % self.save_every_n_episode == 0):
					pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
					self.saver.save(self.sess, os.path.join(self.model_path, 'model_dqn'), global_step=episode)
			else:
				rwd_avg = np.mean(self.reward_his[-17:])
				if rwd_avg > rwd_avg_max:
					pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)
					self.saver.save(self.sess, os.path.join(self.model_path, 'model_dqn'), global_step=episode)
					rwd_avg_max = rwd_avg
					print()
					print()
					print("Saving best model with avg reward: ", rwd_avg_max)
					print()
				if (episode % self.save_every_n_episode == 0):
					pickle.dump(self.reward_his, open(self.reward_his_path, 'wb'), True)

			print(end='\r')
			print('Step: %i/%i,  Episode: %i,  Action: %i,  Episode Reward: %.0f,  Epsilon: %.2f, Loss: %.5f' % (step, self.n_steps, episode, action, episode_reward, self.epsilon, loss))


	def make_action(self, observation, test=True):
		##################
		# PREDICT ACTION #
		##################
		"""
		Input:
			observation: np.array
				stack 4 last preprocessed frames, shape: (84, 84, 4)

		Return:
			action: int
				the predicted action from trained model
		"""
		observation = np.expand_dims(observation, axis=0) # to have batch dimension when feed into tf placeholder
		if test: self.epsilon = 0.01
		if (np.random.uniform() > self.epsilon):
			actions_value = self.sess.run(self.q_new, feed_dict={self.s: observation}) # forward feed the observation and get q value for every actions
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0, self.n_actions)
		return action # self.env.get_random_action()


	def plot(self):
		##############################
		# PLOT LEARNING REWARD CURVE #
		##############################
		import matplotlib.pyplot as plt
		if np.sum(self.reward_his) == 0: self.reward_his = pickle.load(open(self.reward_his_path, 'rb'))
		avg_rwd = []
		for i in range(len(self.reward_his)):
			if i < 30:
				avg_rwd.append(np.mean(self.reward_his[:i]))
			else:
				avg_rwd.append(np.mean(self.reward_his[i-30:i]))
		plt.plot(np.arange(len(avg_rwd)), avg_rwd)
		plt.ylabel('Average Reward in Last 30 Episodes')
		plt.xlabel('Number of Episodes')
		plt.show()


