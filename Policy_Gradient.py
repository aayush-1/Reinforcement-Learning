import gym
import numpy as np
import random
# The code is in TensorFlow 2.0
import tensorflow as tf
from collections import deque
import argparse
import matplotlib.pyplot as plt


class Agent:
	def __init__(self, env,reward_to_go,advantage_normalization,batch_size):
		# Environment to use
		self.env = env
		self.reward_to_go=reward_to_go
		self.advantage_normalization=advantage_normalization
		# Replay memory
		# self.memory = defque(maxlen=10000)

		# Discount factor
		self.discount_factor = 0.99
		self.learning_rate=0.001
		self.base=np.zeros((1000))
		self.batch_size=batch_size
		self.x=0

		# self.batch_size = 64  
		# self.train_start_size = 1000
		self.state_size = self.env.observation_space.shape
		self.action_size = self.env.action_space.n

		self.states, self.actions, self.rewards = [], [], []
		self.states_batch, self.actions_batch, self.rewards_batch = [], [], []

		# Model being trained
		self.model = self.create_model()
		# Target model used to predict Q(S,A)
		# self.target_model = self.create_model()

	def create_model(self):
		model = tf.keras.Sequential([
			tf.keras.layers.Dense(24,activation='relu',input_shape=self.state_size, kernel_initializer="he_uniform"),
			tf.keras.layers.Dense(24,activation='relu', kernel_initializer="he_uniform"),
			tf.keras.layers.Dense(self.action_size,activation='softmax', kernel_initializer="he_uniform")
			])
		model.compile(loss=self.custom_loss, optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
		return model

	def custom_loss(self,y_actual,y_pred):
		log_action_prob=tf.keras.backend.log(y_pred)
		loss=-log_action_prob*y_actual
		loss=tf.keras.backend.sum(loss,axis=1)
		loss=tf.keras.backend.mean(loss)
		return loss

	def act(self, state):
		policy = self.model.predict(state).flatten()
		# print(policy)
		return np.random.choice(self.action_size, 1, p=policy)[0]

	def remember(self, state, action, reward):
		self.states.append(state)
		self.rewards.append(reward)
		self.actions.append(action)

	def discount_rewards(self, rewards):
		discounted_rewards =[] 
		for i in range(len(rewards)):
			r=np.ones_like(rewards[i])
			if (not self.reward_to_go):
				# print("**************************************************************************************")
				running_add = 0
				for t in reversed(range(0, len(rewards[i]))):
					running_add = running_add * self.discount_factor + rewards[i][t]
				discounted_rewards.append(r*running_add)   

			else:
				running_add = 0
				for t in reversed(range(0, len(rewards[i]))):
					running_add = running_add * self.discount_factor + rewards[i][t]
					r[t] = running_add
				discounted_rewards.append(r)

		if self.advantage_normalization and self.batch_size>1 and self.reward_to_go:
			for i in range(1000):
				sum=0
				for j in range(len(discounted_rewards)):
					if(i<len(discounted_rewards[j])):
						sum=sum+discounted_rewards[j][i]
				self.base[i]=sum/self.batch_size
			# print(self.base)
			for i in range(len(discounted_rewards)):
				for j in range(len(discounted_rewards[i])):
					discounted_rewards[i][j]-=self.base[j]
				# print(i)
				# print(discounted_rewards[i])


				discounted_rewards[i] -= np.mean(discounted_rewards[i])
				discounted_rewards[i] /= np.std(discounted_rewards[i])
				discounted_rewards[i]+=1
		return discounted_rewards

	def train(self):
		if(self.x<self.batch_size):
			self.x+=1
			self.states_batch=self.states_batch+self.states
			self.rewards_batch.append(self.rewards)
			self.actions_batch.append(self.actions)
			self.states, self.actions, self.rewards = [], [], []

		elif(self.x==self.batch_size):
			discounted_rewards = self.discount_rewards(self.rewards_batch)
			# print(discounted_rewards)
			length=len(self.states_batch)
			# update_inputs=[]
			advantages = np.zeros((length, self.action_size))
			b=0
			for i in range(len(self.actions_batch)):
				episode_length = len(self.actions_batch[i])

			# update_inputs = np.zeros((episode_length, self.state_size[0]))
			# advantages = np.zeros((episode_length, self.action_size))

				for j in range(episode_length):
					# update_inputs.append(self.states_batch[i][j][0].reshape(4,1))
					a=np.zeros((self.action_size))
					a[self.actions_batch[i][j]]=discounted_rewards[i][j]
					advantages[b][self.actions_batch[i][j]]=discounted_rewards[i][j]
					b+=1
			inp=[a[0].tolist() for a in self.states_batch]
			self.model.fit(np.array(inp), advantages, epochs=1, verbose=0)
			self.states, self.actions, self.rewards = [], [], []
			self.states_batch, self.actions_batch, self.rewards_batch = [], [], []	
			self.x=0




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', action="store",dest="env",default = 'CartPole-v0')
	parser.add_argument('--batch_size', action="store",dest="batch_size", type=int)
	parser.add_argument('--reward_to_go', default=False,action='store_true')
	parser.add_argument('--advantage_normalization', default=False,action='store_true')
	args = parser.parse_args()
	print("Reward to go == ",args.reward_to_go)
	print("Advantage normalization == ",args.advantage_normalization)
	env = gym.make(args.env)

	trials = 2001
	scores, episodes = [], []
	scores_test=[]
	scores_test_average=[]
	scores_test_max=[]
	dqn_agent = Agent(env=env,reward_to_go=args.reward_to_go,advantage_normalization=args.advantage_normalization,batch_size=args.batch_size)
	test=0
	for trial in range(trials):
		score=0
		cur_state = env.reset().reshape(1, dqn_agent.state_size[0])
		done=False

		while not done:
			# env.render()
			action = dqn_agent.act(cur_state)
			new_state, reward, done, _ = env.step(action)

			new_state = new_state.reshape(1, dqn_agent.state_size[0])
			# reward = reward if not done or score == 499 else -100
			dqn_agent.remember(cur_state, action, reward)
			# print(reward)
			
			score+=reward
			cur_state = new_state
			if done:
				print("episode:", trial, "  score:", score)
				dqn_agent.train()
				env.reset()
				# score = score if score == 500 else score + 100
				scores.append(score)
				episodes.append(trial)
				# if np.mean(scores[-min(20, len(scores)):]) > 180:
				# 	# dqn_agent.save_model("success_model")
				# break
	

		if(trial%50==0 and trial>0):
			scores_t=[]

			for i in range(10):
			# new_model=dqn_agent.load_model()
				cur_state = env.reset().reshape(1, dqn_agent.state_size[0])
				done=False
				score=0
				while not done:
					# env.render()
					action = dqn_agent.act(cur_state)
					new_state, reward, done, _ = env.step(action)
					new_state = new_state.reshape(1, dqn_agent.state_size[0])
					cur_state = new_state
					score+=reward
					# height_max=max(height_max,new_state[0][0])
					if done:
						scores_t.append(score)
						print("Succesfully completed with Reward = {}".format(score))
						break
			scores_test_average.append(np.mean(scores_t))
			scores_test_max.append(np.max(scores_t))
			scores_test.append(trial)
			env.close()

	# plt.figure(1)	
	# plt.plot(episodes, scores, label = "Reward") 
	# plt.xlabel('Iteration') 
	# plt.ylabel('Reward') 
	# plt.legend()
	# plt.title('Learning Curve')
	# plt.savefig("RTG_AD.png") 	

	# plt.figure(2)
	# plt.plot(scores_test, scores_test_average, label = "Average Scores") 
	# plt.plot(scores_test, scores_test_max, label = "Maximum Scores") 
	# plt.xlabel('Iteration') 
	# plt.ylabel('Score') 
	# plt.legend()
	# plt.title('Learning Curves')
	# plt.savefig("Plot.png") 
	# plt.show() 



# if __name__ == "__main__":
#     main()