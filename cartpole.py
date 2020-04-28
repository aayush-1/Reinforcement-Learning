import gym
env = gym.make('CartPole-v0')
print("CartPole-v0\nAction Space " , env.action_space)
print("Environment Shape: ",env.observation_space)
print("\n")
for i_episode in range(2):
	observation = env.reset()
	for t in range(200):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("******************************************\n")
			print("Episode finished after {} timesteps".format(t+1))
			print("\n******************************************")
			break
env.close()
