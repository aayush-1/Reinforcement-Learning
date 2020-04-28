import gym
env = gym.make('MountainCar-v0')
print("LunarLander-v2\nAction Space " , env.action_space)
print("Environment Shape: ",env.observation_space)
print("\n")
for i_episode in range(1):
	observation = env.reset()
	for t in range(200):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		# print("reward  ",reward)
		if done:
			print("******************************************\n")
			print("Episode finished after {} timesteps".format(t+1))
			print("\n******************************************")
			break
env.close()
