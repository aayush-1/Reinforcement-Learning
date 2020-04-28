import gym
env = gym.make('LunarLander-v2')
print("LunarLander-v2\nAction Space " , env.action_space)
print("Environment Shape: ",env.observation_space)
print("\n")
for i_episode in range(10):
	observation = env.reset()
	done=False
	x=0
	while not done:
		env.render()
		# print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print("reward  ",reward)
		x+=1
		if done:
			print("******************************************\n")
			print("Episode finished after {} timesteps".format(x))
			print("\n******************************************")
			break
env.close()
