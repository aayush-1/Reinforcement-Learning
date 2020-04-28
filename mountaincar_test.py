import gym
import numpy as np
import random
# The code is in TensorFlow 2.0
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env):
        # Environment to use
        self.env = env
        # Replay memory
        self.memory = deque(maxlen=10000)

        # Discount factor
        self.gamma = 0.99

        # Initial exploration factor
        self.epsilon = 1.0
        # Minimum value exploration factor
        self.epsilon_min = 0.005
        # Decay for epsilon
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 50000

        self.batch_size = 64
        self.train_start_size = 1000
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # Learning rate
        self.learning_rate = 0.001

        # Model being trained
        self.model = self.create_model()
        # Target model used to predict Q(S,A)
        self.target_model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32,activation='relu',input_shape=self.state_size, kernel_initializer="he_uniform"),
            tf.keras.layers.Dense(16,activation='relu', kernel_initializer="he_uniform"),
            tf.keras.layers.Dense(self.action_size,activation='linear', kernel_initializer="he_uniform")
            ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        # Decay exploration rate by epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.train_start_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size[0]))
        update_target = np.zeros((self.batch_size, self.action_size))


        target=np.array(mini_batch)[:,[0,3]]
        A=[]
        B=[]
        for i in range(target.shape[0]):
            a=[]
            a.append(target[i][0][0][0])
            a.append(target[i][0][0][1])
            A.append(a)
            b=[]
            b.append(target[i][1][0][0])
            b.append(target[i][1][0][1])
            B.append(b)

        A=np.array(A)
        update_input=A
        target=self.model.predict(A)   
        C=self.target_model.predict(B)


        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]

            if done:
                target[i][action] = reward
            else:
                target[i][action] = reward + self.gamma * np.amax(C[i])
        update_target = target

        self.model.fit(update_input, update_target,batch_size=self.batch_size, epochs=1, verbose=0)

    def target_train(self):
        # Simply copy the weights of the model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self):
        return tf.keras.models.load_model('trained_model')



def main():
    env = gym.make("MountainCar-v0")

    trials = 10
    trial_len = 500

    dqn_agent = Agent(env=env)
    for trial in range(trials):
        new_model=dqn_agent.load_model()
        cur_state = env.reset().reshape(1, 2)
        height_max=-1e10
        for step in range(trial_len):
            env.render()
            action=np.argmax(new_model.predict(cur_state)[0])
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 2)
            cur_state = new_state
            height_max=max(height_max,new_state[0][0])
            if done:
                print("Succesfully completed with Max_Height = {} and Reward = {}".format(height_max,step))
                break

        env.close()



    X = np.random.uniform(-1.2, 0.6, 10000).reshape(10000,1)
    Y = np.random.uniform(-0.07, 0.07, 10000).reshape(10000,1)
    data=np.hstack((X,Y))
    print(data.shape)
    action=np.argmax(new_model.predict(data),axis=1)
    print(action.shape)

    for i in range(10000):
        if action[i]==0:
            plt.plot(X[i],Y[i],'ro')
        if action[i]==1:
            plt.plot(X[i],Y[i],'bo')
        if action[i]==2:
            plt.plot(X[i],Y[i],'go')
    plt.xlabel('Position') 
    plt.ylabel('Velocity')
    plt.title('Policy : Red- push left : Blue- no push : Green- right push ')
    plt.savefig('Policy.png')
    plt.show()




if __name__ == "__main__":
    main()