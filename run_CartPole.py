"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0

file = open('run_CartPole.txt','w') 
 
file.write('Run Data:\n') 

for i_episode in range(200):

    observation = env.reset()
    ep_r = 0
    t = 0
    while True:
        t = t + 1
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        print("Obs: ",observation_)
        print("Reward: ",reward)
        print("action: ",action)
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'Score: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            file.write('episode: {} ep_r: {} epsilon: {} '.format(i_episode,round(ep_r,2),round(RL.epsilon,2)))
            file.write('timesteps: {}\n'.format(t))
            break

        observation = observation_
        total_steps += 1

file.close() 