from Environment import Environment
from RL_brain import DeepQNetwork

env = Environment()

RL = DeepQNetwork(n_actions=5,
                  n_features=8,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0

file = open('run_CartPole.txt','w') 
 
file.write('Run Data:\n') 

for i_episode in range(2000000):

    observation = env.reset()
    ep_r = 0
    t = 0
    while True:
        t = t + 1
        env.render()
        action = RL.choose_action(observation)

        observation_, reward, done = env.step(action)
        #print(done)
        #print("Obs: ",observation_)
        #print("Reward: ",reward)
        #print("action: ",action)

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