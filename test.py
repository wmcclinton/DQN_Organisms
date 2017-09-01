from Environment import Environment
from RL_brain import DeepQNetwork
import tensorflow as tf
import os

env = Environment()

RL = DeepQNetwork(n_actions=5,
                  n_features=6,
                  learning_rate=0.01, reward_decay=1, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.01,)

#For Loading
status = input("Load (y/n): ")
if(status == "y"):
    status = input("Load Folder (1,2): ")

    if(status == "2"):
        RL.saver.restore(RL.sess, "save/model2.ckpt")
        print("Model 2 restored")
    else:
        RL.saver.restore(RL.sess, "save/model.ckpt")
        print("Model restored")

status = input("Save Folder (1,2): ")



total_steps = 0
save_iter = 12

file = open('test.txt','w') 
 
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
                  'score: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            file.write('episode: {} score: {} epsilon: {} '.format(i_episode,round(ep_r,2),round(RL.epsilon,2)))
            file.write('timesteps: {}\n'.format(t))
            break
        
        observation = observation_
        total_steps += 1
    
    #For Saving
    if i_episode % save_iter == 0 and i_episode != 0:
        if(status == "2"):
            RL.saver.save(RL.sess, "save/model2.ckpt")
            print("Model 2 saved")
        else:
            RL.saver.save(RL.sess, "save/model.ckpt")
            print("Model saved")

file.close() 