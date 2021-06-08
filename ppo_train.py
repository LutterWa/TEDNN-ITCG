import random
import time
import math
import os
import numpy as np
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from scipy.io import savemat
from collect_data.missile import MISSILE
from collect_data.missile_TrEN import MISSILE_TrEN
from corrector_module.ppo_model import PPO

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度

BATCH_SIZE = 32  # update batch size
TRAIN_EPISODES = 1000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
GAMMA = 0.99
REWARD_SAVE_CASE = 0
dnn = keras.models.load_model("./predictor_module/flight_model4.h5")
# dnn = keras.models.load_model("./predictor_module/flight_TrEN.h5")


class OMissile(MISSILE):
    def __init__(self):
        super().__init__(k=5.0)
        dnn_state = np.array([self.Y[1] / 1e2, self.Y[2], self.R / 1e4, self.q])[np.newaxis, :]
        self.tgo = max(float(dnn.predict([dnn_state, dnn_state, dnn_state, dnn_state, dnn_state],
                                         use_multiprocessing=True)), 1e-5)

    def get_tgo(self):
        dnn_state = np.array([self.Y[1] / 1e2, self.Y[2], self.R / 1e4, self.q])[np.newaxis, :]
        self.tgo = max(float(dnn.predict([dnn_state, dnn_state, dnn_state, dnn_state, dnn_state],
                                         use_multiprocessing=True)), 1e-5)
        return self.tgo

    def get_state(self, t_target):
        tgo = self.get_tgo()
        state_local = [max(((t_target - self.Y[0]) - tgo) / tgo, 0.)]  # (期望剩余飞行时间 - 实际剩余飞行时间)/实际剩余飞行时间
        return np.array(state_local)

    def get_reward(self, t_target):
        tgo = self.tgo
        e_local = ((t_target - self.Y[0]) - tgo) / tgo
        reward_local = 0.99 * math.exp(-e_local ** 2) + \
                       0.01 * math.exp((self.Y[4] - self.R) * tgo / 1e4)

        return np.array(reward_local)


if __name__ == '__main__':
    env = OMissile()

    # set the init parameter
    state_dim = 1
    action_dim = 1
    action_bound = 2.0 * 9.81  # action limitation
    t0 = time.time()
    model_num = 0

    dict_reward = {'episode_reward': [], 'target_time': [], 'actual_time': []}
    all_episode_reward = []

    train = False  # choose train or test
    if train:
        agent = PPO(state_dim, action_dim, action_bound)
        for episode in range(int(TRAIN_EPISODES)):
            desired_tgo = []  # 期望的tgo
            actual_tgo = []  # 实际的tgo
            episode_reward = 0

            env.modify()  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
            td = env.get_tgo() * random.uniform(1.2, 1.5)
            state = env.get_state(td)
            action = agent.get_action(state)  # get new action with old state
            # 弹道模型
            done = False
            while done is False:
                # collect state, action and reward
                if int(env.Y[0] * 100) % 10 == 0:  # 0.1s
                    action = agent.get_action(state)  # get new action with old state
                done = env.step(action=float(action))
                if int(env.Y[0] * 100) % 10 == 0:  # 0.1s
                    state_ = env.get_state(td)  # get new state with new action
                    reward = env.get_reward(td)  # get new reward with new action
                    agent.store_transition(state, action, reward)  # train with old state
                    state = state_  # update state
                    episode_reward += reward

                desired_tgo.append(td - env.Y[0])
                actual_tgo.append(env.tgo)

                # update ppo
                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(state_, done, GAMMA=GAMMA)
                    agent.update()
            # end of one episode
            env.plot_data(0)

            # use the terminal data to update once
            if len(agent.reward_buffer) != 0:
                agent.reward_buffer[-1] -= env.R + (td - env.Y[0]) ** 2
                agent.finish_path(state, done, GAMMA=GAMMA)
                agent.update()
            episode_reward -= env.R + (td - env.Y[0]) ** 2

            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Training | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[0], td - env.Y[0]))

            # calculate the discounted episode reward
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * .9 + episode_reward * .1)

            # save the episode data
            dict_reward['episode_reward'].append(episode_reward)
            dict_reward['target_time'].append(td)
            dict_reward['actual_time'].append(env.Y[0])

            # save model and data
            if episode_reward > REWARD_SAVE_CASE:
                REWARD_SAVE_CASE = episode_reward
                # if abs(td - env.Y[0]) < 0.5:
                agent.save_model('./corrector_module/ppo_model/agent{}'.format(model_num))
                savemat('./ppo_reward.mat', dict_reward)
                model_num = (model_num + 1) % 20

        agent.save_model('./corrector_module/ppo_model/agent_end')
        savemat('./ppo_reward.mat', dict_reward)

        plt.figure(1)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['PPO', time.strftime("%Y_%m%d_%H%M")])))
        plt.show()
    else:
        # test
        dict_sim = {"trajectory": [], "am": [], "ab": []}

        agent = PPO(state_dim, action_dim, action_bound, r'./corrector_module/ppo_model')
        for episode in range(TEST_EPISODES):
            desired_tgo = []  # 期望的tgo
            actual_tgo = []  # 实际的tgo
            action = 0.
            episode_reward = 0.

            # env.modify()  # [0., 200., 0, -20000., 20000, 200]
            env.modify()  # [0., 200., -180 / RAD, 5000., 10000, 100]
            td = env.get_tgo() * random.uniform(1.1, 1.2)
            # td = 120
            state = env.get_state(td)
            done = False
            t = []
            while done is False:
                if int(env.Y[0] * 100) % 10 == 0:  # 0.1s
                    # action = max(state[0]* env.tgo / 10, 0.)
                    action = agent.get_action(state, greedy=True)  # use the mean of distribution as action
                done = env.step(action=action)
                if int(env.Y[0] * 100) % 10 == 0:  # 0.1s
                    state = env.get_state(td)
                    reward = env.get_reward(td)
                    episode_reward += reward

                desired_tgo.append(td - env.Y[0])
                actual_tgo.append(env.tgo)

            # env.plot_data(figure_num=0)
            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Testing | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
                  .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[0], td - env.Y[0]))

            # plt.figure(1)
            # plt.ion()
            # plt.clf()
            # plt.plot(np.array(env.reY)[:, 0], np.array(desired_tgo)[:-1], 'k--', label='desired tgo')
            # plt.plot(np.array(env.reY)[:, 0], np.array(actual_tgo)[:-1], 'k-', label='actual tgo')
            # plt.xlabel('Time (s)')
            # plt.ylabel('t_go(s)')
            # plt.legend()
            # plt.grid()
            #
            # plt.pause(0.1)
            dict_sim['trajectory'].append(env.reY)
            dict_sim['am'].append(env.ream)
            dict_sim['ab'].append(env.reab)

            dict_reward['episode_reward'].append(episode_reward)
            dict_reward['target_time'].append(td)
            dict_reward['actual_time'].append(env.Y[0])

        savemat('./ppo_monte_pre5.mat', dict(dict_reward, **dict_sim))
