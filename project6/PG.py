import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import time
# Hyper Parameters
LR = 0.001                   # 学习率learning rate
log_freq = 20           #载入日志频率
GAMMA = 0.99                 # 折扣率
MEMORY_CAPACITY = 10000		#经验池大小
episode_size = 4000		#测试集次数
env = gym.make('CartPole-v0')	#引入环境
# env = env.unwrapped
x_label = []   #episode
y_label = []    #reward
z_label = []    #损失loss
v1_label = []   #reward方差
v2_label = []   #loss方差
N_ACTIONS = env.action_space.n		#环境输入动作
N_STATES = env.observation_space.shape[0]	#环境状态转为数组
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     #环境与dqn输入矩阵大小限定
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 32)
        self.out = nn.Linear(32, N_ACTIONS)
     

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.out(x), dim=-1)
        return prob


class PG(object):
    def __init__(self):
        self.net = Net()                              # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = []
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
    #动作选择
    def choose_action(self, x):
        prex = self.net(x)
        dist = Categorical(prex)
        action = dist.sample()
        log_prx = dist.log_prob(action)
        return int(action),log_prx
    #存入经验池
    def store_transition(self, *transition):
        if len(self.memory) == MEMORY_CAPACITY :
            self.memory.pop(0)
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self):
        log_probs,rewards = zip(*self.memory)
        log_probs = torch.stack(log_probs)
        T = len(rewards)
        ret = np.empty(T,dtype=np.float32)
        next_ret = 0.0
        for t in reversed(range(T)):
            next_ret = rewards[t] + GAMMA * next_ret
            ret[t] = next_ret
        ret = torch.tensor(ret)
        loss = -ret * log_probs
        #可视化相关
        z_label.append(int(loss.mean()))
        v2_label.append(np.var(z_label))
        #反向传递
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()
time1=time.time()
pg = PG()
d = 0
for i_episode in range(episode_size):	#进行episode
    s = env.reset()	#获取环境初始信息
    ep_r = 0		#奖励值
    while True:	
        # env.render()
        a ,log_p= pg.choose_action(s)	#选择动作
        # take action
        s_,r, done, info = env.step(a)#得到下一个状态的信息
        pg.store_transition(log_p,r)#加入经验池
        ep_r += r	#奖励增加
        s = s_		#状态切换
        if done:
            pg.learn()
            pg.memory.clear()
            x_label.append(i_episode)
            y_label.append(ep_r)
            v1_label.append(np.var(y_label))
            if ep_r ==200 :
                d+=1
            print('Ep: ', i_episode,'| Ep_reward: ', ep_r)

        if done:
            break
time2=time.time()
print('cost time',time2-time1)
print('complete rate ', d/episode_size)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x_label,y_label,label = 'reward')
ax1.plot(x_label,v1_label,c='r',label='reward\'s variance')
plt.xlabel('episode')
plt.legend()

ax2 = fig.add_subplot(122)
ax2.plot([i for i in range(len(z_label))],z_label,label = 'loss\'s mean')
ax2.plot([i for i in range(len(z_label))],v2_label,label = 'loss\'s variance')
plt.xlabel('times')
plt.legend()
plt.title('complete rate '+str(d/episode_size)+'   cost time'+str(time2-time1))
plt.show()