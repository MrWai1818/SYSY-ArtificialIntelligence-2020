import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Hyper Parameters
BATCH_SIZE = 128		#batch大小
LR = 0.001                   # 学习率learning rate
EPSILON = 1               # 贪心策略greedy policy
EPSILON_de = 0.999			#epsilon会逐渐减小
EPSILON_min = 0.05			#epsilon有一个最小值
GAMMA = 0.99                 # 折扣率
TARGET_REPLACE_ITER = 100   # 每学习100次更新一次目标策略
MEMORY_CAPACITY = 10000		#经验池大小
episode_size = 700	#测试集次数
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
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = []
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
    #动作选择
    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() < EPSILON:   # greedy
            t = np.random.randint(0, env.action_space.n)
            action = t
        else:   # random
            action_value = self.eval_net(x)
            action = torch.max(action_value, dim=-1)[1].numpy()
        return int(action)
    #存入经验池
    def store_transition(self, *transition):
        if len(self.memory) == MEMORY_CAPACITY :
            self.memory.pop(0)
        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self):
        # 更新目标策略网络
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从经验池中随机选取若干状态作为batch
        sample_index = np.random.choice(len(self.memory), BATCH_SIZE)
        # 将选出来的状态的各个参数进行格式化，以便于输入神经网络
        batch = [self.memory[i] for i in sample_index]
        b_s,b_a,b_r,b_s_ ,dones = zip(*batch)
        b_a = torch.LongTensor(b_a)
        dones = torch.FloatTensor(dones)
        b_r = torch.FloatTensor(b_r)
        b_s = torch.FloatTensor(b_s)
        b_s_ = torch.FloatTensor(b_s_)

        #将每个选出状态输入行为策略网络计算
        q_eval = self.eval_net(b_s).gather(-1, b_a.unsqueeze(-1)).squeeze(-1)  # shape (batch, 1)
        #将每个选出状态输入目标策略网络计算
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        #根据公式算出目标Q值以近似最佳状态
        q_target = b_r + GAMMA * (1 - dones) * torch.max(q_next, dim=-1)[0] 
        #计算损失值
        loss = self.loss_func(q_eval, q_target)
        #可视化相关
        z_label.append(int(loss))
        v2_label.append(np.var(z_label))
        #行为策略网络反向传递
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
time1=time.time()
dqn = DQN()
d = 0
for i_episode in range(episode_size):	#进行episode
    s = env.reset()	#获取环境初始信息
    ep_r = 0		#奖励值
    while True:	
        # env.render()
        a = dqn.choose_action(s)	#选择动作
        # take action
        s_,r, done, info = env.step(a)#得到下一个状态的信息
        dqn.store_transition(s, a, r, s_,done)#加入经验池
        ep_r += r	#奖励增加
        s = s_		#状态切换
        if dqn.memory_counter >= MEMORY_CAPACITY:
            # if EPSILON > EPSILON_min:
            #     EPSILON *= EPSILON_de
            dqn.learn()
            if done:
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
ax2.plot([i for i in range(len(z_label))],z_label,label = 'loss')
ax2.plot([i for i in range(len(z_label))],v2_label,label = 'loss\'s variance')
plt.xlabel('times')
plt.legend()
plt.title('complete rate '+str(d/episode_size)+'   cost time'+str(time2-time1))
plt.show()