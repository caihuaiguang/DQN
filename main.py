from collections import deque
import os#
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0#全局随机数种子
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1_000_000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
# WARM_STEPS = 50
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000       # 评估频率，每100_000次停下来评估一下 

rand = random.Random()        # [0,1]中的任一浮点数值
rand.seed(GLOBAL_SEED)        # 根据输入seed固定获得相同的随机数
new_seed = lambda: rand.randint(0, 1000_000)#0~1000000中任选其一
os.mkdir(SAVE_PREFIX)         # 在"./models"创建目录

torch.manual_seed(new_seed()) # 将new_seed赋值给cpu的随机数种子
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#创建设备：GPU/CPU?
device = torch.device("cpu")
env = MyEnv(device)
agent = Agent(                # 根据预设参数初始化
    env.get_action_dim(),     # 返回3，三个动作：["NOOP", "RIGHT", "LEFT"]
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)# 循环队列，三者分别对应通道、容量、设备，容量为MEM_SIZE=100_000

#### Training ####
obs_queue: deque = deque(maxlen=5)#创建观察队列
done = True

progressive = tqdm(range(MAX_STEPS), 
                   total=MAX_STEPS,   #   预期的迭代次数
                   ncols=50,          #  可以自定义进度条的总长度
                   leave=False, unit="b")
for step in progressive:#step=int
    if done:#新一轮游戏？
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)#将观察得到内容置入obs_queue
#         print(len(obs_queue))
    training = len(memory) > WARM_STEPS#检测memory长度是否超过WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()#根据观察转换为状态
    action = agent.run(state, training)             # 以epsilon为参数随机选择一个动作
    obs, reward, done = env.step(action)            # 执行动作获得r，和下一个状态
#     print(len(obs)) 执行此语句得到obs的大小为1
    obs_queue.append(obs)#将新观测结果插入obs_queue
    memory.push(env.make_folded_state(obs_queue), action, reward, done)# 保存经验

    if step % POLICY_UPDATE == 0 and training:      # 每 POLICY_UPDATE = 4 次训练，并且要满足memory大小大于WARM_STEPS=50_000
        agent.learn(memory, BATCH_SIZE)             # 从memory中随机采样BATCH_SIZE = 32来帮助更新

    if step % TARGET_UPDATE == 0:                   # 每TARGET_UPDATE= 10_000轮将target网络更新为policy网络的权重
        agent.sync()

    if step % EVALUATE_FREQ == 0:#保存当前状况
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")  # 3d表示3位数的表达形式比如“000”
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
