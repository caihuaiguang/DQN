from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim#动作可选维数
        self.__device = device#设备
        self.__gamma = gamma#γ值

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self.__policy = DQN(action_dim, device).to(device)   # policy网络
        self.__target = DQN(action_dim, device).to(device)   # target网络 
        
        if restore is None:
            self.__policy.apply(DQN.init_weights)
        else:
            self.__policy.load_state_dict(torch.load(restore))
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:     # 返回action
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:     # 返回vlue
        """learn trains the value network via TD-learning."""
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)#随机选取一个样本
        # SGD优化的基本要求之一是训练数据是独立且均匀分布的
        # 当Agent与环境交互时，经验元组的序列可以高度相关,所以要打乱采样
        #将样本送入学习
        values = self.__policy(state_batch.float()).gather(1, action_batch) #Q表：value=Q(s,a)
        
        ##########dueling DQN修改思路:拆分Q(s,a)=V(s)+A(s,a)，其中V(s)为状态s本身的价值，A(s,a)为动作a的价值##########
        ##########V(s)是一个标量，A(s,a)是一个向量。在相加时V(s)会自动复制到与A(s,a)维度一致##########
        
        values_next = self.__target(next_batch.float()).max(1).values.detach() #Q'表，但是具体参数不清楚
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch         #当完成了（done=1），y_j=r_j;否则y_j=r_j+q()见论文算法
        loss = F.smooth_l1_loss(values, expected) # 损失函数

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
