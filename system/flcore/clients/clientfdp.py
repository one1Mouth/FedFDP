import copy

import torch
import torch.nn as nn
import numpy as np
import time

from flcore.clients.clientbase import Client
from flcore.optimizers.dp_optimizer import DPAdam, DPSGD

from utils.privacy import *
import torch.nn.utils as utils


class clientFDP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lambda_prf = args.lambda_prf
        self.dp_norm = args.dp_norm
        self.batch_sample_ratio = args.batch_sample_ratio
        self.fairness = args.fairness
        self.auto_s = args.auto_s
        self.global_loss = 0.0
        self.loss_batch_avg = 0.0  # 一个batch里的loss的值的均值，需要回传给中心方的
        self.loss_in_test_batch = 0.0  # 梯度下降之后的模型，在测试集上做一次前向，拿到loss
        self.loss_in_train_batch = 0.0  # 梯度下降之后的模型，在训练集上做一次前向，拿到loss
        self.clip_norm_for_loss = args.clip_norm_for_loss
        self.noise_multiplier_for_loss = args.noise_multiplier_for_loss

        self.last_loss = args.clip_norm_for_loss

        if self.auto_s:
            self.dp_norm = 1.0  # 正则化裁剪，需要C=1，因为C在加噪的时候会在方差上有影响

        if self.privacy:
            self.optimizer = DPSGD(
                l2_norm_clip=self.dp_norm,  # 裁剪范数
                noise_multiplier=self.dp_sigma,
                minibatch_size=self.batch_size,  # batch_size
                microbatch_size=1,  # 几个样本梯度进行一次裁剪
                # 后面这些参数是继承父类的（SGD优化器的一些参数）
                params=self.model.parameters(),
                lr=self.learning_rate,
            )

    def train(self):
        print("---------------------------------------")
        print(f"Client {self.id} is training, privacy={self.privacy}, AUTO-S={self.auto_s}, fairness={self.fairness}")
        minibatch_size = int(self.train_samples * self.batch_sample_ratio)
        trainloader = self.load_train_data_minibatch(minibatch_size=minibatch_size, iterations=1)  # 打印检查过了，只有一个batch

        self.model.train()  # 在训练开始之前写上 model.trian() ，在测试时写上 model.eval()
        start_time = time.time()

        max_local_epochs = self.local_epochs  # epoch=1, limit batch=1
        if self.train_slow:  # False
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):  # FedPRF中限定epochs=1
            for i, (x, y) in enumerate(trainloader):  # load_train_data_minibatch，只有一个batch，循环只有一次
                print("本次采样个数len(y): ", len(y))
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if self.privacy:  # 如果是要DP的，那就逐样本梯度下降
                    for j, (x_single, y_single) in enumerate(zip(x, y)):  # 遍历每个样本
                        self.optimizer.zero_microbatch_grad()  # 梯度清空
                        output = self.model(torch.unsqueeze(x_single.to(torch.float32), 0))  # 逐样本的原因，这里x要升维
                        loss = self.loss(output, torch.unsqueeze(y_single.to(torch.long), 0))  # 逐样本的原因，这里y要升维
                        loss.backward()  # 求导得到梯度
                        fair_coef = 1.0  # coefficient 系数
                        if self.fairness:
                            fair_coef = 1 + self.lambda_prf * (loss.item() - self.global_loss)  # 公平性系数

                        self.optimizer.microbatch_step(self.privacy, self.auto_s,
                                                       self.fairness, fair_coef)  # 这里做每个样本的梯度裁剪和梯度累加操作
                    self.optimizer.step_dp()  # 这里做的是梯度加噪和梯度平均更新下降的操作
                else:  # 如果不要DP的
                    output = self.model(x)  # 前向传播
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()  # 梯度缓存清零，以确保每个训练批次的梯度都是从头开始计算的
                    loss.backward()  # 对损失值 `loss` 进行反向传播，计算模型参数的梯度
                    if self.fairness:  # 如果需要公平，就在学习率上加，“动态学习率”方案
                        fair_coef = 1 + self.lambda_prf * (loss.item() - self.global_loss)  # 公平性系数
                        self.optimizer.defaults['lr'] = self.learning_rate * fair_coef  # 不加DP，就在学习率上动手脚加公平
                    self.optimizer.step()  # 梯度下降

        self.loss_in_test_batch = self.get_loss_in_test_set(batch_size=64)

        # 这里trainloader的batchsize不要太小，不然DP_loss的噪声影响很大
        batch_loss_list = self.get_per_sample_loss_list_in_train_set(trainloader=trainloader)
        self.loss_in_train_batch = sum(batch_loss_list) / (len(batch_loss_list) + 1e-2)
        if self.privacy:
            self.loss_batch_avg = dp_process_for_scalar_list(batch_loss_list,
                                                             clipping_norm=self.last_loss,
                                                             noise_multiplier=self.noise_multiplier_for_loss)
            print(
                f"DP处理前的loss_batch_avg={self.loss_in_train_batch}, DP处理后的loss_batch_avg={self.loss_batch_avg}")
            self.last_loss = max(0, self.loss_batch_avg)
        else:
            self.loss_batch_avg = self.loss_in_train_batch

        print("client {} is training, loss_batch_avg = {:.4f}, loss_in_test_batch = {:.4f}"
              .format(self.id, self.loss_batch_avg, self.loss_in_test_batch))

        if self.learning_rate_decay:  # False
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
