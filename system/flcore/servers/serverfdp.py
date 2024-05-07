import os
import time
from random import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import ujson

from flcore.clients.clientfdp import clientFDP
from flcore.optimizers.utils.dp_utils import compute_rdp, compute_eps
from flcore.servers.serverbase import Server
from threading import Thread

from utils.data_utils import read_server_testset
from system.flcore.optimizers.utils.RDP.compute_dp_sgd import apply_dp_sgd_analysis


class FedFDP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFDP)
        self.global_loss = 0.0  # 是各client的loss的加权，不是中心方loss(叫server_loss)
        self.rs_psi = []  # 公平性参数psi的列表
        self.rs_server_acc = []  # 中心方测出来的准确率
        self.rs_server_loss = []  # 中心方测出来的loss，不是各client的loss的加权
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数,用来测server_loss的
        self.batch_sample_ratio = args.batch_sample_ratio
        self.dp_sigma = args.dp_sigma  # 算epsilon的时候要用

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def send_models_and_global_loss(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)
            """
            注意这里下发的是加权本地loss，而非用中心方数据算出来的loss
            中心方的测试数据不能用于训练
            """
            client.global_loss = self.global_loss

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)  # 不太懂，为啥要乘2

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"

        current_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
        parent_directory = os.path.dirname(current_path)  # 找到当前脚本的父目录
        parent_directory = os.path.dirname(parent_directory)  # 找到父目录的父目录
        parent_directory = os.path.dirname(parent_directory)  # system
        root_directory = os.path.dirname(parent_directory)  # 项目根目录的绝对路径
        config_json_path = root_directory + "\\dataset\\" + self.dataset + "\\config.json"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # 计算一下隐私 epsilon (更新后的代码，不用global_rounds来计算epsilon了，而是用epsilon去决定global_rounds)
        # orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        # eps, opt_order = apply_dp_sgd_analysis(q=self.batch_sample_ratio,
        #                                        sigma=self.dp_sigma,
        #                                        # steps: 单个客户端本地迭代总轮数, FedPRF一轮只有一个iteration, 所以填global_rounds
        #                                        steps=self.global_rounds,
        #                                        orders=orders,
        #                                        delta=1e-5)
        # print("eps:", format(eps) + "| order:", format(opt_order))
        #
        # rdp = compute_rdp(q=self.batch_sample_ratio, noise_multiplier=self.dp_sigma, steps=self.global_rounds,
        #                   orders=orders)  # 先算RDP、也就是RDP定义下总的隐私损失alpha，如果根据RDP算的话，可能会依据RDP的文章
        # rdp2 = compute_rdp(q=self.batch_sample_ratio, noise_multiplier=self.args.clip_norm_for_loss,
        #                    steps=self.global_rounds,
        #                    orders=orders)  # 先算RDP、也就是RDP定义下总的隐私损失alpha，如果根据RDP算的话，可能会依据RDP的文章
        # eps, opt_order = compute_eps(orders, rdp + rdp2, delta=1e-5)  # 再根据RDP转换为对应的最佳eps和lamda

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)  # goal的作用在这呢
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            extra_msg = f"dataset = {self.dataset}, learning_rate = {self.learning_rate},\n" \
                        f"rounds = {self.global_rounds}, batch_sample_ratio = {self.batch_sample_ratio},\n" \
                        f"num_clients = {self.num_clients}, algorithm = {self.algorithm} \n" \
                        f"have_PD = {self.args.privacy}, dp_sigma = {self.args.dp_sigma}\n" \
                        f"dp_norm = {self.args.dp_norm}, epsilon = {self.args.dp_epsilon}\n" \
                        f"have_fair = {self.args.fairness}, fair_lambda = {self.args.lambda_prf}\n"
            with open(config_json_path) as f:
                data = ujson.load(f)

            extra_msg = extra_msg + "--------------------config.json------------------------\n" \
                                    "num_clients={}, num_classes={}\n" \
                                    "non_iid={}, balance={},\n" \
                                    "partition={}, alpha={}\n".format(
                data["num_clients"], data["num_classes"], data["non_iid"],
                data["balance"], data["partition"], data["alpha"])

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_server_acc', data=self.rs_server_acc)
                hf.create_dataset('rs_server_loss', data=self.rs_server_loss)
                hf.create_dataset('rs_psi', data=self.rs_psi)
                hf.create_dataset('rs_clients_acc', data=self.rs_clients_acc)
                hf.create_dataset('uploaded_weights', data=self.uploaded_weights)
                hf.create_dataset('extra_msg', data=extra_msg, dtype=h5py.string_dtype(encoding='utf-8'))

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models_and_global_loss()  # 下发模型及global_loss

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model by personalized")
                self.evaluate()

            global_loss_list = []
            for client in self.selected_clients:
                client.train()
                global_loss_list.append(client.loss_batch_avg)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.global_loss = sum([p * q for p, q in zip(self.uploaded_weights, global_loss_list)])

            if self.dlg_eval and i % self.dlg_gap == 0:  # 算 峰值信噪比
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)  # 本轮的时间开销
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print("\nEvaluate global model by global")
                self.evaluate_fairness()
                self.evaluate_server(q=0.2, test_batch_size=64)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("Best local_avg_accuracy={:.4f}, Last local_avg_accuracy={:.4f}".format(
            max(self.rs_test_acc), self.rs_test_acc[-1]))
        print("Best Psi={:.6f}, Last Psi={:.6f}".format(min(self.rs_psi), self.rs_psi[-1]))
        print("Best server_accuracy={:.4f}, Last server_accuracy={:.4f}".format(
            max(self.rs_server_acc), self.rs_server_acc[-1]))
        print("Last server_loss={:.4f}".format(self.rs_server_loss[-1]))
        print("Average time cost per round={:.4f}".format(sum(self.Budget[1:]) / len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPRF)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
