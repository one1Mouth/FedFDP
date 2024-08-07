import copy
import math

import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serverfdp import FedFDP
from flcore.trainmodel.models import FedAvgCNN, Digit5CNN
from utils.mem_utils import MemReporter
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)  # 3407

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model

        # model_str == "cnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "Digit5" in args.dataset:
            args.model = Digit5CNN().to(args.device)
        else:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        print(args.model)

        # select algorithm
        # if args.algorithm == "FedPRF":
        server = FedFDP(args, i)

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    ## general
    # goal
    parser.add_argument('-go', "--goal", type=str, default="013-1-1",
                        help="The goal for this experiment")
    # dp_epsilon
    parser.add_argument('-dpe', "--dp_epsilon", type=float, default=3.52)
    # lambda_prf; 5e1的时候，泊松采样0.05的方式会让loss到正无穷,此时auc的评估会出问题，所以auc相关的计算被我删掉了
    parser.add_argument('-lp', "--lambda_prf", type=float, default=1e-1,
                        help="The ratio of learning rates between the two optimization objectives for FedPRF")
    # global_rounds; dgrdbe=Ture的话, global_rounds会被重新根据epsilon计算
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    # bsr_q: batch sample ratio (Poisson sampling)
    parser.add_argument('-bsr', "--batch_sample_ratio", type=float, default=0.05,
                        help="The ratio of Poisson sampling, 'q' in the paper FedPRF")
    # sigma
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.1)
    # norm
    parser.add_argument('-dpn', "--dp_norm", type=float, default=0.1)
    # 数据集
    parser.add_argument('-data', "--dataset", type=str, default="fmnist")  # mnist, Cifar10, fmnist
    # algorithm
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    # learning_rate
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=1.0,
                        help="Local learning rate")  # DPSGD 学习率在0.1这个级别，再低一个数量级acc升不了，再高一个数量级loss指数爆炸
    # num_clients
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    # local_epochs
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    # local_iterations: DPSGD里没有epoch的概念，都是按一个batch来的iterations
    parser.add_argument('-li', "--local_iterations", type=int, default=1,
                        help="DP-FedSGD need li")
    # fairness
    fair_help = r"you can set {fairness = False} or {lambda_prf = 0} to shut down fair module."
    parser.add_argument('-fair', "--fairness", type=bool, default=False,
                        help="fairness" + fair_help)
    # privacy:
    parser.add_argument('-dp', "--privacy", type=bool, default=True,
                        help="differential privacy")

    # dp_global_rounds_decide_by_epsilon
    parser.add_argument('-dgrdbe', "--dp_gr_decide_by_epsilon", type=bool, default=True,
                        help="global rounds decide by epsilon")
    # dp_delta
    parser.add_argument('-dpd', "--dp_delta", type=float, default=1e-5)
    # AUTO-S
    parser.add_argument('-as', "--auto_s", type=bool, default=False,
                        help="Clipping method: AUTO-S(True) or Abadi(False)")
    # clip_norm_for_loss:
    parser.add_argument('-cnfl', "--clip_norm_for_loss", type=float, default=2.5,
                        help="clip norm for batch loss avg")
    # noise_multiplier_for_loss:
    parser.add_argument('-nmfl', "--noise_multiplier_for_loss", type=float, default=1.0,
                        help="noise multiplier for batch loss avg")
    # batch_size (凡是有DP的，都使用泊松采样，这个batch size只会在一些default的时候有用)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 客户端参加的比例(client drift程度)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")

    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")  # 这里如果设成3，就会自动跑三次 wow!
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")  # 几轮一次test的意思

    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)  # ？
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)  # ？ ↓下面应该都是跟具体算法有关了
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    print("lambda_prf: {}".format(args.lambda_prf))
    print("=" * 50)

    run(args)
