import h5py
import matplotlib.pyplot as plt

# 打开HDF5文件


# file_path = 'mnist_FedPRF_001_1_1_0.h5'  # 替换为你的H5文件路径
from utils.fairness import weighted_variance

file_path = "fmnist_FedFDP_012-1-23_0.h5"

with h5py.File(file_path, 'r') as hf:
    # 读取数据集
    rs_test_acc_data = hf['rs_test_acc'][:]
    rs_train_loss_data = hf['rs_train_loss'][:]
    rs_server_acc_data = hf['rs_server_acc'][:]
    rs_server_loss_data = hf['rs_server_loss'][:]
    rs_clients_acc = hf['rs_clients_acc'][:]
    uploaded_weights = hf['uploaded_weights'][:]
    rs_psi_data = hf['rs_psi'][:]
    extra_msg = hf['extra_msg'][()]

    extra_msg = extra_msg.decode('utf-8')
    print("----------------------------------------------")
    print(extra_msg)
    print("rs_psi:", rs_psi_data)
    print("rs_server_acc:", rs_server_acc_data)
    print("best_server_acc", max(rs_server_acc_data))
    print("rs_server_loss:", rs_server_loss_data)
    print("rs_clients_acc:", rs_clients_acc)
    print("uploaded_weights:", uploaded_weights)
    print("----------------------------------------------")

    print(f"last_psi={rs_psi_data[-1]}, last_acc={rs_server_acc_data[-1]}, best_acc={max(rs_server_acc_data)}")
    weight_var_of_acc = weighted_variance(rs_clients_acc, uploaded_weights)
    print(f"client_acc的加权方差={weight_var_of_acc}")

    # 创建子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 子图1：rs_test_acc_data和rs_server_acc_data
    axs[0].plot(rs_test_acc_data, label='rs_test_acc', alpha=0.5, linestyle='--')
    axs[0].plot(rs_server_acc_data, label='rs_server_acc', alpha=0.5, linestyle='--')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Value')
    axs[0].set_title(file_path)
    axs[0].legend()

    # 子图2：rs_train_loss_data和rs_server_loss_data
    axs[1].plot(rs_train_loss_data, label='rs_train_loss', alpha=0.5, linestyle='--')
    axs[1].plot(rs_server_loss_data, label='rs_server_loss', alpha=0.5, linestyle='--')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Value')
    axs[1].set_title(file_path)
    axs[1].legend()

    # 子图3：rs_tau_list
    axs[2].plot(rs_psi_data, label='rs_psi_data', alpha=0.5, linestyle='--')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Value')
    axs[2].set_title(file_path)
    axs[2].legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()
