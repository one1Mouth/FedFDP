from opacus import PrivacyEngine
import numpy as np

MAX_GRAD_NORM = 1.0
DELTA = 1e-5


def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_sigma,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA


def dp_process_for_scalar_list(scalar_list, clipping_norm=1.0, noise_multiplier=1.0):
    """
    标量DP化的函数:逐个裁剪，求和后统一加噪，然后取平均
    """
    dp_scalar = 0.0
    for i, scalar in enumerate(scalar_list):
        scalar_list[i] = min(scalar, clipping_norm)  # 逐个裁剪
    # 生成均值为0的标量噪声，list不应该太短，不然噪声影响比较大
    noise = np.random.normal(0, clipping_norm * noise_multiplier)
    dp_scalar = (sum(scalar_list) + noise) / (len(scalar_list) + 1e-10)
    return dp_scalar





