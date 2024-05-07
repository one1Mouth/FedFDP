# 计算模型/梯度的二范数
def compute_l2_norm_of_model(model):
    params = model.parameters()
    total_norm_grad = 0.
    total_norm_model = 0.
    for param in params:  # param是具体的那个张量（逐层），偏执单元和正常单元的张量分开
        if param.requires_grad:
            total_norm_grad += param.grad.data.norm(2).item() ** 2.  # 对每层求范数然后把根号开出来，然后对它们求和
            total_norm_model += param.data.norm(2).item() ** 2.  # 对每层求范数然后把根号开出来，然后对它们求和

    total_norm_grad = total_norm_grad ** .5  # 最后求和的数再取平方根，完成了单样本梯度范数的计算
    total_norm_model = total_norm_model ** .5  # 最后求和的数再取平方根，完成了单样本梯度范数的计算

    # 原本的计算方式，以为传了loss.back之后的模型进来就是传了梯度进来，实际上算的是模型的L2Norm
    # l2_norm = 0.0
    # model_dict = {}
    # for key, var in model.state_dict().items():
    #     model_dict[key] = var.clone()
    #
    # for key in model_dict:
    #     l2_norm += model_dict[key].norm(2) ** 2
    #
    # l2_norm = l2_norm ** 0.5
    return total_norm_grad, total_norm_model
