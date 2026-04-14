"""平滑数学算子模块

本模块提供了用于梯度优化的平滑（可微）数学函数。
这些函数用于替代经济模型中的非可微操作（如 min, max, if-else），
使得整个模型能够通过自动微分进行梯度下降优化。
"""

import torch

def smooth_max(x: torch.Tensor, y: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """平滑最大值函数（基于 LogSumExp）
    
    公式：S_max(x, y; k) = (1/k) * log(exp(k*x) + exp(k*y))
    
    当 k 趋向无穷大时，该函数逼近 max(x, y)。
    较大的 k 值使函数更接近真实的 max，但可能导致数值不稳定。
    
    参数:
        x: 第一个输入张量
        y: 第二个输入张量
        k: 平滑参数（逆温度系数），默认 50.0。值越大越"硬"（接近真实 max）
        
    返回:
        平滑最大值
        
    注意:
        使用数值稳定的 LSE 实现：
        max_val = max(x, y)
        result = max_val + (1/k) * log(exp(k*(x-max_val)) + exp(k*(y-max_val)))
    """
    # 使用数值稳定的 LogSumExp 实现
    # 公式: log(e^A + e^B) = M + log(e^(A-M) + e^(B-M))，其中 M = max(A, B)
    max_val = torch.maximum(x, y)
    return max_val + (1.0 / k) * torch.log(torch.exp(k * (x - max_val)) + torch.exp(k * (y - max_val)))

def smooth_min(x: torch.Tensor, y: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """平滑最小值函数
    
    公式：S_min(x, y; k) = -S_max(-x, -y; k)
    
    通过对最大值函数取负实现最小值的平滑近似。
    
    参数:
        x: 第一个输入张量
        y: 第二个输入张量
        k: 平滑参数，默认 50.0
        
    返回:
        平滑最小值
    """
    return -smooth_max(-x, -y, k)

def smooth_step(x: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """平滑阶跃函数（Sigmoid 近似）
    
    近似 Heaviside 阶跃函数：
    - 当 x > 0 时，返回接近 1 的值
    - 当 x < 0 时，返回接近 0 的值
    - 当 x = 0 时，返回 0.5
    
    公式：sigmoid(k * x) = 1 / (1 + exp(-k * x))
    
    参数:
        x: 输入张量
        k: 平滑参数，默认 50.0。值越大过渡越陡峭
        
    返回:
        平滑阶跃值（0 到 1 之间）
    """
    return torch.sigmoid(k * x)

def smooth_share_lower(val_lower: torch.Tensor, val_higher: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """平滑份额指示函数
    
    返回 (val_lower <= val_higher) 的平滑近似：
    - 当 val_lower < val_higher 时，结果接近 1（表示选择 lower）
    - 当 val_lower > val_higher 时，结果接近 0（表示不选择 lower）
    
    用途示例：
        在 Armington 模型中，当本国价格低于外国价格时，
        本国商品的份额应接近 1（完全替代情况下）。
    
    参数:
        val_lower: 较低值（如本国价格）
        val_higher: 较高值（如外国价格）
        k: 平滑参数，默认 50.0
        
    返回:
        平滑份额（0 到 1 之间）
        
    实现:
        实际上等于 smooth_step(val_higher - val_lower, k)
    """
    return smooth_step(val_higher - val_lower, k)
