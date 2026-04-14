r"""核心模型与静态均衡（trade_game 独立实验运行时）。

层次说明（分层指引）：
- `trade_game/runtime` 是当前 grad/LLM 主实验的独立运行时副本。
- `eco_model_v2` 仍是生产/理论版本的权威校验基线。
- `eco_simu` 仍是旧运行时和 API 服务层。

目的：保持最小可用的两国多部门静态一般均衡骨架，并清晰标注经济学含义与公式对应关系（中文注释）。

包含：
- 数据与参数：CountryParams, ModelParams, create_symmetric_parameters, normalize_model_params
- Armington/CES 工具函数：safe_log, armington_share, armington_price, armington_quantity
- 静态均衡：EquilibriumLayout, _country_block, _equilibrium_residuals, solve_initial_equilibrium

经济学要点（与 production_network_simulation0916.tex 对应）：
- 生产函数：部门 i 的产出
    Y_i = A_i \\prod_j \\Big[ X_{ij} \\Big]^{\\alpha_{ij}}，其中当 j 为可贸易部门时，使用嵌套 CES 组合
    X_{ij}^{CES} = \\left[ (\\gamma_{ij} X_{ij}^I)^{\\rho_{ij}} + ((1-\\gamma_{ij}) X_{ij}^O)^{\\rho_{ij}} \\right]^{\\alpha_{ij}/\\rho_{ij}}。
- 成本（对偶）与零利润条件：边际成本（单位成本）λ_i 满足
    ln λ_i = - ln A_i + \sum_j \alpha_{ij} ln P_j^*，其中 P_j^* 对可贸易品使用 Armington 对偶价格（见 armington_price）。
- 消费者效用：对不可贸易品为 Cobb–Douglas，对可贸易品为嵌套 CES；由此得到消费需求与预算约束。
- 市场清算与国际收支：
    · 国内品供给 Y_j 用于中间品与终端消费及出口；
    · 进口支付以出口收入（含基础出口向量）约束（见 _country_block 中贸易平衡残差）。
"""
# Copied from the former analysis.model runtime to isolate current experiments.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares

try:
    from .smooth_ops import smooth_min, smooth_share_lower
except ImportError:
    # 独立脚本执行时的降级方案（相对导入可能失败）
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from smooth_ops import smooth_min, smooth_share_lower

SMOOTH_K = 20.0  # 平滑近似的参数（控制平滑程度）

EPS = 1e-9
# 恢复 float64（当前 Python 循环结构不适合 GPU 加速）
TORCH_DTYPE = torch.float64

# GPU 加速配置
# 当前状态：禁用 GPU（Python 循环中频繁的 float(tensor[i,j]) 导致 GPU 同步开销过大）
# 如需启用 GPU，需要先重构 compute_output/compute_marginal_cost 为向量化实现
INIT_DEVICE = torch.device("cpu")
COMPUTE_DEVICE = torch.device("cpu")  # 暂时禁用 GPU

# 保留 GPU 检测代码供未来使用
# if torch.cuda.is_available():
#     COMPUTE_DEVICE = torch.device("cuda")
#     print(f"[GPU] CUDA 可用，计算设备: {torch.cuda.get_device_name(0)}")

DEFAULT_DEVICE = INIT_DEVICE

torch.set_default_dtype(TORCH_DTYPE)


# ---------------------
# 数据类与参数工具
# ---------------------


def _infer_device(*tensors) -> torch.device:
    """从输入张量推断设备。
    
    遍历所有输入，返回第一个 torch.Tensor 的设备。
    如果没有张量输入，返回 DEFAULT_DEVICE。
    """
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return DEFAULT_DEVICE


@dataclass(frozen=True)
class CountryParams:
    """国家参数块（定标后均为张量）。

    字段含义：
    - alpha[i,j]: 生产函数中部门 i 对部门 j 中间投入的产出弹性（Leontief-Cobb 混合时，非零值为 CES 权重）。
    - gamma[i,j]: 可贸易部门 j 的 Armington 国内权重 γ_{ij}（0..1）；不可贸易 j 处置为 1.0。
    - rho[i,j]: 可贸易部门的 Armington 形状参数 ρ_{ij}（与替代弹性 σ=1/(1-ρ) 相关）。
    - beta[j]: 消费层面的品类 j 的预算份额（对可贸易 j 进入嵌套 CES）。
    - A[i]: 部门 i 的全要素生产率（TFP）。
    - exports[j]: 作为外生"基础出口"的品类 j 数量（便于冲击与配额建模）。
    - gamma_cons[j], rho_cons[j]: 消费层面可贸易 j 的 Armington 参数。
    - import_cost[j]: 进口 j 的"到岸价"相对系数（含关税/运输/其它附加）。
    """
    alpha: torch.Tensor
    gamma: torch.Tensor
    rho: torch.Tensor
    beta: torch.Tensor
    A: torch.Tensor
    exports: torch.Tensor
    gamma_cons: torch.Tensor
    rho_cons: torch.Tensor
    import_cost: torch.Tensor
    
    def to(self, device: torch.device) -> "CountryParams":
        """将所有张量迁移到指定设备。"""
        return CountryParams(
            alpha=self.alpha.to(device),
            gamma=self.gamma.to(device),
            rho=self.rho.to(device),
            beta=self.beta.to(device),
            A=self.A.to(device),
            exports=self.exports.to(device),
            gamma_cons=self.gamma_cons.to(device),
            rho_cons=self.rho_cons.to(device),
            import_cost=self.import_cost.to(device),
        )
    
    @property
    def device(self) -> torch.device:
        """返回参数所在设备。"""
        return self.alpha.device


@dataclass(frozen=True)
class ModelParams:
    """模型顶层参数。

    - home/foreign: 两国的参数块
    - tradable_idx: 可贸易部门索引（对应上文 j ∈ T）
    - non_tradable_idx: 非贸易部门索引（j ∈ N）
    """
    home: CountryParams
    foreign: CountryParams
    tradable_idx: np.ndarray
    non_tradable_idx: np.ndarray
    
    def to(self, device: torch.device) -> "ModelParams":
        """将两国参数迁移到指定设备。"""
        return ModelParams(
            home=self.home.to(device),
            foreign=self.foreign.to(device),
            tradable_idx=self.tradable_idx,
            non_tradable_idx=self.non_tradable_idx,
        )
    
    @property
    def device(self) -> torch.device:
        """返回参数所在设备。"""
        return self.home.device


def create_symmetric_parameters() -> Dict[str, dict]:
    """构造对称的两国 6 部门基线参数（简化校验用）。

    设可贸易部门集合为 {2,3,4,5}，γ=0.5，ρ=0.2；不可贸易部门 γ 置 1，表示仅用本国供给。
    返回的字典与 normalize_model_params 兼容，便于直接求解均衡。
    """
    n = 6
    alpha_base = 0.15
    alpha_H = np.full((n, n), alpha_base)
    alpha_F = np.full((n, n), alpha_base)
    np.fill_diagonal(alpha_H, 0.0)
    np.fill_diagonal(alpha_F, 0.0)

    gamma_H = np.full((n, n), 0.5)
    gamma_F = np.full((n, n), 0.5)
    tradable = [2, 3, 4, 5]
    for i in range(n):
        for j in range(n):
            if j not in tradable or i == j:
                gamma_H[i, j] = 1.0
                gamma_F[i, j] = 1.0

    rho_H = np.full((n, n), 0.2)
    rho_F = np.full((n, n), 0.2)
    beta_H = np.ones(n) / n
    beta_F = np.ones(n) / n
    A_H = np.ones(n)
    A_F = np.ones(n)
    Export_H = np.zeros(n)
    Export_F = np.zeros(n)
    gamma_c_H = gamma_H[0].copy()
    gamma_c_F = gamma_F[0].copy()
    rho_c_H = rho_H[0].copy()
    rho_c_F = rho_F[0].copy()
    import_cost = np.ones(n)

    return {
        "H": {
            "alpha_ij": alpha_H,
            "gamma_ij": gamma_H,
            "rho_ij": rho_H,
            "beta_j": beta_H,
            "A_i": A_H,
            "Export_i": Export_H,
            "gamma_cj": gamma_c_H,
            "rho_cj": rho_c_H,
            "import_cost": import_cost,
        },
        "F": {
            "alpha_ij": alpha_F,
            "gamma_ij": gamma_F,
            "rho_ij": rho_F,
            "beta_j": beta_F,
            "A_i": A_F,
            "Export_i": Export_F,
            "gamma_cj": gamma_c_F,
            "rho_cj": rho_c_F,
            "import_cost": import_cost,
        },
        "tradable_sectors": tradable,
    }


def _to_country_params(block: Dict[str, np.ndarray]) -> CountryParams:
    """将字典转换为 CountryParams。
    
    注意：使用 INIT_DEVICE (CPU) 以确保与 scipy.optimize 兼容。
    仿真器初始化后会将参数迁移到 COMPUTE_DEVICE（GPU）。
    """
    alpha_np = np.asarray(block["alpha_ij"], dtype=float)
    gamma_np = np.asarray(block["gamma_ij"], dtype=float)
    rho_np = np.asarray(block["rho_ij"], dtype=float)
    beta_np = np.asarray(block["beta_j"], dtype=float)
    A_np = np.asarray(block["A_i"], dtype=float)
    exports_np = np.asarray(block.get("Export_i", np.zeros_like(beta_np)), dtype=float)
    gamma_cons_np = np.asarray(block.get("gamma_cj", gamma_np[0]), dtype=float)
    rho_cons_np = np.asarray(block.get("rho_cj", rho_np[0]), dtype=float)
    import_cost_np = np.asarray(block.get("import_cost", np.ones_like(beta_np)), dtype=float)

    # 使用 INIT_DEVICE (CPU) 以便 scipy 优化
    alpha = torch.as_tensor(alpha_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    gamma = torch.as_tensor(gamma_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    rho = torch.as_tensor(rho_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    beta = torch.as_tensor(beta_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    A = torch.as_tensor(A_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    exports = torch.as_tensor(exports_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    gamma_cons = torch.as_tensor(gamma_cons_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    rho_cons = torch.as_tensor(rho_cons_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    import_cost = torch.as_tensor(import_cost_np, dtype=TORCH_DTYPE, device=INIT_DEVICE)
    return CountryParams(alpha, gamma, rho, beta, A, exports, gamma_cons, rho_cons, import_cost)



def normalize_model_params(raw_params) -> ModelParams:
    if isinstance(raw_params, ModelParams):
        return raw_params
    home = _to_country_params(raw_params["H"])
    foreign = _to_country_params(raw_params["F"])
    tradable = np.array(sorted(raw_params["tradable_sectors"]), dtype=int)
    n = home.alpha.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[tradable] = True
    non_tradable = np.array([i for i in range(n) if not mask[i]], dtype=int)
    return ModelParams(home=home, foreign=foreign, tradable_idx=tradable, non_tradable_idx=non_tradable)


# ---------------------
# Armington / CES 与通用数学
# ---------------------


def safe_log(x: torch.Tensor) -> torch.Tensor:
    """数值安全的对数：log(max(x, EPS))，避免 0 或负数导致的 NaN。"""
    return torch.log(torch.clamp(x, min=EPS))


def armington_share(gamma, p_dom, p_for, rho):
    """Armington 份额 θ(p)：在给定价格下的“本国产品份额”。

    记替代弹性 σ = 1/(1-ρ)，对偶需求权重 w_d = g^σ · p_d^{1-σ}，w_f = (1-g)^σ · p_f^{1-σ}，
    则 θ = w_d / (w_d + w_f)。边界情形（ρ→1 即 σ→∞）回落到择优最低价；ρ→0 为 Cobb–Douglas。
    """
    # 从输入张量推断设备
    device = _infer_device(p_dom, p_for, gamma, rho)
    g = torch.as_tensor(gamma, dtype=TORCH_DTYPE, device=device)
    p_d = torch.clamp(torch.as_tensor(p_dom, dtype=TORCH_DTYPE, device=device), min=EPS)
    p_f = torch.clamp(torch.as_tensor(p_for, dtype=TORCH_DTYPE, device=device), min=EPS)
    r = torch.as_tensor(rho, dtype=TORCH_DTYPE, device=device)
    sigma = torch.where(torch.abs(1.0 - r) < 1e-10, torch.tensor(1.0, dtype=TORCH_DTYPE, device=device), 1.0 / (1.0 - r))

    # Phase 3: 修复平滑函数触发条件
    # 当 ρ 接近 1 时（即 σ 趋于无穷），使用平滑近似
    if float(torch.abs(1.0 - r)) < 0.02:  # ρ > 0.98 时使用平滑
        # 完全替代极限情况使用平滑近似
        # 如果 p_d < p_f，份额应趋向 1.0（全选本国）
        # 如果 p_d > p_f，份额应趋向 0.0（全选外国）
        # smooth_share_lower(x, y) 当 x < y 时返回 ~1
        return torch.clamp(smooth_share_lower(p_d, p_f, k=SMOOTH_K), 1e-6, 1.0 - 1e-6)

    if float(torch.abs(sigma - 1.0)) < 1e-8:
        return torch.clamp(g, 1e-6, 1 - 1e-6)

    w_d = (g ** sigma) * (p_d ** (1.0 - sigma))
    w_f = ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    share = w_d / torch.clamp(w_d + w_f, min=EPS)
    return torch.clamp(share, 1e-6, 1 - 1e-6)


def armington_price(gamma, p_dom, p_for, rho):
    """Armington 对偶价格 P^*(p)：嵌套 CES 的单位成本函数。

    P^* = [ g^σ p_d^{1-σ} + (1-g)^σ p_f^{1-σ} ]^{1/(1-σ)}；σ=1/(1-ρ)。
    边界：σ→∞ 取 min(p_d, p_f)；σ→1 退化为几何平均。
    """
    # 从输入张量推断设备
    device = _infer_device(p_dom, p_for, gamma, rho)
    g = torch.as_tensor(gamma, dtype=TORCH_DTYPE, device=device)
    p_d = torch.clamp(torch.as_tensor(p_dom, dtype=TORCH_DTYPE, device=device), min=EPS)
    p_f = torch.clamp(torch.as_tensor(p_for, dtype=TORCH_DTYPE, device=device), min=EPS)
    r = torch.as_tensor(rho, dtype=TORCH_DTYPE, device=device)
    sigma = torch.where(torch.abs(1.0 - r) < 1e-10, torch.tensor(1.0, dtype=TORCH_DTYPE, device=device), 1.0 / (1.0 - r))

    # Phase 3: 修复平滑函数触发条件
    if float(torch.abs(1.0 - r)) < 0.02:  # ρ > 0.98 时使用平滑
        return smooth_min(p_d, p_f, k=SMOOTH_K)

    if float(torch.abs(sigma - 1.0)) < 1e-8:
        return torch.exp(g * torch.log(p_d) + (1.0 - g) * torch.log(p_f))

    inner = (g ** sigma) * (p_d ** (1.0 - sigma)) + ((1.0 - g) ** sigma) * (p_f ** (1.0 - sigma))
    return torch.clamp(inner, min=EPS) ** (1.0 / (1.0 - sigma))


def armington_quantity(gamma, x_dom, x_for, alpha, rho):
    """Armington 物量合成：作为生产函数的"有效投入"。

    X^{CES} = [ (g x_d)^ρ + ((1-g) x_f)^ρ ]^{α/ρ}；α≤0 时回到 1（不使用 j）。
    """
    # 从输入张量推断设备
    device = _infer_device(x_dom, x_for, gamma, rho)
    if alpha <= 0.0:
        return torch.tensor(1.0, dtype=TORCH_DTYPE, device=device)
    g = torch.as_tensor(gamma, dtype=TORCH_DTYPE, device=device)
    x_d = torch.clamp(torch.as_tensor(x_dom, dtype=TORCH_DTYPE, device=device), min=EPS)
    x_f = torch.clamp(torch.as_tensor(x_for, dtype=TORCH_DTYPE, device=device), min=EPS)
    r = float(torch.as_tensor(rho, dtype=TORCH_DTYPE, device=device))
    if abs(r) < 1e-10:
        return torch.exp(alpha * (g * torch.log(x_d) + (1 - g) * torch.log(x_f)))
    comp = g * (x_d ** r) + (1 - g) * (x_f ** r)
    return torch.clamp(comp, min=EPS) ** (alpha / r)


def compute_output(params: CountryParams, X_dom: torch.Tensor, X_imp: torch.Tensor, tradable_mask: np.ndarray) -> torch.Tensor:
    """生产函数：逐部门计算 Y_i。

    - 对不可贸易 j：直接使用 X_dom^{α_{ij}}
    - 对可贸易 j：使用 Armington 物量合成（见 armington_quantity）
    - 总产出：Y_i = A_i · Π_j component^{α_{ij}}
    支持单样本（2D）与批量（3D）计算。
    """
    n = params.alpha.shape[0]
    dims = X_dom.dim()
    if dims == 3:
        batch = X_dom.shape[0]
        Y = torch.zeros((batch, n), dtype=TORCH_DTYPE, device=X_dom.device)
        for i in range(n):
            prod = torch.clamp(params.A[i], min=EPS).expand(batch)
            for j in range(n):
                a = float(params.alpha[i, j])
                if a <= 0.0:
                    continue
                if tradable_mask[j]:
                    qty = armington_quantity(params.gamma[i, j], X_dom[:, i, j], X_imp[:, i, j], a, params.rho[i, j])
                    prod = prod * torch.clamp(qty, min=EPS)
                else:
                    comp = torch.clamp(X_dom[:, i, j], min=EPS)
                    prod = prod * (comp ** a)
            Y[:, i] = torch.clamp(prod, min=EPS)
        return Y
    elif dims == 2:
        Y = torch.zeros(n, dtype=TORCH_DTYPE, device=X_dom.device)
        for i in range(n):
            prod = torch.clamp(params.A[i], min=EPS)
            for j in range(n):
                a = float(params.alpha[i, j])
                if a <= 0.0:
                    continue
                if tradable_mask[j]:
                    qty = armington_quantity(params.gamma[i, j], X_dom[i, j], X_imp[i, j], a, params.rho[i, j])
                    prod = prod * torch.clamp(qty, min=EPS)
                else:
                    comp = torch.clamp(X_dom[i, j], min=EPS)
                    prod = prod * (comp ** a)
            Y[i] = torch.clamp(prod, min=EPS)
        return Y
    else:
        raise ValueError("X_dom 必须是 2D 或 3D 张量")


def compute_marginal_cost(params: CountryParams, prices: torch.Tensor, import_prices: torch.Tensor,
                          tradable_mask: np.ndarray) -> torch.Tensor:
    """单位成本（边际成本）λ_i：零利润条件下 P_i = λ_i。

    log λ_i = -log A_i + Σ_j α_{ij} log P_j^*，其中
    P_j^* = P_j（不可贸易）或 Armington 对偶价格（可贸易，见 armington_price）。
    批/单样本均支持。
    """
    n = params.alpha.shape[0]
    dims = prices.dim()
    if dims == 2:
        batch = prices.shape[0]
        lambdas = torch.zeros((batch, n), dtype=TORCH_DTYPE, device=prices.device)
        for i in range(n):
            log_cost = -safe_log(params.A[i]).expand(batch)
            for j in range(n):
                a = float(params.alpha[i, j])
                if a <= 0.0:
                    continue
                if tradable_mask[j]:
                    p_idx = armington_price(params.gamma[i, j], prices[:, j], import_prices[:, j], params.rho[i, j])
                    log_cost = log_cost + a * safe_log(p_idx)
                else:
                    log_cost = log_cost + a * safe_log(prices[:, j])
            lambdas[:, i] = torch.exp(log_cost)
        return lambdas
    elif dims == 1:
        lambdas = torch.zeros(n, dtype=TORCH_DTYPE, device=prices.device)
        for i in range(n):
            log_cost = -safe_log(params.A[i])
            for j in range(n):
                a = float(params.alpha[i, j])
                if a <= 0.0:
                    continue
                if tradable_mask[j]:
                    p_idx = armington_price(params.gamma[i, j], prices[j], import_prices[j], params.rho[i, j])
                    log_cost = log_cost + a * safe_log(p_idx)
                else:
                    log_cost = log_cost + a * safe_log(prices[j])
            lambdas[i] = torch.exp(log_cost)
        return lambdas
    else:
        raise ValueError("prices 必须是 1D 或 2D 张量")


def value_added_share(params: CountryParams) -> torch.Tensor:
    """增加值份额 v_i = 1 - Σ_j α_{ij}，用于将部门收入近似映射为“要素收入”。"""
    row_sum = params.alpha.sum(dim=1)
    return torch.clamp(1.0 - row_sum, min=1e-6)


def compute_income(params: CountryParams, prices: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """国民收入近似：I = Σ_i P_i Y_i · v_i。

    注：tex 文档中以要素市场（劳动）计价；此处用增加值份额近似聚合为要素收入，
    与多要素扩展一致（若显式建模要素，则可替换为要素价格·需求的求和）。
    """
    va = value_added_share(params)
    if prices.dim() == 2:
        return torch.sum(prices * outputs * va.unsqueeze(0), dim=1)
    return torch.sum(prices * outputs * va)


# ---------------------
# 静态均衡（方程与求解）
# ---------------------


StateDict = Dict[str, Dict[str, torch.Tensor]]


class EquilibriumLayout:
    """变量打包布局器：确定均衡向量的拼装/拆解与“可贸易”索引映射。

    - tradable_idx：可贸易集合的列索引；X_imp/C_imp 以“压缩形”存储，仅包含这些列/坐标。
    - pack/unpack：将字典状态与向量相互转换（优化器使用 log-变量）。
    - expand_*：将压缩矩阵/向量展开回全维度便于计算。
    """
    def __init__(self, n_sectors: int, tradable_idx: np.ndarray):
        self.n = int(n_sectors)
        self.tradable_idx = np.array(sorted(tradable_idx), dtype=int)
        tmask = np.zeros(self.n, dtype=bool)
        tmask[self.tradable_idx] = True
        self.tradable_mask = tmask
        self.non_tradable_idx = np.array([i for i in range(self.n) if not tmask[i]], dtype=int)
        self.n_tradable = len(self.tradable_idx)
        self._tradable_pos = {int(j): k for k, j in enumerate(self.tradable_idx)}

        self._fields = []
        for c in ("H", "F"):
            self._fields.extend([
                (c, "X_dom", (self.n, self.n)),
                (c, "X_imp", (self.n, self.n_tradable)),
                (c, "C_dom", (self.n,)),
                (c, "C_imp", (self.n_tradable,)),
                (c, "price", (self.n,)),
            ])

        self.total_size = int(sum(int(np.prod(s)) for _, _, s in self._fields))
        self._slices: Dict[Tuple[str, str], Tuple[int, int, Tuple[int, ...]]] = {}
        cur = 0
        for c, f, s in self._fields:
            size = int(np.prod(s))
            self._slices[(c, f)] = (cur, cur + size, s)
            cur += size

    def new_state(self, fill: float = EPS) -> StateDict:
        st: StateDict = {}
        for c in ("H", "F"):
            st[c] = {
                "X_dom": torch.full((self.n, self.n), fill, dtype=TORCH_DTYPE, device=INIT_DEVICE),
                "X_imp": torch.full((self.n, self.n_tradable), fill, dtype=TORCH_DTYPE, device=INIT_DEVICE),
                "C_dom": torch.full((self.n,), fill, dtype=TORCH_DTYPE, device=INIT_DEVICE),
                "C_imp": torch.full((self.n_tradable,), fill, dtype=TORCH_DTYPE, device=INIT_DEVICE),
                "price": torch.ones((self.n,), dtype=TORCH_DTYPE, device=INIT_DEVICE),
            }
        return st

    def pack(self, state: StateDict) -> torch.Tensor:
        vec = torch.empty(self.total_size, dtype=TORCH_DTYPE, device=INIT_DEVICE)
        for (c, f), (s, e, _) in self._slices.items():
            vec[s:e] = state[c][f].reshape(-1)
        return vec

    def unpack(self, vector: torch.Tensor) -> StateDict:
        st: StateDict = {"H": {}, "F": {}}
        for (c, f), (s, e, shp) in self._slices.items():
            st[c][f] = vector[s:e].reshape(shp)
        return st

    def expand_matrix(self, condensed: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((self.n, self.n), dtype=TORCH_DTYPE, device=condensed.device)
        full[:, self.tradable_idx] = condensed
        return full

    def expand_vector(self, condensed: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((self.n,), dtype=TORCH_DTYPE, device=condensed.device)
        full[self.tradable_idx] = condensed
        return full

    def tradable_position(self, sector: int) -> int:
        return self._tradable_pos[int(sector)]

    def is_tradable(self, sector: int) -> bool:
        return bool(self.tradable_mask[int(sector)])


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """相对误差缩放：用于构造稳健的残差（避免量纲/尺度影响）。"""
    scale = torch.maximum(torch.maximum(torch.abs(expected), torch.abs(actual)), torch.tensor(1.0, dtype=TORCH_DTYPE, device=actual.device))
    return (actual - expected) / scale


def _weighted_relative_error(actual: torch.Tensor, expected: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """带权重的相对误差，用于增强重要约束的优先级。"""
    base_error = _relative_error(actual, expected)
    return base_error * weight


def _country_block(state_c: Dict[str, torch.Tensor], state_o: Dict[str, torch.Tensor],
                   params_c: CountryParams, layout: EquilibriumLayout) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """生成某国均衡残差（零利润、要素需求、消费需求与外贸收支）。

    组成：
    - 零利润：P_i 与单位成本 λ_i 的相对误差（∀ i）
    - 中间品需求：对不可贸易 j，X_{ij} 与 α_{ij} P_i Y_i / P_j；
                  对可贸易 j，分别对国内/进口部分使用 Armington 份额 θ 与 (1-θ)
    - 终端消费：对不可贸易 j，C_j 与 β_j I / P_j；可贸易同理加入 θ_c
    - 对外收支：Σ_j P_j·(对方对我进口 + 我基础出口) = Σ_j P_j^O·(我对外进口)
    返回：
    - 残差列表
    - 本国产出向量
    - 本国收入标量
    """
    prices = state_c["price"]
    import_prices = params_c.import_cost * state_o["price"]

    X_dom = state_c["X_dom"]
    X_imp = layout.expand_matrix(state_c["X_imp"])
    C_dom = state_c["C_dom"]
    C_imp = layout.expand_vector(state_c["C_imp"])

    outputs = compute_output(params_c, X_dom, X_imp, layout.tradable_mask)
    income = compute_income(params_c, prices, outputs)

    res: list = []

    lambdas = compute_marginal_cost(params_c, prices, import_prices, layout.tradable_mask)
    for i in range(layout.n):
        res.append(_relative_error(prices[i], lambdas[i]))

    for i in range(layout.n):
        Pi, Yi = prices[i], outputs[i]
        for j in range(layout.n):
            a = float(params_c.alpha[i, j])
            if a <= 0.0:
                res.append(_relative_error(X_dom[i, j], torch.tensor(EPS, dtype=TORCH_DTYPE, device=Pi.device)))
                if layout.is_tradable(j):
                    res.append(_relative_error(X_imp[i, j], torch.tensor(EPS, dtype=TORCH_DTYPE, device=Pi.device)))
                continue
            if layout.is_tradable(j):
                theta = armington_share(params_c.gamma[i, j], prices[j], import_prices[j], params_c.rho[i, j])
                exp_dom = a * theta * Pi * Yi / torch.clamp(prices[j], min=EPS)
                exp_imp = a * (1 - theta) * Pi * Yi / torch.clamp(import_prices[j], min=EPS)
                res.append(_relative_error(X_dom[i, j], exp_dom))
                res.append(_relative_error(X_imp[i, j], exp_imp))
            else:
                exp_dom = a * Pi * Yi / torch.clamp(prices[j], min=EPS)
                res.append(_relative_error(X_dom[i, j], exp_dom))

    for j in range(layout.n):
        b = float(params_c.beta[j])
        if b <= 0.0:
            res.append(_relative_error(C_dom[j], torch.tensor(EPS, dtype=TORCH_DTYPE, device=prices.device)))
            if layout.is_tradable(j):
                res.append(_relative_error(C_imp[j], torch.tensor(EPS, dtype=TORCH_DTYPE, device=prices.device)))
            continue
        if layout.is_tradable(j):
            theta_c = armington_share(params_c.gamma_cons[j], prices[j], import_prices[j], params_c.rho_cons[j])
            exp_dom = b * theta_c * income / torch.clamp(prices[j], min=EPS)
            exp_imp = b * (1 - theta_c) * income / torch.clamp(import_prices[j], min=EPS)
            res.append(_relative_error(C_dom[j], exp_dom))
            res.append(_relative_error(C_imp[j], exp_imp))
        else:
            exp_dom = b * income / torch.clamp(prices[j], min=EPS)
            res.append(_relative_error(C_dom[j], exp_dom))

    partner_X_imp = layout.expand_matrix(state_o["X_imp"])
    partner_C_imp = layout.expand_vector(state_o["C_imp"])
    exports_value = torch.dot(prices, partner_X_imp.sum(dim=0) + partner_C_imp) + torch.dot(prices, params_c.exports)
    imports_value = torch.dot(import_prices, X_imp.sum(dim=0) + C_imp)
    res.append(_relative_error(exports_value, imports_value))

    # ==================== 商品市场出清约束 ====================
    # Y_j = Σ_i X_{ij} + C_j + Export_j
    # 对方对我的进口 = 我的出口
    # 使用更高权重（10倍）以优先满足供需平衡。
    # 这里显式约束全部部门；当前收入和贸易平衡是近似实现，不能依赖 Walras 定律自动推出最后一个部门。
    MARKET_CLEARING_WEIGHT = 10.0
    exports = partner_X_imp.sum(dim=0) + partner_C_imp + params_c.exports
    for j in range(layout.n):
        total_demand_j = X_dom[:, j].sum() + C_dom[j] + exports[j]
        res.append(_weighted_relative_error(outputs[j], total_demand_j, MARKET_CLEARING_WEIGHT))
    # ==================== 商品市场出清约束 END ================

    return res, outputs, income


def _equilibrium_residuals(log_vec: torch.Tensor, layout: EquilibriumLayout, params: ModelParams) -> torch.Tensor:
    """组装两国的总残差，并钉住名义锚（两国某部门价格=1）。"""
    pos = torch.exp(log_vec)
    state = layout.unpack(pos)
    res_H, _, _ = _country_block(state["H"], state["F"], params.home, layout)
    res_F, _, _ = _country_block(state["F"], state["H"], params.foreign, layout)
    residuals = res_H + res_F
    residuals.append(_relative_error(state["H"]["price"][0], torch.tensor(1.0, dtype=TORCH_DTYPE, device=pos.device)))
    residuals.append(_relative_error(state["F"]["price"][0], torch.tensor(1.0, dtype=TORCH_DTYPE, device=pos.device)))
    return torch.stack([torch.as_tensor(r, dtype=TORCH_DTYPE, device=pos.device) for r in residuals])


def _initial_guess(params: ModelParams, layout: EquilibriumLayout) -> StateDict:
    """启发式初值：
    - 初始价格 P=1，进口价按 import_cost 缩放
    - 中间品/消费以 Cobb–Douglas 需求式近似（可贸易插入 θ/θ_c）
    - 保障所有变量正值（>=EPS）以便对数参数化
    """
    st = layout.new_state(fill=EPS)
    for label, c_params in (("H", params.home), ("F", params.foreign)):
        c = st[label]
        prices = c["price"]
        prices[:] = 1.0
        import_prices = c_params.import_cost * torch.ones(layout.n, dtype=TORCH_DTYPE, device=INIT_DEVICE)
        base_output = torch.maximum(c_params.A, torch.tensor(1.0, dtype=TORCH_DTYPE, device=INIT_DEVICE))
        for i in range(layout.n):
            Pi, Yi = prices[i], base_output[i]
            for j in range(layout.n):
                a = float(c_params.alpha[i, j])
                if a <= 0.0:
                    continue
                if layout.is_tradable(j):
                    theta = torch.clamp(c_params.gamma[i, j], 1e-3, 1 - 1e-3)
                    idx = layout.tradable_position(j)
                    c["X_dom"][i, j] = torch.clamp(a * theta * Pi * Yi / torch.clamp(prices[j], min=EPS), min=EPS)
                    c["X_imp"][i, idx] = torch.clamp(a * (1 - theta) * Pi * Yi / torch.clamp(import_prices[j], min=EPS), min=EPS)
                else:
                    c["X_dom"][i, j] = torch.clamp(a * Pi * Yi / torch.clamp(prices[j], min=EPS), min=EPS)

        income = torch.sum(prices * base_output * value_added_share(c_params))
        for j in range(layout.n):
            b = float(c_params.beta[j])
            if b <= 0.0:
                continue
            if layout.is_tradable(j):
                theta_c = torch.clamp(c_params.gamma_cons[j], 1e-3, 1 - 1e-3)
                idx = layout.tradable_position(j)
                c["C_dom"][j] = torch.clamp(b * theta_c * income / torch.clamp(prices[j], min=EPS), min=EPS)
                c["C_imp"][idx] = torch.clamp(b * (1 - theta_c) * income / torch.clamp(import_prices[j], min=EPS), min=EPS)
            else:
                c["C_dom"][j] = torch.clamp(b * income / torch.clamp(prices[j], min=EPS), min=EPS)
    return st


def _build_country_solution(state_c: Dict[str, torch.Tensor], state_o: Dict[str, torch.Tensor],
                            params_c: CountryParams, layout: EquilibriumLayout) -> Dict[str, Dict[str, np.ndarray]]:
    """将解状态转换为外部可消费的分块结构（中间品、消费、价格）。"""
    prices = state_c["price"]
    X_dom = state_c["X_dom"]
    X_imp = layout.expand_matrix(state_c["X_imp"])
    C_dom = state_c["C_dom"]
    C_imp = layout.expand_vector(state_c["C_imp"])

    X_I = torch.zeros_like(X_dom)
    X_I[:, layout.tradable_idx] = X_dom[:, layout.tradable_idx]
    C_I = torch.zeros_like(prices)
    C_I[layout.tradable_idx] = C_dom[layout.tradable_idx]
    C_total = torch.zeros_like(prices)
    C_total[layout.non_tradable_idx] = C_dom[layout.non_tradable_idx]
    import_prices = params_c.import_cost * state_o["price"]

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    return {
        "intermediate_inputs": {"X_ij": _to_np(X_dom), "X_I_ij": _to_np(X_I), "X_O_ij": _to_np(X_imp)},
        "final_consumption": {"C_j": _to_np(C_total), "C_I_j": _to_np(C_I), "C_O_j": _to_np(C_imp)},
        "prices": {"P_j": _to_np(prices), "P_I_j": _to_np(prices), "P_O_j": _to_np(import_prices)},
    }


def solve_initial_equilibrium(exogenous_params, max_iterations: int = 400, tolerance: float = 1e-8):
    """求解两国静态初始均衡（非线性最小二乘，log-变量）。

    方法：对所有正向量取对数，使用 scipy.optimize.least_squares 最小化残差；
    输出包含两国解的结构化块与收敛信息（是否成功/迭代次数/残差范数）。
    注意：本实现以“增加值份额”近似要素收入，若需显式要素市场可在 compute_income 处扩展。
    
    注意：scipy.optimize.least_squares 不支持 GPU 张量，因此在此函数内显式使用 INIT_DEVICE (CPU)。
    """
    params = normalize_model_params(exogenous_params)
    layout = EquilibriumLayout(params.home.alpha.shape[0], params.tradable_idx)
    guess_state = _initial_guess(params, layout)
    log_guess = torch.log(torch.clamp(layout.pack(guess_state), min=EPS)).detach().cpu().numpy()

    def residuals(v: np.ndarray) -> np.ndarray:
        # Phase 4: 显式使用 INIT_DEVICE (CPU) 进行 scipy 优化
        vec = torch.as_tensor(v, dtype=TORCH_DTYPE, device=INIT_DEVICE)
        res = _equilibrium_residuals(vec, layout, params)
        return res.detach().cpu().numpy()

    result = least_squares(
        residuals,
        log_guess,
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_iterations,
        verbose=0,
    )

    # Phase 4: 结果保持在 INIT_DEVICE 上，后续由 bootstrap_simulator 迁移到 COMPUTE_DEVICE
    sol_vec = torch.as_tensor(np.exp(result.x), dtype=TORCH_DTYPE, device=INIT_DEVICE)
    sol_state = layout.unpack(sol_vec)
    final_resid = float(np.linalg.norm(residuals(result.x)))

    home_block = _build_country_solution(sol_state["H"], sol_state["F"], params.home, layout)
    foreign_block = _build_country_solution(sol_state["F"], sol_state["H"], params.foreign, layout)
    return {
        "H": home_block,
        "F": foreign_block,
        "convergence_info": {
            "converged": bool(result.success),
            "iterations": int(result.nfev),
            "final_residual": final_resid,
            "solver_message": str(result.message),
        },
    }


def test_simple_case():
    params = create_symmetric_parameters()
    layout_params = normalize_model_params(params)
    layout = EquilibriumLayout(layout_params.home.alpha.shape[0], layout_params.tradable_idx)
    guess = _initial_guess(layout_params, layout)
    log_vec = torch.log(torch.clamp(layout.pack(guess), min=EPS))
    resid = _equilibrium_residuals(log_vec, layout, layout_params)
    print(f"初始残差范数: {np.linalg.norm(resid.detach().cpu().numpy()):.3e}")
    info = solve_initial_equilibrium(params, max_iterations=200, tolerance=1e-8)["convergence_info"]
    print(f"收敛: {info['converged']}, 残差: {info['final_residual']:.3e}, 迭代: {info['iterations']}")
    return info


__all__ = [
    "CountryParams",
    "ModelParams",
    "create_symmetric_parameters",
    "normalize_model_params",
    "EPS",
    "TORCH_DTYPE",
    "DEFAULT_DEVICE",
    "safe_log",
    "armington_share",
    "armington_price",
    "armington_quantity",
    "compute_output",
    "compute_marginal_cost",
    "compute_income",
    "solve_initial_equilibrium",
    "test_simple_case",
]
