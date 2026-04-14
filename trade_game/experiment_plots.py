"""
绑图工具模块：用于可视化博弈分析和供需状况。
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

# ==================== 用户配置区域 ====================
# 热启动期数配置：设置热启动阶段的期数，用于压缩显示
WARMUP_PERIODS = 1000  # 热启动期数，根据实际仿真配置修改

# 热启动压缩比例：热启动阶段在图表中占据的宽度比例
# 例如 0.1 表示热启动占 10% 宽度，博弈阶段占 90%
WARMUP_WIDTH_RATIO = 0.1
# 决策标签数值格式（保留小数位）
DECISION_DECIMALS = 3
# 决策标签最多显示的部门数（超出则省略）
DECISION_MAX_ITEMS = 4
# ==================== 用户配置区域 END ================


def _compress_x_axis(x, warmup_end, total_steps, width_ratio=WARMUP_WIDTH_RATIO):
    """
    对 x 轴进行非线性映射，压缩热启动阶段。
    
    参数：
        x: 原始时间步数组
        warmup_end: 热启动结束点
        total_steps: 总时间步数
        width_ratio: 热启动部分占总宽度的比例
    
    返回：
        x_compressed: 压缩后的 x 坐标
    """
    x = np.array(x)
    x_compressed = np.zeros_like(x, dtype=float)
    
    # 计算映射参数
    # 热启动阶段 [0, warmup_end] -> [0, width_ratio]
    # 博弈阶段 [warmup_end, total_steps] -> [width_ratio, 1]
    
    warmup_mask = x <= warmup_end
    game_mask = x > warmup_end
    
    # 热启动阶段：线性压缩
    if warmup_end > 0:
        x_compressed[warmup_mask] = (x[warmup_mask] / warmup_end) * width_ratio
    
    # 博弈阶段：线性映射
    game_range = total_steps - warmup_end
    if game_range > 0:
        x_compressed[game_mask] = width_ratio + ((x[game_mask] - warmup_end) / game_range) * (1 - width_ratio)
    
    return x_compressed


def _create_compressed_ticks(warmup_end, total_steps, events=None, width_ratio=WARMUP_WIDTH_RATIO):
    """
    创建压缩后的刻度位置和标签。
    使用事件期数（决策回合）作为刻度标签。
    
    参数：
        warmup_end: 热启动结束点
        total_steps: 总时间步数
        events: 政策事件列表，包含 'period' 字段
        width_ratio: 热启动压缩比例
    
    返回：
        tick_positions: 刻度在压缩坐标系中的位置
        tick_labels: 刻度对应的原始时间标签
    """
    # 热启动阶段：只显示 0 和 warmup_end
    warmup_labels = [0, warmup_end]
    warmup_positions = [0, width_ratio]
    
    # 博弈阶段：使用事件期数作为刻度
    if events and len(events) > 0:
        # 从事件中提取期数
        event_periods = sorted(set(e.get("period", 0) for e in events if e.get("period", 0) > warmup_end))
        game_labels = event_periods
    else:
        # 没有事件时，使用自动刻度
        game_steps = total_steps - warmup_end
        if game_steps <= 10:
            step = 1
        elif game_steps <= 50:
            step = 5
        else:
            step = game_steps // 5
        game_labels = list(range(warmup_end + step, total_steps + 1, step))
    
    # 确保包含最后一个时间点
    if total_steps not in game_labels:
        game_labels.append(total_steps)
    
    # 排除 warmup_end（已在热启动部分）
    game_labels = [l for l in game_labels if l > warmup_end]
    
    # 计算压缩后的位置
    game_positions = [width_ratio + ((l - warmup_end) / (total_steps - warmup_end)) * (1 - width_ratio) 
                      for l in game_labels]
    
    tick_positions = warmup_positions + game_positions
    tick_labels = warmup_labels + game_labels
    
    return tick_positions, tick_labels


def _add_warmup_shading(ax, width_ratio):
    """在热启动区域添加灰色背景。"""
    ax.axvspan(0, width_ratio, alpha=0.15, color='gray', label='Warmup')
    # 添加分隔线
    ax.axvline(width_ratio, color='gray', linestyle='--', alpha=0.5, linewidth=1)


def _format_sector_mapping(mapping, *, decimals=DECISION_DECIMALS, max_items=DECISION_MAX_ITEMS):
    """将 {sector: value} 显示为稳定短文本（固定小数位 + 可选省略）。"""
    if not mapping:
        return "{}"
    try:
        keys = sorted(mapping.keys(), key=lambda k: int(k))
    except Exception:
        keys = sorted(mapping.keys(), key=lambda k: str(k))
    total = len(keys)
    shown = keys[: int(max_items)] if max_items and total > int(max_items) else keys
    parts = []
    for k in shown:
        try:
            kk = int(k)
        except Exception:
            kk = k
        try:
            vv = float(mapping[k])
            vs = f"{vv:.{int(decimals)}f}"
        except Exception:
            vs = str(mapping[k])
        parts.append(f"{kk}:{vs}")
    suffix = f", ...+{total - len(shown)}" if total > len(shown) else ""
    return "{" + ", ".join(parts) + suffix + "}"


def _aggregate_decisions_from_events(events):
    """将 sim.policy_events 聚合成 {period: {'H':{'tariff':{},'quota':{}}, 'F':{...}}}。"""
    out = {}
    if not events:
        return out
    for e in events:
        if not isinstance(e, dict):
            continue
        period = e.get("period", None)
        if period is None:
            continue
        try:
            t = int(period)
        except Exception:
            continue
        country = (e.get("country") or "").upper()
        if country not in ("H", "F"):
            continue
        ev_type = e.get("type")
        if ev_type not in ("import_tariff", "export_control"):
            continue
        sectors = e.get("sectors") or {}
        if not isinstance(sectors, dict):
            continue
        bucket = out.setdefault(t, {}).setdefault(country, {})
        if ev_type == "import_tariff":
            bucket.setdefault("tariff", {}).update(sectors)
        elif ev_type == "export_control":
            bucket.setdefault("quota", {}).update(sectors)
    return out


def plot_game_analysis(summary, events, save_path, warmup_periods=None):
    """
    绑制博弈分析图表（收入、贸易差额、价格、产出）。
    热启动阶段会被按比例压缩显示。
    额外增加“决策时间线”面板：标注每次 H/F 的关税与配额决策。
    """
    if warmup_periods is None:
        warmup_periods = WARMUP_PERIODS
    
    metrics = ["income", "trade_balance_val", "price_mean", "output_growth"]
    metric_labels = {
        "income": "Income",
        "trade_balance_val": "Trade Balance",
        "price_mean": "Price Index (Mean)",
        "output_growth": "Real GDP Growth (%)"
    }
    
    steps = len(summary["H"]["income"])
    x_orig = np.arange(steps)
    
    # 检查是否需要压缩
    use_compression = warmup_periods > 0 and warmup_periods < steps - 5
    
    # 2x2 指标面板 + 底部决策时间线（跨两列）
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.7])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_timeline = fig.add_subplot(gs[2, :])
    decisions_by_t = _aggregate_decisions_from_events(events)
    
    for i, key in enumerate(metrics):
        ax = axes[i]
        if key in summary["H"]:
            val_H = np.array(summary["H"][key])
            val_F = np.array(summary["F"][key])
            
            if use_compression:
                # 使用压缩坐标
                x_plot = _compress_x_axis(x_orig, warmup_periods, steps - 1)
                
                ax.plot(x_plot, val_H, label="H", color="blue", linewidth=1.5)
                ax.plot(x_plot, val_F, label="F", color="red", linewidth=1.5)
                
                # 设置压缩后的刻度（使用事件期数）
                tick_pos, tick_labels = _create_compressed_ticks(warmup_periods, steps - 1, events=events)
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_labels, fontsize=8)
                ax.set_xlim(0, 1)
                
                # 添加热启动区域标记
                _add_warmup_shading(ax, WARMUP_WIDTH_RATIO)
                
                # 标记政策事件
                for e in (events or []):
                    t = e.get("period", 0)
                    if t < steps:
                        t_compressed = _compress_x_axis([t], warmup_periods, steps - 1)[0]
                        ax.axvline(t_compressed, color="green", linestyle="--", alpha=0.7, linewidth=1)
            else:
                # 不压缩
                ax.plot(x_orig, val_H, label="H", color="blue")
                ax.plot(x_orig, val_F, label="F", color="red")
                
                for e in (events or []):
                    t = e.get("period", 0)
                    if t < steps:
                        ax.axvline(t, color="gray", linestyle="--", alpha=0.5)
            
            ax.set_title(metric_labels.get(key, key))
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, linestyle=":", alpha=0.6)

    # -------------------------
    # 决策时间线面板（H/F）
    # -------------------------
    ax_timeline.set_title("Policy Decisions Timeline", fontsize=10)
    ax_timeline.set_yticks([1, 0])
    ax_timeline.set_yticklabels(["H", "F"])
    ax_timeline.set_ylim(-0.7, 1.7)
    ax_timeline.grid(True, axis="y", linestyle=":", alpha=0.4)

    if use_compression:
        tick_pos, tick_labels = _create_compressed_ticks(warmup_periods, steps - 1, events=events)
        ax_timeline.set_xticks(tick_pos)
        ax_timeline.set_xticklabels(tick_labels, fontsize=8)
        ax_timeline.set_xlim(0, 1)
        _add_warmup_shading(ax_timeline, WARMUP_WIDTH_RATIO)

        def x_map(t):
            return float(_compress_x_axis([t], warmup_periods, steps - 1)[0])
    else:
        ax_timeline.set_xlim(0, steps - 1)

        def x_map(t):
            return float(t)

    for t in sorted(decisions_by_t.keys()):
        if t < 0 or t >= steps:
            continue
        x = x_map(int(t))
        ax_timeline.axvline(x, color="green", linestyle="--", alpha=0.35, linewidth=1)

        for y, c, color in [(1, "H", "blue"), (0, "F", "red")]:
            d = decisions_by_t.get(int(t), {}).get(c, {})
            tar = _format_sector_mapping(d.get("tariff", {}) or {})
            quo = _format_sector_mapping(d.get("quota", {}) or {})
            text = f"τ{tar}\nq{quo}"
            ax_timeline.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_supply_demand_gap(sim, save_path, warmup_periods=None):
    """
    可视化各部门的供给与需求。
    热启动阶段会被按比例压缩显示。
    """
    if warmup_periods is None:
        warmup_periods = WARMUP_PERIODS
    
    history_H = sim.history["H"]
    history_F = sim.history["F"]
    steps = len(history_H)
    n_sectors = sim.params.home.alpha.shape[0]
    
    # 准备数据数组
    H_supply = np.zeros((steps, n_sectors))
    H_demand = np.zeros((steps, n_sectors))
    F_supply = np.zeros((steps, n_sectors))
    F_demand = np.zeros((steps, n_sectors))
    
    def compute_sd(country_sim, state, import_price=None):
        if import_price is None:
            import_price = state.imp_price
        outputs, pX_dom, pX_imp, pC_dom, pC_imp, _, _ = country_sim._plan_demands(state, import_price)
        supply = outputs.detach().cpu().numpy()
        planned_total = pX_dom.sum(dim=-2) + pC_dom + state.export_base
        demand = planned_total.detach().cpu().numpy()
        return supply, demand

    for t in range(steps):
        sH = history_H[t]
        sF = history_F[t]
        sup_H, dem_H = compute_sd(sim.home_sim, sH)
        H_supply[t] = sup_H
        H_demand[t] = dem_H
        sup_F, dem_F = compute_sd(sim.foreign_sim, sF)
        F_supply[t] = sup_F
        F_demand[t] = dem_F
    
    x_orig = np.arange(steps)
    
    # 检查是否需要压缩
    use_compression = warmup_periods > 0 and warmup_periods < steps - 5
    
    cols = 2
    rows = (n_sectors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()
    
    for s in range(n_sectors):
        ax = axes[s]
        
        if use_compression:
            x_plot = _compress_x_axis(x_orig, warmup_periods, steps - 1)
            
            # H 国：蓝色
            ax.plot(x_plot, H_supply[:, s], label="H Supply", color="blue", linewidth=1.5)
            ax.plot(x_plot, H_demand[:, s], label="H Demand", color="blue", linestyle="--", linewidth=1.5)
            
            # F 国：红色
            ax.plot(x_plot, F_supply[:, s], label="F Supply", color="red", linewidth=1.5)
            ax.plot(x_plot, F_demand[:, s], label="F Demand", color="red", linestyle="--", linewidth=1.5)
            
            # 设置压缩后的刻度
            tick_pos, tick_labels = _create_compressed_ticks(warmup_periods, steps - 1)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_xlim(0, 1)
            
            # 添加热启动区域标记
            _add_warmup_shading(ax, WARMUP_WIDTH_RATIO)
        else:
            ax.plot(x_orig, H_supply[:, s], label="H Supply", color="blue", linewidth=1.5)
            ax.plot(x_orig, H_demand[:, s], label="H Demand", color="blue", linestyle="--", linewidth=1.5)
            ax.plot(x_orig, F_supply[:, s], label="F Supply", color="red", linewidth=1.5)
            ax.plot(x_orig, F_demand[:, s], label="F Demand", color="red", linestyle="--", linewidth=1.5)
        
        ax.set_title(f"Sector {s}")
        ax.grid(True, alpha=0.3)
        
        if s == 0:
            ax.legend(loc='best', fontsize='small')
            
    # 隐藏未使用的子图
    for k in range(n_sectors, len(axes)):
        axes[k].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
