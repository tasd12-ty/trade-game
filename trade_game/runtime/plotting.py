"""
eco_simu 可视化模块
===================

提供两国经济仿真结果的图表绘制功能。

主要函数:
- plot_history: 两国宏观指标时间序列
- plot_sector_paths: 单国多部门对比
- plot_diagnostics: 综合诊断面板
- plot_history_agent_view: 智能体决策标注视图
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# =============================================================================
# 全局配置
# =============================================================================

# 两国标准配色
COLORS = {'H': '#1f77b4', 'F': '#ff7f0e'}
LINESTYLES = {'H': '-', 'F': '--'}
COUNTRY_LABELS = {'H': 'Home', 'F': 'Foreign'}


def _configure_chinese_fonts() -> None:
    """尝试配置中文字体支持，按优先级查找可用字体。"""
    candidates = [
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 
        'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS',
    ]
    for font in candidates:
        try:
            fm.findfont(font, fallback_to_default=False)
            plt.rcParams['font.family'] = font
            plt.rcParams['font.sans-serif'] = [font]
            break
        except ValueError:
            continue
    plt.rcParams['axes.unicode_minus'] = False


_configure_chinese_fonts()


# =============================================================================
# 辅助函数
# =============================================================================

def _safe_rebase_growth(series: np.ndarray) -> np.ndarray:
    """计算相对于首期的增长率(%)。
    
    Args:
        series: 原始时间序列
        
    Returns:
        增长率序列，首期为0%
    """
    if series.size == 0:
        return series
    base = float(series[0])
    if abs(base) < 1e-12:
        return np.zeros_like(series)
    return ((series / base) - 1.0) * 100.0


def _trim_data(summary: Dict[str, Dict], trim: int) -> Dict[str, Dict]:
    """裁剪warmup期数据并重新计算增长率。
    
    Args:
        summary: sim.summarize_history()的输出
        trim: 裁剪的期数
        
    Returns:
        裁剪后的数据字典
    """
    result = {}
    for country in ("H", "F"):
        src = summary.get(country, {})
        data = {}
        for key, val in src.items():
            arr = np.array(val, float)
            data[key] = arr[trim:] if trim else arr
        # 重新计算增长率
        data["income_growth"] = _safe_rebase_growth(np.array(data.get("income", []), float))
        data["output_growth"] = _safe_rebase_growth(np.array(data.get("output_sum", []), float))
        result[country] = data
    return result


def _save_figure(fig: plt.Figure, save_path: Optional[str], show: bool) -> None:
    """统一的图表保存/显示逻辑。"""
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def _calc_warmup_trim(total_len: int, warmup: int) -> int:
    """计算有效的warmup裁剪期数。"""
    trim = max(int(warmup or 0), 0)
    if total_len > 0 and trim >= total_len:
        trim = max(total_len - 1, 0)
    return trim


# =============================================================================
# 主绘图函数
# =============================================================================

def plot_history(
    sim: Any,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    warmup: int = 0,
) -> None:
    """绘制两国宏观指标时间序列。
    
    Args:
        sim: TwoCountryDynamicSimulator实例
        metrics: 要绘制的指标列表，默认['income', 'output_sum', 'trade_balance', 'income_growth']
        save_path: 保存路径（可选）
        show: 是否显示图表
        warmup: 裁剪的warmup期数
    """
    metrics = metrics or ["income", "output_sum", "trade_balance", "income_growth"]
    summary = sim.summarize_history()
    total_len = len(sim.history["H"])
    trim = _calc_warmup_trim(total_len, warmup)
    periods = np.arange(total_len - trim)
    data = _trim_data(summary, trim)

    # 绑定子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    # 绑定每个指标
    for ax, metric in zip(axes, metrics):
        if metric not in data["H"]:
            continue
        for country in ("H", "F"):
            ax.plot(
                periods, data[country][metric],
                color=COLORS[country], linestyle=LINESTYLES[country],
                linewidth=2, marker='o', markersize=2, label=COUNTRY_LABELS[country]
            )
        ylabel = metric.replace("_", " ").title()
        if "growth" in metric:
            ylabel += " (%)"
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(frameon=True)

    axes[-1].set_xlabel("Periods", fontsize=10)
    fig.suptitle("Two-Country Economic Simulation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)

    _save_figure(fig, save_path, show)


def plot_sector_paths(
    sim: Any,
    country: str,
    metric: str,
    sectors: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    relative: bool = True,
    warmup: int = 0,
) -> None:
    """绘制单国多部门指标路径。
    
    Args:
        sim: TwoCountryDynamicSimulator实例
        country: 'H' 或 'F'
        metric: 指标类型 ('price', 'output', 'export', 'consumption_domestic', 
                'consumption_import', 'intermediate_domestic', 'intermediate_import')
        sectors: 部门索引列表（默认全部）
        relative: 是否显示相对变化(%)
        warmup: 裁剪的warmup期数
    """
    country = country.upper()
    states = sim.history[country]
    trim = _calc_warmup_trim(len(states), warmup)
    states = states[trim:]
    
    if not states:
        return

    # 提取部门数据
    METRIC_EXTRACTORS = {
        "price": lambda s: s.price.detach().cpu().numpy(),
        "output": lambda s: s.output.detach().cpu().numpy(),
        "export": lambda s: s.export_actual.detach().cpu().numpy(),
        "consumption_domestic": lambda s: s.C_dom.detach().cpu().numpy(),
        "consumption_import": lambda s: s.C_imp.detach().cpu().numpy(),
        "intermediate_domestic": lambda s: s.X_dom.sum(dim=0).detach().cpu().numpy(),
        "intermediate_import": lambda s: s.X_imp.sum(dim=0).detach().cpu().numpy(),
    }
    
    if metric not in METRIC_EXTRACTORS:
        raise ValueError(f"Unsupported metric: {metric}. Options: {list(METRIC_EXTRACTORS.keys())}")
    
    series = np.vstack([METRIC_EXTRACTORS[metric](s) for s in states])
    periods = np.arange(series.shape[0])
    sectors = sectors or list(range(series.shape[1]))
    
    # 计算相对变化
    if relative and series.shape[0] > 1:
        display = np.zeros_like(series)
        for i in range(series.shape[1]):
            base = max(series[0, i], 1e-9)
            display[:, i] = ((series[:, i] / base) - 1) * 100 if base > 1e-9 else 0
        ylabel_suffix = " - Relative Change (%)"
    else:
        display = series
        ylabel_suffix = ""

    # 绑定图表
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    
    for i, sec in enumerate(sectors):
        ax.plot(periods, display[:, sec], color=colors[i], linewidth=2, label=f"Sector {sec}")

    TITLES = {
        "price": "Price", "output": "Output", "export": "Export Value",
        "consumption_domestic": "Domestic Consumption", "consumption_import": "Import Consumption",
        "intermediate_domestic": "Domestic Intermediate", "intermediate_import": "Import Intermediate",
    }
    
    title = f"{country} - {TITLES.get(metric, metric)}{ylabel_suffix}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Periods", fontsize=10)
    ax.set_ylabel(TITLES.get(metric, metric) + ylabel_suffix, fontsize=10)
    
    if relative:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    _save_figure(fig, save_path, show)


def plot_diagnostics(
    sim: Any,
    save_path: Optional[str] = None,
    show: bool = False,
    warmup: int = 0,
) -> None:
    """绘制综合诊断面板（2x2子图 + 统计摘要）。
    
    包含: 收入水平、贸易平衡、收入增长率、统计摘要
    """
    summary = sim.summarize_history()
    total_len = len(sim.history["H"])
    trim = _calc_warmup_trim(total_len, warmup)
    periods = np.arange(total_len - trim)
    data = _trim_data(summary, trim)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # 收入水平
    ax1.plot(periods, data["H"]["income"], 'b-', linewidth=2, label="Home")
    ax1.plot(periods, data["F"]["income"], 'r--', linewidth=2, label="Foreign")
    ax1.set_title("Income Levels", fontweight='bold')
    ax1.set_ylabel("Income")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 贸易平衡
    ax2.plot(periods, data["H"]["trade_balance"], 'g-', linewidth=2, label="Home")
    ax2.plot(periods, data["F"]["trade_balance"], 'm--', linewidth=2, label="Foreign")
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title("Trade Balance", fontweight='bold')
    ax2.set_ylabel("Trade Balance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 收入增长率
    ax3.plot(periods, data["H"]["income_growth"], 'b-', linewidth=2, label="Home")
    ax3.plot(periods, data["F"]["income_growth"], 'r--', linewidth=2, label="Foreign")
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title("Income Growth Rate", fontweight='bold')
    ax3.set_ylabel("Growth Rate (%)")
    ax3.set_xlabel("Periods")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 统计摘要
    ax4.axis('off')
    h_growth = data["H"]["income_growth"][-1]
    f_growth = data["F"]["income_growth"][-1]
    h_vol = np.std(data["H"]["income_growth"])
    f_vol = np.std(data["F"]["income_growth"])
    
    stats_text = f"""Diagnostic Summary ({len(periods)} periods)

Income Performance:
• Home Final Growth: {h_growth:.1f}%
• Foreign Final Growth: {f_growth:.1f}%

Volatility:
• Home σ: {h_vol:.1f}%
• Foreign σ: {f_vol:.1f}%

Trade Balance:
• Home Final: {data['H']['trade_balance'][-1]:.2f}
• Foreign Final: {data['F']['trade_balance'][-1]:.2f}
"""
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3), va='top')

    plt.suptitle("Two-Country Economic Simulation - Dashboard", fontsize=14, fontweight='bold')
    plt.tight_layout()

    _save_figure(fig, save_path, show)


def plot_history_agent_view(
    sim: Any,
    agent_log: List[Dict],
    k_per_step: int,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    annotate_decisions: bool = True,
    warmup: Optional[int] = None,
) -> None:
    """智能体视角的历史图，标注每回合决策。
    
    Args:
        sim: 仿真器实例
        agent_log: 智能体日志列表，每项含 {t, obs, action, reward}
        k_per_step: 每回合推进的步数
        metrics: 指标列表
        annotate_decisions: 是否标注决策摘要
        warmup: 裁剪期数（None则自动推断）
    """
    metrics = metrics or ["income", "output_sum", "trade_balance", "income_growth"]
    summary = sim.summarize_history()

    total_len = len(sim.history.get("H", []))
    rounds = len(agent_log or [])
    k = max(k_per_step, 1)
    
    # 计算warmup
    if warmup is None:
        warmup_steps = max(total_len - 1 - rounds * k, 0)
    else:
        warmup_steps = _calc_warmup_trim(total_len, warmup)

    periods = np.arange(total_len - warmup_steps)
    data = _trim_data(summary, warmup_steps)

    # 绑定子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in data["H"]:
            continue
        for country in ("H", "F"):
            ax.plot(
                periods, data[country][metric],
                color=COLORS[country], linestyle=LINESTYLES[country],
                linewidth=2, marker='o', markersize=2, label=COUNTRY_LABELS[country]
            )
        ylabel = metric.replace("_", " ").title()
        if "growth" in metric:
            ylabel += " (%)"
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(frameon=True)

    axes[-1].set_xlabel("Agent Rounds", fontsize=10)
    fig.suptitle("Two-Country Simulation — Agent View", fontsize=14, fontweight='bold')

    # 标注决策
    if annotate_decisions and rounds > 0:
        ax0 = axes[0]
        for i in range(rounds):
            x = i * k
            ax0.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
            
            # 简短摘要
            action = agent_log[i].get('action', {}) if i < len(agent_log) else {}
            if action:
                h_act = action.get('H', {})
                f_act = action.get('F', {})
                label = f"H:τ{len(h_act.get('import_tariff', {}))} F:τ{len(f_act.get('import_tariff', {}))}"
                ymax = max(data['H'][metrics[0]].max(), data['F'][metrics[0]].max())
                ax0.annotate(label, xy=(x, ymax), fontsize=6, rotation=90, va='bottom', ha='center', color='gray')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08)

    _save_figure(fig, save_path, show)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "plot_history",
    "plot_sector_paths", 
    "plot_diagnostics",
    "plot_history_agent_view",
    "plot_history_with_events",
]


def plot_history_with_events(
    sim: Any,
    policy_events: List[Dict],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    warmup: int = 0,
    title: str = "Two-Country Simulation - Policy Events",
) -> None:
    """绘制两国宏观指标时间序列，并标注政策事件。
    
    Args:
        sim: TwoCountryDynamicSimulator实例
        policy_events: 政策事件列表，每项含 {t, actor, action, note}
                       格式: [{"t": 0, "actor": "H", "action": "tariff +20%", "note": "..."}]
                       如果 t 时刻无事件，可传入 {"t": t, "actor": "-", "action": "无动作"}
        metrics: 要绘制的指标列表
        save_path: 保存路径（可选）
        show: 是否显示图表
        warmup: 裁剪的warmup期数
        title: 图表标题
    """
    metrics = metrics or ["income", "output_sum", "trade_balance", "income_growth"]
    summary = sim.summarize_history()
    total_len = len(sim.history["H"])
    trim = _calc_warmup_trim(total_len, warmup)
    periods = np.arange(total_len - trim)
    data = _trim_data(summary, trim)

    # 创建子图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    # 绑定每个指标
    for ax, metric in zip(axes, metrics):
        if metric not in data["H"]:
            continue
        for country in ("H", "F"):
            ax.plot(
                periods, data[country][metric],
                color=COLORS[country], linestyle=LINESTYLES[country],
                linewidth=2, marker='o', markersize=2, label=COUNTRY_LABELS[country]
            )
        ylabel = metric.replace("_", " ").title()
        if "growth" in metric:
            ylabel += " (%)"
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(frameon=True, loc='upper right')

    # 标注政策事件
    if policy_events:
        ax0 = axes[0]
        ymin, ymax = ax0.get_ylim()
        y_range = ymax - ymin
        
        # 按时间分组事件
        events_by_t = {}
        for ev in policy_events:
            t = ev.get("t", 0)
            if t not in events_by_t:
                events_by_t[t] = []
            events_by_t[t].append(ev)
        
        # 标注每个决策时间点
        for t, evs in sorted(events_by_t.items()):
            # 绘制垂直线
            for ax in axes:
                ax.axvline(x=t, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # 构建标注文本
            lines = [f"t={t}"]
            for ev in evs:
                actor = ev.get("actor", "?")
                action = ev.get("action", "?")
                lines.append(f"{actor}: {action}")
            label_text = "\n".join(lines)
            
            # 在顶部子图上添加标注
            ax0.annotate(
                label_text,
                xy=(t, ymax),
                xytext=(t + 2, ymax + y_range * 0.05),
                fontsize=8,
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5)
            )

    axes[-1].set_xlabel("Periods", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)

    _save_figure(fig, save_path, show)

