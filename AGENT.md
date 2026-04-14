# Agent 代码阅读指南

## 项目定位

`trade_game` 是一个独立的两国多部门贸易政策动态博弈实验包。核心目标是复现实验型贸易政策博弈流程：先求初始静态均衡，再做 warmup，使供需和价格指数进入稳定区间，之后施加触发政策并让两国通过梯度或 LLM 做策略响应。

当前默认梯度入口使用 CN2017/US2017 数据，推荐 warmup 参数为：

```text
theta_price = 0.002
warmup_periods = 10000
rounds = 10
```

## 代码结构

- `trade_game/config.py`：统一配置 dataclass，包括国家、触发政策、优化器、约束、目标函数、参数源和 LLM 配置。
- `trade_game/params.py`：参数读取层。默认读取包内 `trade_game/io-final`，也可通过 `TRADE_GAME_DATA_DIR` 指向外部数据。
- `trade_game/runtime/model.py`：静态均衡和经济模型核心，包括 Armington/CES 价格与数量、产出、边际成本、收入和初始均衡求解。
- `trade_game/runtime/sim.py`：动态仿真器。负责价格更新、国内供给配给、外汇约束、关税/出口配额政策和历史记录。
- `trade_game/objective.py`：可微目标函数，聚合收入增长、贸易余额和价格稳定性。
- `trade_game/grad_game.py`：当前主要梯度博弈入口，默认配置已内联在 `default_grad_config()`。
- `trade_game/llm_game.py` 与 `trade_game/llm/`：LLM 决策入口和 OpenAI-compatible 客户端。
- `trade_game/param_search.py`：warmup 参数搜索工具。
- `trade_game/experiment_plots.py`：实验图表输出，生成宏观路径图和供需缺口图。

## 关键经济逻辑

动态仿真遵循“计划需求 -> 价格更新 -> 供给配给 -> 外汇约束 -> 重算产出/收入”的顺序：

1. `_plan_demands()` 根据当前价格、进口价格、收入和 Armington 份额计算计划中间需求与消费需求。
2. `_update_prices()` 使用供需缺口更新价格，缺口口径为 `planned_X_dom + planned_C_dom + planned_export - output`。当前默认使用按供给归一的缺口。
3. `_allocate_goods()` 当国内计划需求超过供给时，对中间品、消费和出口按同一比例配给。
4. `_allocate_imports_fx()` 使用出口收入约束进口支付，并可叠加对方出口供给上限。
5. `compute_output()` 与 `compute_income()` 根据成交投入和新价格更新下一期状态。

注意：`planned_export` 当前统一使用 `state.export_base`。这和价格更新及配给口径保持一致，也避免把上一期实际出口误当成下一期外生计划出口。

## 数据说明

包内带 CN/US 的 2017、2018 参数数据：

```text
trade_game/io-final/CN2017
trade_game/io-final/CN2018
trade_game/io-final/US2017
trade_game/io-final/US2018
```

每个国家目录包含该年份的 CSV 参数、价格/出口辅助表、`factor_params.csv` 和 `metadata.json`。包内还保留对应的 5-sector normalized Excel 原始表与参数说明 Markdown，便于追溯参数来源。如需扩展到其它年份，按同样文件名结构添加目录即可。

## 运行命令

安装依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

运行梯度博弈：

```bash
python -m trade_game.grad_game
```

运行参数搜索：

```bash
python -m trade_game.param_search --theta-values 0.002 --warmup-values 10000
```

运行 LLM 博弈：

```bash
export DEEPSEEK_API_KEY=...
python -m trade_game.llm_game
```

## 维护约定

- 优先保持 `trade_game` 独立，不依赖原大仓库中的 `analysis`、`eco_simu` 或根目录 `io-final`。
- 新增数据时优先放入 `trade_game/io-final/<COUNTRY_YEAR>/`，并确认 `load_io_params()` 可读取。
- 实验入口应显式记录关键参数：`theta_price`, `warmup_periods`, `rounds`, `decision_interval`, `lookahead_periods`, `active_sectors`。
- 大改动态规则后必须重新跑 `trade_game.param_search`，不要沿用旧 warmup 参数。
- 当前静态初始均衡求解仍可能达到迭代上限，研究级结论应同时报告 `convergence_info` 和 warmup 后供需/价格稳定性指标。
