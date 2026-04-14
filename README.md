# Trade Game

两国多部门贸易政策动态博弈实验包。项目已从原仓库中抽离，包含 `trade_game` 运行代码和 CN/US 2017、2018 参数数据，可直接运行梯度博弈、LLM 博弈和 warmup 参数搜索。

## 环境安装

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 数据

默认数据位于：

```text
trade_game/io-final/CN2017
trade_game/io-final/CN2018
trade_game/io-final/US2017
trade_game/io-final/US2018
```

包内还保留了对应的 5-sector normalized Excel 原始表和参数说明 Markdown，便于追溯参数来源。

`trade_game.params.load_io_params()` 默认读取包内数据。若要使用外部数据目录，可设置：

```bash
export TRADE_GAME_DATA_DIR=/path/to/io-final
```

## 主要入口

运行梯度博弈，默认使用 CN2017/US2017、`theta_price=0.002`、`warmup_periods=10000`、`rounds=10`：

```bash
python -m trade_game.grad_game
```

输出默认写入：

```text
results_grad_real_io_CN2017_US2017_H50_theta0002_warmup10000_r10/
```

运行 warmup 参数搜索：

```bash
python -m trade_game.param_search --theta-values 0.002 --warmup-values 10000
```

运行 LLM 博弈需要安装 `openai` 并设置对应密钥，例如 DeepSeek：

```bash
export DEEPSEEK_API_KEY=...
python -m trade_game.llm_game
```

## 快速检查

```bash
python -m py_compile $(find trade_game -name '*.py')
python - <<'PY'
from trade_game.params import load_io_params
p = load_io_params("CN2017", "US2017")
print(p["H"]["alpha_ij"].shape, p["F"]["alpha_ij"].shape)
PY
```

## 维护说明

给后续 agent 阅读的代码设计说明见 [AGENT.md](AGENT.md)。
