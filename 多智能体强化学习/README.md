# PACER-MP：面向 Melting Pot 的伙伴自适应因果角色集成

PACER 将在线伙伴上下文、可分化角色专家和因果影响信用结合，目标是在 Melting Pot 的未见伙伴、未见场景与人口组合上获得更稳定的零样本/少样本协作能力。

目录内容：

- `src/pacer_marl/`：纯 NumPy 参考实现和 Melting Pot 2.4.0 适配器。
- `tests/`：单元测试与端到端测试。
- `configs/`：环境和 Melting Pot 实验配置。

## 1. 已验证的最小运行

代码仅强制依赖 NumPy。Python 3.10–3.12 均可：

```bash
cd 多智能体强化学习
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pacer_marl.cli smoke --episodes 30 --output runs/smoke
PYTHONPATH=src python -m unittest discover -s tests -v
```

`smoke` 会在程序化多智能体协调环境中训练、评估、保存权重并写出 CSV/JSON 指标。该路径不需要 GPU，用于验证算法、配置、检查点和指标链路是否完整。

## 2. 完整实验

```bash
PYTHONPATH=src python -m pacer_marl.cli train \
  --config configs/pacer_toy.json \
  --output runs/toy_seed7

PYTHONPATH=src python -m pacer_marl.cli evaluate \
  --config configs/pacer_toy.json \
  --checkpoint runs/toy_seed7/checkpoint_final.npz \
  --episodes 100
```

训练伙伴池与评测伙伴池彼此分离，因此评测数值不是训练集上的自我对弈分数。

## 3. Melting Pot 2.4.0

Melting Pot 依赖 DeepMind Lab2D。官方的 `dmlab2d` 预编译轮子并非覆盖所有 macOS/ARM 或 Python 组合，所以建议在 Linux x86_64 + Python 3.10/3.11 或官方 Codespace 中运行。

```bash
python -m pip install -r requirements-meltingpot.txt
PYTHONPATH=src python -m pacer_marl.cli train \
  --config configs/pacer_meltingpot.json \
  --output runs/mp_cooking_seed7
```

`pacer_meltingpot.json` 默认在 `collaborative_cooking__cramped`、`collaborative_cooking__circuit` 和 `collaborative_cooking__ring` 之间按 episode 采样训练。评测可将 `environment.kind` 改为 `meltingpot_scenario`，并将 `names` 设为如 `collaborative_cooking__cramped_0` 的官方 scenario；此时环境内置未知背景 bot，算法仅控制 focal player。

## 4. 主要输出

- `metrics.csv`：episode return、役中协调率、角色熵、伙伴类型。
- `checkpoint_*.npz`：角色专家、门控、价值函数和影响统计。
- `evaluation.json`：未见伙伴上的均值、标准差、最差十分位和各伙伴类型结果。

## 5. 实验规范

正式论文实验至少使用 5 个随机种子，报告均值与 95% bootstrap 置信区间。除平均集体回报外，还应报告未见伙伴交叉对弈、最差分位数、适配半衰期、环境后悔、角色使用熵与跨场景排名稳定性。
