# RAG-UAV-Network-Slicing

> **Archived.** This repository is a snapshot of a research prototype and is no longer maintained.

LLM / RAG-guided reward-weight selection for multi-agent reinforcement learning (MAPPO) in a UAV-based 5G network slicing simulator.

## Attribution

The UAV network slicing environment, the MAPPO agents and the training pipeline in this repository are based on the work of Thanh-Dao Nguyen et al.:

> T.-D. Nguyen, N.-T. Nguyen, T.-D. Nguyen, N. Van Huynh, D.-H. Tran, S. Chatzinotas, "Collaborative Intelligence for UAV-Satellite Network Slicing: Towards a Joint QoS-Energy-Fairness MADRL Optimization," arXiv:2512.08322, 2025. <https://arxiv.org/abs/2512.08322>

```bibtex
@misc{nguyen2025uavsatellite,
  title         = {Collaborative Intelligence for UAV-Satellite Network Slicing: Towards a Joint QoS-Energy-Fairness MADRL Optimization},
  author        = {Nguyen, Thanh-Dao and Nguyen, Ngoc-Tan and Nguyen, Thai-Duong and Huynh, Nguyen Van and Tran, Dinh-Hieu and Chatzinotas, Symeon},
  year          = {2025},
  eprint        = {2512.08322},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2512.08322}
}
```

This fork preserves the original commit history (authored by Thanh-Dao Nguyen) and adds the LLM / RAG layer on top:

- environment-aware reward weight selection through an LLM (zero-shot, few-shot, knowledge-base retrieval),
- a pgvector knowledge base of past experiences,
- a human-in-the-loop workflow,
- evaluation flow and plots.

If you reuse this code, please cite the paper above.

## Idea

MAPPO agents optimise a reward of the form `w_qos * QoS + w_energy * energy + w_fairness * fairness`. Fixed weights fit one traffic condition poorly across changing conditions. Here, an LLM reads the current environment state (slice mix, mobility mix, UE arrival/departure rates), derives a "pressure" analysis, and proposes reward weights. Training then continues with those weights.

Three prompting modes:

| Mode | Prompt | Description |
|------|--------|-------------|
| `zs` | [prompt_zs.md](prompt_zs.md) | Zero-shot, no examples |
| `fs` | [prompt_fs.md](prompt_fs.md) | Few-shot, static hand-picked examples |
| `kb` | [prompt_kb.md](prompt_kb.md) | RAG: top-k similar past experiences retrieved from a pgvector DB by environment and evaluation vectors |
| `none` | – | Baseline, static reward weights, no LLM |

## Repository layout

| Path | Purpose |
|------|---------|
| [main.py](main.py) | Entry point: LLM-in-the-loop training workflow |
| [ml_main.py](ml_main.py) | Variant without the LLM/DB loop |
| [environment.py](environment.py) | UAV network slicing simulator |
| [agents_mappo.py](agents_mappo.py), [train_mappo.py](train_mappo.py) | MAPPO agent and training manager |
| [env_sampling.py](env_sampling.py) | Sampling of environment configs |
| [semantic_analyzer.py](semantic_analyzer.py) | Environment "pressure" analysis |
| [prompt_builder.py](prompt_builder.py) | Jinja2 prompt rendering |
| [llm.py](llm.py) | LLM client (OpenRouter, OpenAI-compatible API) |
| [kb_handler.py](kb_handler.py), [db_config.py](db_config.py), [init_kb.py](init_kb.py) | Knowledge base on PostgreSQL + pgvector |
| [config/](config) | Environment, agent and training YAML configs |
| [eval_plots/](eval_plots) | Evaluation figures |

## Setup

Requires Python 3.10+ and, for `kb` mode, a PostgreSQL instance with the `pgvector` extension.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
MINIMAX=<OpenRouter API key>
DATABASE_URL=postgresql://user:password@host:port/dbname
```

The model name and endpoint are hardcoded in [llm.py](llm.py).

## Usage

```bash
python main.py --mode kb --seed 42            # RAG mode, runs until stopped
python main.py --mode fs --eval 20 --seed 42  # few-shot, 20 evaluation samples
python main.py --mode none                    # static-weight baseline
```

Arguments:

- `--mode {zs,fs,kb,none}`: prompting strategy (default `kb`)
- `--eval N`: run N sampled evaluation environments, `0` runs indefinitely
- `--seed S`: random seed (default 42)

At startup the script asks for `auto` or `hybrid` (human-in-the-loop) operation. In hybrid mode you can approve the LLM analysis, re-run it, or give feedback, and decide whether results are added to the knowledge base.

Logs are written to `logs/`. Trained checkpoints go to `saved_models/`.

## Results

| | |
|---|---|
| ![mean](eval_plots/mean.png) | ![avg improvement](eval_plots/avg_improvement.png) |
| ![low pressure](eval_plots/low_pressure.png) | ![high pressure](eval_plots/high_pressure.png) |

## Status and limitations

- Archived research prototype; no support or updates.
- The knowledge base contents and trained models are not included.
- Dependency versions are pinned to what was used at the time.
- No license file is included. The original code repository is private and its license is unknown, so reuse terms for the base code are unclear; contact the original authors before reusing it.
