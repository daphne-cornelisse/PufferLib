# Craftax Mini Anchor Regularization

This workflow trains a frozen behavior-cloned anchor policy from the vendored
Craftax Mini human trajectories, then uses that anchor as a KL regularizer during
RL.

## Train the anchor

The anchor trainer replays human trajectories through native Craftax Mini to
collect exact observation/action pairs. By default it trains only successful
human trajectories.

```bash
python scripts/train_craftax_mini_anchor.py
```

Output:

```text
resources/craftax_mini/craftax_mini_anchor_policy.bin
```

To include failed trajectories too:

```bash
python scripts/train_craftax_mini_anchor.py --include-failures
```

## Use the anchor in RL

`config/craftax_mini.ini` points at the anchor:

```ini
[train]
use_reg = false
reg_coef = 0.01
anchor_model_path = 'resources/craftax_mini/craftax_mini_anchor_policy.bin'
```

Turn regularization on for RL:

```bash
python -m pufferlib.pufferl train craftax_mini --train.use-reg true
```

The native CUDA backend loads `train.anchor_model_path` into a frozen policy when
`use_reg` is true. The PPO loss adds:

```text
reg_coef * KL(policy || anchor)
```

for discrete action policies.

## Short ablation run

On machines where the CPU vector training backend is unavailable or unstable,
use the standalone Craftax Mini ablation harness:

```bash
python scripts/train_craftax_mini_reg_ablation.py
```

Outputs:

```text
resources/craftax_mini/craftax_mini_rl_no_reg.bin
resources/craftax_mini/craftax_mini_rl_reg.bin
```

This script trains two small actor-critic policies with the same seeds and
settings:

- `no_reg`: plain RL
- `reg`: RL plus KL to `craftax_mini_anchor_policy.bin`

