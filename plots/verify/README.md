# Verification pass, 2026-09-05

Ground-truth re-derivation of every ViT-B / IN-1k run and the init/end-state statistics behind
docs section 0d. Nothing here trusts an arm NAME: configurations come from each run's own
`Namespace(...)` dump in `logs/ft_<job>_<name>.out` (`arm_truth.py`) or, for runs launched
before those logs existed, from the `args` stored in the run's full checkpoint (`inventory_old.py`).

| script | what it does | output |
|---|---|---|
| `arm_truth.py` | parse every slurm `.out`: job name, SLURM_ID, seed, init flags, first start (pre/post rank-sync fix), epochs + last-epoch acc from `log.txt` | `cache/verify/arm_truth.json` |
| `sig_print.py [arms]` | distinct init signatures per job name, with accuracies | stdout |
| `inventory_old.py` | result dirs with no `.out`: recover args from `checkpoint-*.pth`, name by construction | `cache/verify/old_runs.json`, `old_runs_named.json` |
| `proc_shape.py` | per-slice marginal shape of the proc checkpoint; Student-t fit quality | `cache/verify/proc_shape.json` |
| `gen_dump_cmds.py` + `run_dumps.py` | build every arm's init EXACTLY via `plots/dump_init.py` (main.py's own path), 8 GPUs | `results/init_dumps/<arm>_s0.pth` |
| `init_dist_stats.py` | per-slice scale/shape/rank + LayerNorm-composed effective scales of each dumped init | `results/init_dumps/init_dist_stats.json` |
| `init_forward_stats.py` | rho / attention entropy per block at init on 64 val images | `results/init_dumps/init_forward_stats.json` |
| `end_state_stats.py` | the same at epoch 299 for every completed run | `results/init_dumps/end_state_stats.json` |
| `trajectories.py`, `fit_vs_gen.py` | per-arm trajectories; final train loss vs test accuracy (fig15) | `cache/verify/trajectories.json`, `out/fig15_fit_vs_gen.png` |
| `join_analysis.py`, `end_state_analysis.py`, `fig_end_state.py` | join accuracies with init / end-state statistics | stdout, `out/fig16_end_state.png` |

Run order: arm_truth -> inventory_old -> trajectories -> fit_vs_gen; dumps (sbatch, L40S) -> init_dist_stats / init_forward_stats / end_state_stats (sbatch) -> join_analysis / end_state_analysis / fig_end_state.

## Pre-launch verification of new arms (2026-09-05)

`verify_new_arms.py <gaussian_twin_dump> <rho_dump>` compares a Gaussian-twin init against
`ftbqmlnvo_s0` (all 2-D scales within 1%, kurtosis 3, identical LayerNorm value multisets, zero
linear biases) and a rho-matched init against `ftb4e3fix_s0`'s forward profile. Dumps for the
check can use `results/init_dumps/imnet_small` (val images symlinked as the training root) to
skip the 1.28M-file NFS index: the rho-matching only needs 5000 reference images, and its
train-vs-val agreement is ~3% in blocks 1-8 (docs 0d.8). `queue_new_arms.sh` submits 3 seeds per
arm with the shared-SLURM_ID convention.
