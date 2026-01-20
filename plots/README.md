# Plots

This folder contains small Python utilities to plot experiment results saved under `output/...`.

## Plot metrics by best-of-n and temperature

The `plots/plot_experiment.py` script expects an experiment folder containing `sweep_results.csv`
(for example `output/exp1_bestof_sweep/`).

### Example (Windows / PowerShell)

```powershell
python -m plots.plot_experiment --experiment-dir output/exp1_bestof_sweep
```

Outputs are written by default to:

- `output/exp1_bestof_sweep/plots/`

You can change the output directory and file format:

```powershell
python -m plots.plot_experiment --experiment-dir output/exp1_bestof_sweep --out-dir plots_out/exp1 --format png
```

