# Potential functions for first-order methods — Python

This repository is a file-by-file Python translation of the [original MATLAB code](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods) by Adrien Taylor and Francis Bach. It uses NumPy and CVXPY to formulate the semidefinite programs for finding potential functions, and Matplotlib to plot the results.

The translation follows the original directories, filenames, settings sections, variable names, calculation order, linear matrix inequalities (LMIs), plots, and data format. Each experiment contains its implementation directly, from setup through optimization and saving. Edit the settings at the top of a script and run it.

## Installation and usage

Verified with Python **3.12.13** in an existing conda environment named `py312`. If you have that environment, run:

```bash
conda activate py312
python -m pip install -r requirements.txt
```

`requirements.txt` pins NumPy, CVXPY, Matplotlib, CLARABEL, and the upstream MOSEK dependency. The scripts preserve the original `solver = "MOSEK"` setting. **The numerical results reported for this port were obtained with CLARABEL.** To reproduce those runs, change the setting at the top of the script:

```python
solver = "CLARABEL"
```

Then run the experiment:

```bash
python 1_Deterministic_SmoothConvex/A_GradientDescent.py
```

Edit `L`, `m`, `N`, `delta`, `relax`, the initial and final potentials, `pplot`, and `ssave` in each script, as applicable. Data is saved in the `SaveData/` directory beside the script. Change `nname` to preserve an existing result; saving to the same path overwrites that file.

## CLARABEL measurements

Measured runs of the standalone Python scripts with CLARABEL 0.11.1 and CVXPY 1.9.3. Parameters: `L=1`, `m=0`, stochastic `n=2`, bounded-variance `alpha=0`, and weak-growth `rho=2`. The table lists `N` and `relax`; these settings can differ from script defaults. Plotting and saving were disabled, and numerical-library threads were limited to one.

| Experiment | N | relax | Measured quantity / final objective | Status | Max raw constraint violation | Solver time (s) |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 1/A — Gradient descent | 100 | 0 | `bk_N = 10200.9860918` | `optimal` | `1.761e-08` | 0.022558 |
| 1/B — Proximal gradient | 100 | 0 | `bk_N = 9963.04800805` | `optimal_inaccurate` | `6.893e-05` | 0.134143 |
| 1/C — First accelerated method | 100 | 2 | `dk_N = 4086.68445478` | `optimal_inaccurate` | `3.787e-05` | 0.219248 |
| 1/D — Second accelerated method | 100 | 0 | `dk_N = 5270.39193485` | `optimal` | `5.352e-09` | 0.068079 |
| 2/A — SGD | 100 | 0 | `tau = 200.99997971`; `sum_e = 2060.4544691` | `optimal` | `2.039e-08` | 0.164970 |
| 2/B — SGD with averaging | 100 | 0 | `tau = 101.999928675`; `sum_e = 48.6614291052` | `optimal_inaccurate` | `4.483e-05` | 0.281515 |
| 2/C — SGD with primal averaging | 100 | 0 | `sum_e = 49.5004087844` | `optimal_inaccurate` | `7.363e-07` | 0.040916 |
| 2/D — Evaluation at averaged point | 100 | 0 | `tau = 99.9973689444`; `sum_e = 49.4987257886` | `optimal_inaccurate` | `3.834e-07` | 0.076993 |
| 3/A — Overparametrized | 100 | 3 | `dk_N = 100.179225929` | `optimal_inaccurate` | `1.029e-05` | 0.572055 |
| 4/A — Weak growth, primal averaging | 100 | 1 | `objective = 0` (feasibility) | `optimal` | `1.597e-10` | 0.037386 |
| 5/A — Variance at optimum | 5 | 3 | — | `NumericalError` | — | 0.031384 |

Solver time is the solver-reported solve time, excluding compilation. For the two-stage experiments 2/A, 2/B, and 2/D, it is the sum of both stages; the reported quantities, status, and constraint violation are from the final stage.

Constraint violation is the maximum violation of the original raw constraints at the returned solution. The failed 5/A run returned no primal solution; its unavailable values are marked —.

These are CLARABEL measurements only. Upstream saved `.dat` values are not presented as a new MOSEK benchmark.

## Repository layout

The 16 original MATLAB files correspond to 16 Python files: 11 experiments and five `saveData.py` helpers, one in each directory. The 11 original `.dat` files are also retained. The root contains this README, `requirements.txt`, and `.gitignore`.

| Directory | Experiment | Paper reference |
| --- | --- | --- |
| `1_Deterministic_SmoothConvex` | `A_GradientDescent.py` | Section 3.2.1, Theorem 3, Figure 1 |
| Same directory | `B_ProximalGradientDescent.py` | Appendix C.2, Theorem 9 |
| Same directory | `C_StepsizeSelection_FirstAcceleratedMethod.py` | Appendix C.3, Theorem 10, Figure 3 |
| Same directory | `D_StepsizeSelection_SecondAcceleratedMethod.py` | Appendix C.3, Theorem 11, Figure 4 |
| `2_Stochastic_BoundedVariance` | `A_StochasticGradientDescent.py` | Section 3.2.2, Theorem 5, Figure 2 |
| Same directory | `B_StochasticGradientDescentWithAveraging.py` | Section 3.2.2, Theorem 6 |
| Same directory | `C_StochasticGradientDescentWithPrimalAveraging.py` | Section 3.2.2, Theorem 7 |
| Same directory | `D_StochasticGradientDescent_EvaluationAtAveragedPoint.py` | Appendix D.4, Theorem 12 |
| `3_Stochastic_Overparametrized` | `A_ParameterSelection.py` | Section 4, Appendix E, Theorem 8, Figure 5 |
| `4_Stochastic_WeakGrowth` | `A_StochasticGradientDescentWithPrimalAveraging.py` | Appendix F |
| `5_Stochastic_VarianceAtOptimum` | `A_ParameterSelection.py` | Appendix G |

Each experiment directory contains `saveData.py` and `SaveData/`.

## Correspondence with MATLAB

- `sdpvar` becomes `cvxpy.Variable`, matrix multiplication uses `@`, and transposes use `.T`.
- MATLAB cells such as `Q{k}` become dictionaries such as `Q[k]`, retaining one-based cell indices. NumPy array indices are zero-based.
- MATLAB's `lambda` is named `lam`, since `lambda` is a Python keyword.
- Symmetric variables, positive semidefinite constraints, interpolation conditions, and initial and final potentials follow the original formulation.

Two upstream errors were corrected where necessary for execution:

- The `fzk1` initialization-target typo in `2_Stochastic_BoundedVariance/B_StochasticGradientDescentWithAveraging.m`.
- The declaration order in `5_Stochastic_VarianceAtOptimum/A_ParameterSelection.m`, where the `relax=2` branch references `mu_X` before declaring it.

## Validation

As of 2026-10-21, all 11 standalone experiment scripts have been checked against the upstream model construction at commit [`78a089032dab4f4eceee6178531f55ba2cbc253d`](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods/tree/78a089032dab4f4eceee6178531f55ba2cbc253d).

The original `.m` model construction, before the solver call, was executed using **GNU Octave 11.1.0 and YALMIP 20250626**. Its extracted objectives, variables, and cone constraints were compared with the Python implementation. After aligning variable coordinates and equality signs, the checks compared the constant terms and all independent-variable coefficients of the PSD, equality, and nonnegativity constraints. No LMI reductions or additional constraints were introduced for this comparison.

| Check | Result |
| --- | --- |
| Experiments covered | All 11 scripts |
| Configurations | All 68 passed, covering each upstream `relax` branch |
| Cone coefficient-matrix entries compared | 6,362,598, including zeros |
| Maximum absolute difference | `2.7755575615628914e-17` |
| Comparison tolerance | `2e-12` absolute and relative |
| Script integrity | SHA-256 hashes of all 11 experiments match the validated files |
| Included data | All 11 `.dat` files are byte-for-byte identical to upstream |
| Pre-publication checks | All 16 Python files compile; dependency imports and `pip check` pass |
| Pre-publication numerical check | GD with `N=2` and CLARABEL: `optimal`, objective `8.9999999807`; figure creation and `(3, 5)` data output verified |
| Native MATLAB check | MATLAB R2026b executed; MOSEK 11.2.6 returned error `1008` (`MSK_RES_ERR_MISSING_LICENSE_FILE`) and produced no solution |

The two base parameter sets were `N=2, L=1, m=0` and `N=3, L=2, m=0.25`. Stochastic models used `n=2,3`, and bounded-variance models used `alpha=0,0.7`, respectively.

The checks confirm model agreement for the tested configurations. **No successful MATLAB/MOSEK numerical run is reported here.** Individual potential coefficients can vary with nonunique optima or different solver tolerances.

The included `.dat` files are upstream reference outputs, not results recomputed in Python. For variance-at-optimum with `N=5, L=1, m=0, n=2, relax=3, dN=4.5`, a separate lower-bound argument from the upstream constraints and a matching rational feasible solution establish an optimum of `20.25`. The upstream saved value of approximately `20.05` cannot be used as an exact feasible certificate for that configuration. Its full historical generation settings and solver logs are unavailable, so the cause of the discrepancy has not been determined.

Validation code, reference copies, raw reports, and downloaded tools are kept locally in `.validation/` and excluded from Git and GitHub. This README records the validation method, scope, and results.

## References

Adrien Taylor and Francis Bach, **Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions**, COLT 2019, Proceedings of Machine Learning Research 99:2934–2992.

- [Paper](https://proceedings.mlr.press/v99/taylor19a.html)
- [Original MATLAB implementation by the authors](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods)

This repository is a Python translation of the implementation above.
