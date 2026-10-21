# Potential functions for first-order methods — Python

This repository is a file-by-file Python translation of the [original MATLAB code](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods) by Adrien Taylor and Francis Bach. It uses NumPy and CVXPY to formulate the semidefinite programs for finding potential functions, and Matplotlib to plot the results.

**Status: incomplete numerical reproduction.** Of the 11 CLARABEL runs reported below, four returned `optimal`, six returned `optimal_inaccurate`, and one failed. The model translation is implemented, but reliable numerical reproduction of every experiment remains unfinished.

**MOSEK could not be used in the validation environment, so numerical experiments were run with the free solver CLARABEL.** The tables below show the measured results and their differences from the original repository's saved `.dat` outputs.

The translation follows the original directories, filenames, settings sections, variable names, calculation order, linear matrix inequalities (LMIs), plots, and data format. Each experiment contains its implementation directly, from setup through optimization and saving. Edit the settings at the top of a script and run it.

## Installation and usage

Verified with Python **3.12.13** in an existing conda environment named `py312`. If you have that environment, run:

```bash
conda activate py312
python -m pip install -r requirements.txt
```

`requirements.txt` contains this project's libraries and solver interfaces. The scripts retain the upstream MOSEK default. For the CLARABEL runs documented here, set the solver at the top of the script:

```python
solver = "CLARABEL"
```

Then run the experiment:

```bash
python 1_Deterministic_SmoothConvex/A_GradientDescent.py
```

Edit `L`, `m`, `N`, `delta`, `relax`, the initial and final potentials, `pplot`, and `ssave` in each script, as applicable. Data is saved in the `SaveData/` directory beside the script. Change `nname` to preserve an existing result; saving to the same path overwrites that file.

## Results compared with the original repository

The reference values link to the original repository's saved `.dat` outputs. The CLARABEL values are new runs of the Python scripts; the reference column is not a new MOSEK measurement.

Runs used Python 3.12.13, CVXPY 1.9.3, CLARABEL 0.11.1, `L=1`, `m=0`, stochastic `n=2`, bounded-variance `alpha=0`, and weak-growth `rho=2`. `N` and `relax` are listed for each experiment. Plotting and saving were disabled, with numerical-library threads limited to one. Some settings override script defaults to follow the saved coefficient patterns.

Absolute difference is `|CLARABEL - saved|`; signed difference is `100 * (CLARABEL - saved) / |saved|`. Differences are calculated before rounding the displayed values.

| Experiment (N, relax) | Quantity | Upstream saved | CLARABEL | Absolute difference | Signed difference % | Status |
|---|---|---:|---:|---:|---:|---|
| Gradient descent (100, 0) | τ (= final b_N) | [10201](1_Deterministic_SmoothConvex/SaveData/GradientD.dat) | 10200.986092 | 0.01390823 | -0.0001363418% | optimal |
| Proximal gradient descent (100, 0) | τ (= final b_N) | [9999.9](1_Deterministic_SmoothConvex/SaveData/ProxGradientD.dat) | 9963.048008 | 36.85199 | -0.3685236% | optimal_inaccurate |
| First accelerated method (100, 2) | τ (= final d_N) | [4089.5](1_Deterministic_SmoothConvex/SaveData/FastGradientD1.dat) | 4086.6844548 | 2.815545 | -0.06884815% | optimal_inaccurate |
| Second accelerated method (100, 0) | τ (= final d_N) | [5270.4](1_Deterministic_SmoothConvex/SaveData/FastGradientD2.dat) | 5270.3919349 | 0.008065149 | -0.0001530273% | optimal |
| Bounded-variance SGD (100, 0) | Final τ after stage 2 | [201](2_Stochastic_BoundedVariance/SaveData/SGD.dat) | 200.99997971 | 2.028976e-05 | -1.009441e-05% | optimal |
| Bounded-variance SGD (100, 0) | Σ e_k | [2057.5](2_Stochastic_BoundedVariance/SaveData/SGD.dat) | 2060.4544691 | 2.954469 | +0.1435951% | optimal |
| SGD with averaging (100, 0) | Final τ after stage 2 | [102](2_Stochastic_BoundedVariance/SaveData/SGD_Averaging.dat) | 101.99992868 | 7.132486e-05 | -6.992633e-05% | optimal_inaccurate |
| SGD with averaging (100, 0) | Σ e_k | [48.676](2_Stochastic_BoundedVariance/SaveData/SGD_Averaging.dat) | 48.661429105 | 0.01457089 | -0.02993445% | optimal_inaccurate |
| SGD with primal averaging (100, 0) | Σ e_k | [49.5](2_Stochastic_BoundedVariance/SaveData/SGD_Primal_Averaging.dat) | 49.500408784 | 0.0004087844 | +0.000825827% | optimal_inaccurate |
| SGD at averaged point (100, 0) | Final τ after stage 2 | [100](2_Stochastic_BoundedVariance/SaveData/SGD_Evaluation_AtAveragedPoint.dat) | 99.997368944 | 0.002631056 | -0.002631056% | optimal_inaccurate |
| SGD at averaged point (100, 0) | Σ e_k | [49.5](2_Stochastic_BoundedVariance/SaveData/SGD_Evaluation_AtAveragedPoint.dat) | 49.498725789 | 0.001274211 | -0.002574165% | optimal_inaccurate |
| Overparametrized SGD (100, 3) [a] | τ (= final d_N) | [100.16](3_Stochastic_Overparametrized/SaveData/ParameterSelection.dat) | 100.17922593 | 0.01922593 | +0.01919522% | optimal_inaccurate |
| Weak-growth primal averaging (100, 1) | Σ e_k (certificate) [b] | [3.1255](4_Stochastic_WeakGrowth/SaveData/SGD_Primal_Averaging.dat) | 3.1250295307 | 0.0004704693 | -0.01505261% | optimal |
| Variance at optimum (5, 3) | Σ e_k | [20.05](5_Stochastic_VarianceAtOptimum/SaveData/ParameterSelection.dat) | — | — | — | failed (NumericalError) |

For two-stage SGD experiments, both the final `tau` and the minimized noise coefficient `sum(e)` are shown. Each run follows the original reoptimization procedure with `tol_reopt=1e-5`.

[a] The overparametrized `relax=3` setting is inferred from the saved coefficients; `relax=1` cannot be excluded. Full historical settings are unavailable, so this row is a comparison for the stated configuration.

[b] Weak growth solves a feasibility problem with objective zero. Its `sum(e)` is a certificate coefficient, not an optimized objective.

The saved files use five significant digits. The GD and second accelerated-method differences are below the reference values' rounding resolution. Differences from historical outputs can include rounding and configuration differences as well as numerical effects; they do not isolate solver error.

### CLARABEL constraint residuals and timing

| Experiment | Max raw constraint violation | Solver time (s) |
| --- | ---: | ---: |
| 1/A — Gradient descent | `1.761e-08` | 0.022558 |
| 1/B — Proximal gradient | `6.893e-05` | 0.134143 |
| 1/C — First accelerated method | `3.787e-05` | 0.219248 |
| 1/D — Second accelerated method | `5.352e-09` | 0.068079 |
| 2/A — SGD | `2.039e-08` | 0.164970 |
| 2/B — SGD with averaging | `4.483e-05` | 0.281515 |
| 2/C — SGD with primal averaging | `7.363e-07` | 0.040916 |
| 2/D — Evaluation at averaged point | `3.834e-07` | 0.076993 |
| 3/A — Overparametrized | `1.029e-05` | 0.572055 |
| 4/A — Weak growth, primal averaging | `1.597e-10` | 0.037386 |
| 5/A — Variance at optimum | — | 0.031384 |

Times exclude compilation and plotting, and sum both solves for two-stage experiments. Constraint violations are measured on the original raw constraints at the final returned solution. The failed variance-at-optimum run returned no solution, so its output and difference are left blank above.

`optimal_inaccurate` results should be read with their residuals: for example, SGD with averaging returned a lower `sum(e)` than the saved value but violated an original constraint by `4.48e-5`; the lower value is not evidence of improved accuracy.

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
| Failed-model audit | Variance-at-optimum at `N=5` and `N=10`: all original cone coefficients match exactly; rational feasible witnesses satisfy the full constraints. This is not a successful numerical solve. |

The two base parameter sets were `N=2, L=1, m=0` and `N=3, L=2, m=0.25`. Stochastic models used `n=2,3`, and bounded-variance models used `alpha=0,0.7`, respectively.

These coefficient checks establish model agreement for the tested configurations; they do not measure numerical differences between solvers. Individual potential coefficients can vary with nonunique optima or different solver tolerances.

The included `.dat` files are upstream reference outputs, not results recomputed in Python. For variance-at-optimum with `N=5, L=1, m=0, n=2, relax=3, dN=4.5`, a separate lower-bound argument from the upstream constraints and a matching rational feasible solution establish an optimum of `20.25`. The upstream saved value of approximately `20.05` cannot be used as an exact feasible certificate for that configuration. Its full historical generation settings and solver logs are unavailable, so the cause of the discrepancy has not been determined.

Validation code, reference copies, raw reports, and downloaded tools are kept locally in `.validation/` and excluded from Git and GitHub. This README records the validation method, scope, and results.

## References

Adrien Taylor and Francis Bach, **Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions**, COLT 2019, Proceedings of Machine Learning Research 99:2934–2992.

- [Paper](https://proceedings.mlr.press/v99/taylor19a.html)
- [Original MATLAB implementation by the authors](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods)

This repository is a Python translation of the implementation above.
