# Potential functions for first-order methods — Python

This repository is a file-by-file Python translation of the [original MATLAB code](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods) by Adrien Taylor and Francis Bach. It uses NumPy and CVXPY to formulate the semidefinite programs for finding potential functions, and Matplotlib to plot the results.

The translation follows the original directories, filenames, settings sections, variable names, calculation order, linear matrix inequalities (LMIs), plots, and data format. Each experiment contains its implementation directly, from setup through optimization and saving. Edit the settings at the top of a script and run it.

## Installation and usage

Verified with Python **3.12.13** in an existing conda environment named `py312`. If you have that environment, run:

```bash
conda activate py312
python -m pip install -r requirements.txt
python 1_Deterministic_SmoothConvex/A_GradientDescent.py
```

`requirements.txt` pins the verified versions of NumPy, CVXPY, Matplotlib, MOSEK, and CLARABEL. It lists this project's dependencies rather than exporting the existing conda environment. PyTorch is not used or required. pip installs transitive dependencies as needed.

The default solver is **MOSEK**, matching the original scripts. Installing its Python package does not supply a license: configure a valid [MOSEK license](https://docs.mosek.com/latest/pythonapi/install-interface.html) on the machine that runs the optimization. No solver license files are included in this repository.

To use the free solver CLARABEL, change the setting at the top of the experiment:

```python
solver = "CLARABEL"
```

Some degenerate SDPs can produce accuracy warnings or numerical errors with CLARABEL. The scripts retain the original MOSEK default and LMI formulation; select an alternative solver explicitly using the setting above.

Edit `L`, `m`, `N`, `delta`, `relax`, the initial and final potentials, `pplot`, and `ssave` in each script, as applicable. Data is saved in the `SaveData/` directory beside the script. Change `nname` to preserve an existing result; saving to the same path overwrites that file.

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

The two base parameter sets were `N=2, L=1, m=0` and `N=3, L=2, m=0.25`. Stochastic models used `n=2,3`, and bounded-variance models used `alpha=0,0.7`, respectively.

These results establish agreement of the model construction for the checked configurations. They do not establish complete numerical reproduction of every experiment, plot, or output file under MATLAB and MOSEK. The MOSEK SDK was installed, but numerical execution with MOSEK could not be verified without a valid license. Individual potential coefficients may differ when the optimum is nonunique or the solver and tolerances differ.

The included `.dat` files are upstream reference outputs, not results recomputed in Python. For variance-at-optimum with `N=5, L=1, m=0, n=2, relax=3, dN=4.5`, a separate lower-bound argument from the upstream constraints and a matching rational feasible solution establish an optimum of `20.25`. The upstream saved value of approximately `20.05` cannot be used as an exact feasible certificate for that configuration. Its full historical generation settings and solver logs are unavailable, so the cause of the discrepancy has not been determined.

Validation code, reference copies, raw reports, and downloaded tools are kept locally in `.validation/` and excluded from Git and GitHub. This README records the validation method, scope, and results.

## References

Adrien Taylor and Francis Bach, **Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions**, COLT 2019, Proceedings of Machine Learning Research 99:2934–2992.

- [Paper](https://proceedings.mlr.press/v99/taylor19a.html)
- [Original MATLAB implementation by the authors](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods)

This repository is a Python translation of the implementation above.
