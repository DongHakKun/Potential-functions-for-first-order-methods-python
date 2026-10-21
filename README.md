# Potential functions for first-order methods — Python

A file-by-file Python translation of the [MATLAB implementation](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods) by Adrien Taylor and Francis Bach. Each experiment contains its setup, semidefinite program, plots, and saving code directly. The original directories and 16-file MATLAB/Python correspondence are retained.

**Validation snapshot, 2026-10-21:** all 11 original MATLAB experiments were executed with MOSEK, and all 11 Python experiments were executed with MOSEK, CLARABEL, and SDPA Multiprecision. Both MOSEK implementations and SDPA returned `optimal` for every listed experiment. CLARABEL returned eight `optimal` and three `optimal_inaccurate` results. Numerical differences and original-constraint violations are reported below; identical historical outputs have not been established.

## Installation and usage

Verified in the existing conda environment `py312`, with Python 3.12.13:

```bash
conda activate py312
python -m pip install -r requirements.txt
python 1_Deterministic_SmoothConvex/A_GradientDescent.py
```

The scripts retain the original MOSEK default, which needs a MOSEK license. To use a free solver, edit the settings at the top of the script:

```python
solver = "CLARABEL"  # Fast double-precision solver; tolerance remains 1e-8.
```

For the more accurate free runs reported below:

```python
solver = "SDPA"      # Uses the installed sdpa-multiprecision backend.
tolerance = 1e-12
```

The SDPA branch uses 200-bit arithmetic, `epsilonStar=epsilonDash=tolerance`, `omegaStar=1e12`, and one thread. See the [SDPA installation and backend documentation](https://sdpa-python.github.io/docs/installation/). Inputs and exported coefficients remain Python/NumPy floating-point values; 200-bit solver arithmetic does not imply 200-bit output accuracy.

Edit `L`, `m`, `N`, `delta`, `relax`, the initial/final potentials, `pplot`, and `ssave` in each script, as applicable. Data is saved in the `SaveData/` directory beside the script. Set a different `nname` to preserve an existing file.

## Measured results

The MATLAB column is a fresh execution of the upstream `.m` files in **MATLAB R2026b, YALMIP 20250626, and MOSEK 11.2.6**. Python runs use **CVXPY 1.9.3**, **MOSEK 11.2.6**, **CLARABEL 0.11.1**, and **sdpa-multiprecision 0.2.3**. The upstream saved column contains the original, unchanged `.dat` outputs, rounded to five significant digits.

All runs use `L=1`, `m=0`, stochastic `n=2`, bounded-variance `alpha=0`, and weak-growth `rho=2`. The table specifies `N` and `relax`. MOSEK and CLARABEL use `tolerance=1e-8`; SDPA-MP uses `1e-12`. Plotting/saving were disabled for timings and measurements. Proximal gradient, first accelerated, overparametrized, and variance-at-optimum settings were chosen to follow the saved output patterns and can differ from current script defaults.

| Experiment (N, relax) | Quantity | Upstream saved | MATLAB + MOSEK | Python + MOSEK | Python + CLARABEL | Python + SDPA-MP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gradient descent (100, 0) | τ | [10201](1_Deterministic_SmoothConvex/SaveData/GradientD.dat) | 10200.949481 | 10200.949481 | 10200.986092 | 10201.000000 |
| Proximal gradient (100, 0) | τ | [9999.9](1_Deterministic_SmoothConvex/SaveData/ProxGradientD.dat) | 9999.628875 | 9999.714829 | 9999.983113 | 10000.000000 |
| First accelerated method (100, 2) | τ | [4089.5](1_Deterministic_SmoothConvex/SaveData/FastGradientD1.dat) | 4088.941623 | 4088.997507 | 4086.002196 | 4086.012189 |
| Second accelerated method (100, 0) | τ | [5270.4](1_Deterministic_SmoothConvex/SaveData/FastGradientD2.dat) | 5270.391539 | 5270.391549 | 5270.391935 | 5270.391539 |
| Bounded-variance SGD (100, 0) | Final τ | [201](2_Stochastic_BoundedVariance/SaveData/SGD.dat) | 200.999962 | 200.999951 | 200.999980 | 200.999990 |
| Bounded-variance SGD (100, 0) | Σe | [2057.5](2_Stochastic_BoundedVariance/SaveData/SGD.dat) | 2057.522911 | 2057.254076 | 2060.454469 | 2060.553609 |
| SGD with averaging (100, 0) | Final τ | [102](2_Stochastic_BoundedVariance/SaveData/SGD_Averaging.dat) | 101.999903 | 101.999901 | 101.999929 | 101.999990 |
| SGD with averaging (100, 0) | Σe | [48.676](2_Stochastic_BoundedVariance/SaveData/SGD_Averaging.dat) | 48.634944 | 48.631663 | 48.661429 | 49.324009 |
| SGD with primal averaging (100, 0) | Σe | [49.5](2_Stochastic_BoundedVariance/SaveData/SGD_Primal_Averaging.dat) | 49.500010 | 49.500013 | 49.500409 | 49.500000 |
| SGD at averaged point (100, 0) | Final τ | [100](2_Stochastic_BoundedVariance/SaveData/SGD_Evaluation_AtAveragedPoint.dat) | 99.999911 | 99.999962 | 99.997369 | 99.999990 |
| SGD at averaged point (100, 0) | Σe | [49.5](2_Stochastic_BoundedVariance/SaveData/SGD_Evaluation_AtAveragedPoint.dat) | 49.499973 | 49.500043 | 49.498726 | 49.499995 |
| Overparametrized SGD (100, 3) | τ | [100.16](3_Stochastic_Overparametrized/SaveData/ParameterSelection.dat) | 100.026848 | 100.052824 | 99.999243 | 100.000000 |
| Weak-growth primal averaging (100, 1) | Σe (certificate) | [3.1255](4_Stochastic_WeakGrowth/SaveData/SGD_Primal_Averaging.dat) | 3.125122 | 3.125122 | 3.125030 | 3.125002 |
| Variance at optimum (5, 3) | Σe | [20.05](5_Stochastic_VarianceAtOptimum/SaveData/ParameterSelection.dat) | 20.232957 | 20.224608 | 20.249994 | 20.250000 |

For two-stage SGD, each run follows the original procedure: maximize τ, then minimize Σe subject to `tau >= first_tau - 1e-5`. Each backend uses its own computed first τ, so the second-stage numerical feasible sets can differ. Final τ and Σe are both reported.

The overparametrized saved coefficients are consistent with `relax=3`; historical settings do not exclude `relax=1`. Weak growth minimizes zero: its Σe is a nonunique certificate coefficient. The variance saved file has five iterations, while the current script default has ten. Additional SDPA runs at the unchanged defaults returned variance `N=10`: `40.4999999933`, overparametrized `relax=0`: `116.2220917167`, proximal `relax=2`: `10099.9999999997`, and first accelerated `relax=0`: `4086.0121897908`.

The next table uses the newly executed MATLAB + MOSEK result as the baseline. Δ is `Python - MATLAB`; the signed percentage is `100 * Δ / abs(MATLAB)`.

| Experiment / quantity | Python MOSEK Δ (Δ%) | CLARABEL Δ (Δ%) | SDPA-MP Δ (Δ%) |
| --- | ---: | ---: | ---: |
| Gradient descent — τ | +0 (+0.000000%) | +0.036611 (+0.000359%) | +0.0505193 (+0.000495%) |
| Proximal gradient — τ | +0.0859535 (+0.000860%) | +0.354238 (+0.003543%) | +0.371125 (+0.003711%) |
| First accelerated method — τ | +0.0558835 (+0.001367%) | -2.93943 (-0.071887%) | -2.92943 (-0.071643%) |
| Second accelerated method — τ | +1.02388e-05 (+0.000000%) | +0.000395997 (+0.000008%) | -2.4084e-07 (-0.000000%) |
| Bounded-variance SGD — Final τ | -1.11201e-05 (-0.000006%) | +1.73046e-05 (+0.000009%) | +2.75975e-05 (+0.000014%) |
| Bounded-variance SGD — Σe | -0.268835 (-0.013066%) | +2.93156 (+0.142480%) | +3.0307 (+0.147298%) |
| SGD with averaging — Final τ | -2.3736e-06 (-0.000002%) | +2.56137e-05 (+0.000025%) | +8.69529e-05 (+0.000085%) |
| SGD with averaging — Σe | -0.00328078 (-0.006746%) | +0.0264849 (+0.054457%) | +0.689065 (+1.416810%) |
| SGD with primal averaging — Σe | +3.50181e-06 (+0.000007%) | +0.000398988 (+0.000806%) | -9.79673e-06 (-0.000020%) |
| SGD at averaged point — Final τ | +5.12351e-05 (+0.000051%) | -0.00254191 (-0.002542%) | +7.91467e-05 (+0.000079%) |
| SGD at averaged point — Σe | +6.96442e-05 (+0.000141%) | -0.00124723 (-0.002520%) | +2.19794e-05 (+0.000044%) |
| Overparametrized SGD — τ | +0.0259758 (+0.025969%) | -0.0276052 (-0.027598%) | -0.026848 (-0.026841%) |
| Weak-growth primal averaging — Σe (certificate) | +1.63177e-07 (+0.000005%) | -9.23087e-05 (-0.002954%) | -0.000120111 (-0.003843%) |
| Variance at optimum — Σe | -0.00834942 (-0.041266%) | +0.0170371 (+0.084205%) | +0.0170428 (+0.084233%) |

### Original-constraint checks

The following maximum absolute violations are measured at the final returned solution, including every original LMI, equality, nonnegativity condition, and the second-stage τ floor. Both MOSEK implementations and SDPA returned successful statuses in all rows; the CLARABEL status is shown explicitly.

| Experiment | MATLAB MOSEK | Python MOSEK | CLARABEL | SDPA-MP | CLARABEL status |
| --- | ---: | ---: | ---: | ---: | --- |
| Gradient descent | 1.178e-07 | 1.178e-07 | 1.761e-08 | 3.638e-12 | optimal |
| Proximal gradient | 5.651e-07 | 4.074e-07 | 2.881e-08 | 3.638e-12 | optimal |
| First accelerated method | 3.372e-07 | 3.512e-07 | 7.146e-10 | 2.010e-12 | optimal |
| Second accelerated method | 1.364e-12 | 3.721e-11 | 5.352e-09 | 1.033e-12 | optimal |
| Bounded-variance SGD | 3.649e-05 | 4.185e-05 | 2.039e-08 | 7.629e-14 | optimal |
| SGD with averaging | 8.780e-05 | 8.954e-05 | 4.483e-05 | 1.833e-13 | optimal_inaccurate |
| SGD with primal averaging | 5.702e-07 | 8.059e-07 | 7.363e-07 | 3.838e-13 | optimal_inaccurate |
| SGD at averaged point | 9.111e-07 | 3.541e-06 | 3.834e-07 | 3.553e-13 | optimal_inaccurate |
| Overparametrized SGD | 2.978e-08 | 8.264e-08 | 3.826e-09 | 7.422e-13 | optimal |
| Weak-growth primal averaging | 1.034e-08 | 1.037e-08 | 1.597e-10 | 3.286e-13 | optimal |
| Variance at optimum | 2.551e-09 | 5.034e-09 | 2.026e-09 | 1.375e-13 | optimal |

A returned `optimal` status alone does not guarantee small absolute violations. For example, original MATLAB MOSEK violates the final τ floor by `3.65e-5` in SGD and `8.78e-5` in SGD with averaging. The lower noise coefficients in those runs do not establish a better feasible optimum.

First-stage residuals are recorded separately. Noise coefficients are absent from the first-stage objective and can become large. In the SDPA averaging run, double-precision matrix assembly and eigenvalue evaluation reported a `2.69e-7` violation; evaluating the same returned variables and original affine coefficients at higher precision reduced the maximum violation below `9.4e-13`. The final second-stage double-precision violation is `1.83e-13`. First-stage τ errors also propagate through the reoptimization floor, so final objective digits need a stability check.

### Free solvers at the same MATLAB τ floor

This separate stage-two comparison fixes each floor to the actual MATLAB first-stage τ minus `1e-5`. It removes first-stage τ variation. CLARABEL uses its original stage-two defaults; SDPA-MP uses 200 bits and `epsilonStar=epsilonDash=1e-18`. These controlled runs are distinct from the independent full workflows above.

| Experiment | Common τ floor | Solver | Σe | Max original violation |
| --- | ---: | --- | ---: | ---: |
| 2/A | 200.999998900604 | MATLAB MOSEK | 2057.522911472 | 3.649e-05 |
| 2/A | 200.999998900604 | CLARABEL | 2060.684439851 | 1.655e-06 |
| 2/A | 200.999998900604 | SDPA | 2060.700527162 | 2.608e-14 |
| 2/B | 101.999990857506 | MATLAB MOSEK | 48.634944214 | 8.780e-05 |
| 2/B | 101.999990857506 | CLARABEL | 49.214432362 | 4.072e-06 |
| 2/B | 101.999990857506 | SDPA | 49.325722343 | 2.187e-14 |
| 2/D | 99.999910720443 | MATLAB MOSEK | 49.499973021 | 9.111e-07 |
| 2/D | 99.999910720443 | CLARABEL | 49.500005449 | 1.883e-07 |
| 2/D | 99.999910720443 | SDPA | 49.499955360 | 1.795e-14 |

For variance at optimum with `N=5, L=1, m=0, n=2, relax=3, dN=4.5`, a lower-bound argument and matching rational feasible solution establish an optimum of `20.25`. The SDPA run differs from it by about `1.7e-9`. The upstream saved `20.05` is not an exact feasible certificate for that configuration; its historical generation settings and solver logs are unavailable. The discrepancy has not been attributed solely to translation or to a particular solver.

The bound follows by summing the source-implied inequality `e_k >= 49.5 * (d_next - d_current) - 40.5`: `sum(e) >= 49.5 * 4.5 - 40.5 * 5 = 20.25`. A rational feasible construction with `d_k = 0.9*k` and `e_k = 4.05` attains it.

## Correspondence with MATLAB

- `sdpvar` becomes `cvxpy.Variable`, matrix multiplication uses `@`, and transposes use `.T`.
- MATLAB cells such as `Q{k}` become dictionaries such as `Q[k]`, retaining one-based cell indices. NumPy matrix coordinates are zero-based.
- MATLAB's `lambda` is named `lam`, since `lambda` is a Python keyword.
- The original potential setup, interpolation conditions, full LMI construction, plotting, and data format remain in each experiment.

The translation corrects an allocation-target typo for `fzk1` in bounded-variance averaging and the variance-at-optimum `relax=2` branch referencing `mu_X` before its declaration. MATLAB implicitly allocates `fzk1` later; Python requires its explicit initialization.

For CLARABEL and SDPA, proximal gradient, the first accelerated method, and the two parameter-selection scripts use an equivalent solver representation of singular PSD blocks. The original `cons` and variable objects are retained. Directions that the original constraints force to zero are expressed as equalities, and the remaining principal PSD blocks are submitted to the solver. Comments beside the solver call explain the equivalence and parameter guards. MOSEK receives the original full constraints. No stored solution or known optimum is inserted into a solve.

## Validation

The source reference is upstream commit [`78a089032dab4f4eceee6178531f55ba2cbc253d`](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods/tree/78a089032dab4f4eceee6178531f55ba2cbc253d).

| Check | Result |
| --- | --- |
| Original model construction | All 11 `.m` models executed using GNU Octave 11.1.0 and YALMIP 20250626, before optimization |
| Affine objective/variable/cone comparison | All 68 configurations passed, covering every original `relax` branch |
| Cone coefficient-matrix entries compared | 6,362,598, including zeros; maximum absolute difference `2.7755575615628914e-17` |
| Comparison tolerance | `2e-12` absolute and relative; original full PSD, equality, and nonnegativity constraints |
| Full-sized affine comparison | All 11 table settings: ten at `N=100` and variance at `N=5`; 244,049,687 cone affine entries including zeros, maximum difference exactly zero; objective and variable observations also agree exactly |
| Actual MATLAB/MOSEK execution | All 11 original experiments; all 14 optimization calls returned YALMIP code 0 |
| Actual Python/MOSEK execution | All 11 comparison experiments; all 14 optimization calls returned `optimal` |
| Native solution cross-evaluation | All 11 models and 14 stages: actual MATLAB solutions assigned to the Python variables; 58,720 full LMI/equality entries agree within `3.64e-12`; nonnegativity and τ-floor residuals also agree |
| Free high-precision execution | All 11 comparison experiments plus four unchanged-default runs returned `optimal` |
| SDPA runtime branch coverage | All 68 configurations ran successfully; all 84 optimization calls returned `optimal` |
| Final SDPA residuals | At most `3.64e-12` for the 15 full-sized runs and `1.88e-12` for the 68 branch configurations |
| Plot/save checks | Actual source blocks executed, including variance `N=5/10` and overparametrized `N=100`; finite data and figures verified |
| Included data | All 11 `.dat` files remain byte-for-byte identical to upstream |
| Packaging | All 16 Python files compile; dependency imports and `pip check` pass |

The 68 parameter sets combine `N=2, L=1, m=0` or `N=3, L=2, m=0.25` with every `relax` branch. Stochastic models use `n=2,3`; bounded-variance models use `alpha=0,0.7`. Coefficient checks establish agreement for those tested models. They do not imply identical numerical certificates or unique potential coefficients.

Validation code, raw measurements, reference copies, and tools are kept locally in `.validation/`, excluded from Git and GitHub. The public repository contains the direct implementations and this record of the verification scope and measured results.

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

## References

Adrien Taylor and Francis Bach, **Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions**, COLT 2019, Proceedings of Machine Learning Research 99:2934–2992.

- [Paper](https://proceedings.mlr.press/v99/taylor19a.html)
- [Original MATLAB implementation by the authors](https://github.com/AdrienTaylor/Potential-functions-for-first-order-methods)

This repository is a Python translation of the implementation above.
