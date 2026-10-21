from pathlib import Path
from time import perf_counter

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from saveData import saveData, solve

t_start = perf_counter()
# SOLVER OPTIONS
verbose = 1
tolerance = 1e-8
solver = "MOSEK"  # Same solver as the MATLAB original; requires a MOSEK license.

# OUTPUT OPTIONS
ssave = 1  # Save the results ?
pplot = 1  # Plot the results ?
folder = 'SaveData/'  # If results saved, name of the saving folder
nname = 'GradientD.dat'  # Name of the file

# MINIMIZATION PROBLEM SETUP
L = 1  # Smoothness constant
m = 0  # Strong convexity constant
N = 100  # Number of iterations

# ALGORITHM SETUP: gradient descent with step-size policy delta_k.
delta = lambda k: 1/L  # Step-size (possibly varying function of k)

# INTERMEDIARY POTENTIAL SETUPS (aims at reproducing results from
# Section 3.2.1) via option "relaxation".

# Options:
#   Set relax = 0: use the base potential stated in Section 3.2.1.
#   Set relax = 1: force dk = (2*k+1)*L
#   Set relax = 2: force dk = (2*k+1)*L, ck = 0 and ak = L**2.
#   Set relax = 3: force dk = 0.
relax = 0

# INITIAL AND FINAL POTENTIALS SETUP:
# Potential has the form:
#   ak ||x_k-x*||^2 + bk ||f'(x_k)||^2 + 2 ck <f'(x_k); x_k-x*>
#       + dk (f(x_k)-f(x*)).
a0 = L**2
b0 = 0
c0 = 0
d0 = 0

tau = cp.Variable(name="tau")  # variable to be maximized (see problem (2) in the paper)
aN = 0
bN = tau  # bN is to be maximized
cN = 0
dN = 0

if relax == 1:
    d0 = L
    dN = (2*N+1)*L
elif relax == 2:
    d0 = L
    dN = (2*N+1)*L
    aN = L**2
    cN = 0
elif relax == 3:
    d0 = 0
    dN = 0

################################################################
#                                                              #
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES  #
#                   (end of editable zone)                     #
#                                                              #
################################################################
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0.

# P = [ xk  f'(x_k)  f'(x_{k+1}) ]
# F = [     f(x_k)   f(x_{k+1})  ]
dimG = 3  # dimensions of the Gram matrix P.T@P
dimF = 2  # dimensions of F
nbPts = 3  # x*, x_k and x_{k+1} in the discrete function f

# Notations for the LMI. Keep MATLAB's row-vector shapes explicitly.
xk = np.zeros((1, dimG)); xk[0, 0] = 1  # this is x_k
xs = np.zeros((1, dimG))                # this is x* = 0

gxk = np.zeros((1, dimG)); gxk[0, 1] = 1    # this is f'(x_k)
gxk1 = np.zeros((1, dimG)); gxk1[0, 2] = 1  # this is f'(x_{k+1})
gxs = np.zeros((1, dimG))                   # this is f'(x*) = 0

fxk = np.zeros((1, dimF)); fxk[0, 0] = 1    # this is f(x_k)
fxk1 = np.zeros((1, dimF)); fxk1[0, 1] = 1  # this is f(x_{k+1})
fxs = np.zeros((1, dimF))                   # this is f(x*) = 0

xk1 = lambda k: xk - delta(k) * gxk  # this is x_{k+1}

# Matrix encoding interpolation condition for smooth strongly convex functions
M = 1/2/(L-m) * np.array([
    [-L*m,  L*m,  m, -L],
    [ L*m, -L*m, -m,  L],
    [    m,   -m, -1,  1],
    [   -L,    L,  1, -1],
])

# Notations for potentials
Q0 = cp.bmat([[a0, c0], [c0, b0]]); q0 = d0
QN = cp.bmat([[aN, cN], [cN, bN]]); qN = dN
statesKG = np.hstack(((xk-xs).T, gxk.T)); statesKF = fxk
statesK1G = lambda k: np.hstack(((xk1(k)-xs).T, gxk1.T)); statesK1F = fxk1

################################################################
#                                                              #
#          SETTING UP THE LINEAR MATRIX INEQUALITIES           #
#     (directly embedded within the larger problem (2))       #
#                                                              #
################################################################
# Dictionaries retain MATLAB's one-based cell indices Q[1],...,Q[N+1].
# NumPy matrix coordinates use zero-based indices: (i,j) becomes [i-1,j-1].
Q = {1: Q0, N+1: QN}
d = {1: q0, N+1: qN}
lam = {}  # "lambda" is a Python keyword.
cons_SDP = {}
cons_LIN = {}

cons = [tau >= 0]
for k in range(1, N+1):  # iteration counter (one LMI per iteration)
    if k < N:
        if not relax:  # no constraint on Qk and dk
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:  # force dk = (2*k+1)*L and no constraint on Qk
            d[k+1] = (2*k+1)*L
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 2:  # force dk = (2*k+1)*L, ck = 0 and ak = L**2
            d[k+1] = (2*k+1)*L
            Q[k+1] = cp.bmat([[L**2, 0], [0, cp.Variable(name=f"b_{k+1}")]])
        elif relax == 3:  # force dk = 0 and no constraint on Qk
            d[k+1] = 0
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    cons_SDP[k] = (-statesKG @ Q[k] @ statesKG.T
                   + statesK1G(k) @ Q[k+1] @ statesK1G(k).T)
    cons_LIN[k] = -d[k] * statesKF + d[k+1] * statesK1F

    # POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.)
    X = {1: xs, 2: xk, 3: xk1(k)}  # coordinates
    G = {1: gxs, 2: gxk, 3: gxk1}  # gradients
    F = {1: fxs, 2: fxk, 3: fxk1}  # function values

    # ADD ALL INTERPOLATION CONDITIONS TO THE LMI
    for i in range(1, nbPts+1):
        for j in range(1, nbPts+1):
            if j != i:
                xi = X[i]; xj = X[j]
                gi = G[i]; gj = G[j]
                fi = F[i]; fj = F[j]
                TT = np.vstack((xi, xj, gi, gj))
                cons_SDP[k] = cons_SDP[k] + lam[k][i-1, j-1] * (TT.T @ M @ TT)
                cons_LIN[k] = cons_LIN[k] + lam[k][i-1, j-1] * (fi-fj)
    cons.append(cons_SDP[k] << 0)
    cons.append(cons_LIN[k] == 0)
    cons.append(lam[k] >= 0)
obj = tau

solver_opt = {"solver": solver, "verbose": bool(verbose)}
if solver == "CLARABEL":
    solver_opt.update(tol_feas=tolerance, tol_gap_abs=tolerance,
                      tol_gap_rel=tolerance, max_iter=500)
elif solver == "SCS":
    solver_opt.update(eps=tolerance, max_iters=200000)
elif solver == "SDPA":
    solver_opt.update(epsilonStar=tolerance, epsilonDash=tolerance,
                      mpfPrecision=200, omegaStar=1e12, numThreads=1)
elif solver == "MOSEK":
    solver_opt["mosek_params"] = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance}
problem = cp.Problem(cp.Minimize(-obj), cons)
solverDetails = solve(problem, **solver_opt)
if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    raise RuntimeError(f"The SDP returned {problem.status}; no solution can be plotted or saved.")
print(f"status={problem.status}, objective={obj.value:.10g}")
print(f"Elapsed time: {perf_counter()-t_start:.3f} seconds")

# Try to grasp what happens by plotting !
if pplot:
    plt.close("all")
    ak = np.zeros(N+1)
    bk = np.zeros(N+1)
    ck = np.zeros(N+1)
    dk = np.zeros(N+1)
    for i in range(1, N+2):
        ak[i-1] = float(Q[i][0, 0].value if isinstance(Q[i][0, 0], cp.Expression) else Q[i][0, 0])
        bk[i-1] = float(Q[i][1, 1].value if isinstance(Q[i][1, 1], cp.Expression) else Q[i][1, 1])
        ck[i-1] = float(Q[i][1, 0].value if isinstance(Q[i][1, 0], cp.Expression) else Q[i][1, 0])
        dk[i-1] = float(d[i].value if isinstance(d[i], cp.Expression) else d[i])

    plt.subplot(2, 2, 1)
    plt.plot(np.arange(1, N+2), ak, "-b"); plt.title("ak (coefficient of ||xk-x*||^2)")
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1, N+2), bk, "-b"); plt.title("bk (coefficient of ||f'(xk)||^2)")
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(1, N+2), ck, "-b"); plt.title("ck (coefficient of <f'(xk);xk-x*>)")
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1, N+2), dk, "-b"); plt.title("dk (coefficient of f(xk)-f(x*))")

    if ssave:
        labels = ["k", "ak", "bk", "ck", "dk"]
        data = np.column_stack((np.arange(1, N+2), ak, bk, ck, dk))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
