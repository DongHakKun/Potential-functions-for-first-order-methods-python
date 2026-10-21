from pathlib import Path
from time import perf_counter

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from saveData import saveData, solve

t_start = perf_counter()
# SOLVER OPTIONS
verbose     = 1
tolerance   = 1e-8
solver = "MOSEK"  # Same solver as the MATLAB original; requires a MOSEK license.

# OUTPUT OPTIONS
ssave   = 1  # Save the results ?
pplot   = 1  # Plot the results ?
folder = 'SaveData/'  # If results saved, name of the saving folder
nname  = 'FastGradientD2.dat'  # Name of the file

# MINIMIZATION PROBLEM SETUP
L = 1  # Smoothness constant
m = 0  # Strong convexity constant
N = 100  # Number of iterations

# INTERMEDIARY POTENTIAL SETUPS (aims at reproducing results from Appendix
# C.3; Figure 4) via option "relaxation".
# Options:
#   Set relax = 0: use the base potential stated in Section 3.2.1.
#   Set relax = 1: force ak = L/2; Qk = 0
#   Set relax = 2: force ak = L/2, Qk(1,1)=Qk(1,2)=0, Qk(2,2)=-dk/2/L

relax = 0

# INITIAL AND FINAL POTENTIALS SETUP:
# Potential has the form:
#   Q(1,1) ||x_k-x*||^2 + Q(2,2) ||F'(x_k)||^2 + 2 Q(1,2) <F'(x_k); x_k-x*>
#       + ak ||z_k-x*||^2 + dk (F(x_k)-f(x*)).

Q0 = np.zeros((2, 2))
a0 = L/2
d0 = 0

tau = cp.Variable(name="tau")  # variable to be maximized (see problem (12) in the paper)
QN  = np.zeros((2, 2))
aN  = 0
dN  = tau

if relax == 1:
    a0 = L/2; aN = L/2
    Q0 = np.zeros((2, 2)); QN = np.zeros((2, 2))
elif relax == 2:
    a0 = L/2; aN = L/2
    Q0 = cp.bmat([[0, 0], [0, -d0/2/L]]); QN = cp.bmat([[0, 0], [0, -dN/2/L]])

################################################################
#                                                              #
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES  #
#                   (end of editable zone)                     #
#                                                              #
################################################################
# Minimization of f(x) with f smooth (possibly strongly) convex.
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0.

# P = [ z_k  y_k  y_{k+1} f'(y_k) f'(y_{k+1}) ]
# F = [                   f(y_k)  f(y_{k+1})   ]
dimG = 5  # dimensions of the Gram matrix P.T@P
dimF = 2  # dimensions of F
nbPts = 3  # x*, y_k and y_{k+1} in the discrete function f

# Notations for the LMI. Keep MATLAB's row-vector shapes explicitly.
zk = np.zeros((1, dimG)); zk[0, 0] = 1    # this is z_k
yk = np.zeros((1, dimG)); yk[0, 1] = 1    # this is y_k
yk1 = np.zeros((1, dimG)); yk1[0, 2] = 1  # this is y_{k+1}
xs = np.zeros((1, dimG))                  # this is x*

gyk = np.zeros((1, dimG)); gyk[0, 3] = 1    # this is f'(y_k)
gyk1 = np.zeros((1, dimG)); gyk1[0, 4] = 1  # this is f'(y_{k+1})
gxs = np.zeros((1, dimG))                   # this is f'(x*)

fyk = np.zeros((1, dimF)); fyk[0, 0] = 1    # this is f(y_k)
fyk1 = np.zeros((1, dimF)); fyk1[0, 1] = 1  # this is f(y_{k+1})
fxs = np.zeros((1, dimF))                   # this is f(x*)

# POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.)
X = {1: xs, 2: yk, 3: yk1}  # coordinates
G = {1: gxs, 2: gyk, 3: gyk1}  # gradients
F = {1: fxs, 2: fyk, 3: fyk1}  # function values

# line-search for y_{k+1}
A = {}
A[1] = gyk1.T @ (yk-yk1); A[1] = 1/2 * (A[1]+A[1].T)  # <f'(y_{k+1}); y_k-y_{k+1}> == 0
A[2] = gyk1.T @ (zk-yk); A[2] = 1/2 * (A[2]+A[2].T)    # <f'(y_{k+1}); z_k-y_k> == 0
A[3] = gyk1.T @ gyk; A[3] = 1/2 * (A[3]+A[3].T)        # <f'(y_{k+1}); f'(y_k)> == 0

# Matrix encoding interpolation condition for smooth strongly convex functions
M = 1/2/(L-m) * np.array([
    [-L*m,  L*m,  m, -L],
    [ L*m, -L*m, -m,  L],
    [    m,   -m, -1,  1],
    [   -L,    L,  1, -1],
])

# Notations for potentials
statesKG = np.vstack((yk-xs, gyk)); statesKF = fyk-fxs
statesK1G = np.vstack((yk1-xs, gyk1)); statesK1F = fyk1-fxs

################################################################
#                                                              #
#          SETTING UP THE LINEAR MATRIX INEQUALITIES           #
#     (directly embedded within the larger problem (12))       #
#                                                              #
################################################################
# Dictionaries retain MATLAB's one-based cell indices Q[1],...,Q[N+1].
# NumPy matrix coordinates use zero-based indices: (i,j) becomes [i-1,j-1].
Q = {1: Q0, N+1: QN}
d = {1: d0, N+1: dN}
a = {1: a0, N+1: aN}
lam = {}  # "lambda" is a Python keyword.
mu = {}
S = {}
cons_SDP = {}
cons_LIN = {}

cons = [tau >= 0]
for k in range(1, N+1):  # iteration counter (one LMI per iteration)
    if k < N:
        if not relax:  # no constraint on Qk, dk and ak
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            a[k+1] = cp.Variable(name=f"a_{k+1}")
        elif relax == 1:  # force ak = L/2; Qk = 0
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            a[k+1] = L/2
            Q[k+1] = np.zeros((2, 2))
        elif relax == 2:  # force ak = L/2, Qk(1,1)=Qk(1,2)=0, Qk(2,2)=-dk/2/L
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            a[k+1] = L/2
            Q[k+1] = cp.bmat([[0, 0], [0, -d[k+1]/2/L]])

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    mu[k] = cp.Variable((3, 1), name=f"mu_{k}")  # multipliers for line-search conditions
    S[k] = cp.Variable((4, 4), symmetric=True, name=f"S_{k}")  # this is Sk (see Section C.5)
    cons.append(S[k] >> 0)
    cons.append(S[k][0, 0] == a[k+1])

    # Form the SDP constraint
    cons_SDP[k] = (-a[k] * ((zk-xs).T @ (zk-xs))
                   - statesKG.T @ Q[k] @ statesKG
                   + np.vstack((yk1, zk-yk1, gyk, gyk1)).T @ S[k] @ np.vstack((yk1, zk-yk1, gyk, gyk1))
                   + statesK1G.T @ Q[k+1] @ statesK1G)
    for i in range(1, 4):
        cons_SDP[k] = cons_SDP[k] + mu[k][i-1, 0] * A[i]
    cons_LIN[k] = -d[k] * statesKF + d[k+1] * statesK1F

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
    Q11 = np.zeros(N+1)
    Q12 = np.zeros(N+1)
    Q22 = np.zeros(N+1)
    ak = np.zeros(N+1)
    dk = np.zeros(N+1)
    alphak = np.zeros(N+1)
    tauk = np.zeros(N+1)
    deltak = np.zeros(N+1)
    gammak = np.zeros(N+1)
    gammapk = np.zeros(N+1)  # gamma_k prime

    tolp = 1e-5
    for i in range(1, N+2):
        Q11[i-1] = float(Q[i][0, 0].value if isinstance(Q[i][0, 0], cp.Expression) else Q[i][0, 0])
        Q12[i-1] = float(Q[i][0, 1].value if isinstance(Q[i][0, 1], cp.Expression) else Q[i][0, 1])
        Q22[i-1] = float(Q[i][1, 1].value if isinstance(Q[i][1, 1], cp.Expression) else Q[i][1, 1])
        ak[i-1] = float(a[i].value if isinstance(a[i], cp.Expression) else a[i])
        dk[i-1] = float(d[i].value if isinstance(d[i], cp.Expression) else d[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, N+1):
            tauk[i-1] = np.divide(mu[i][1, 0].value, mu[i][0, 0].value)
            alphak[i-1] = -np.divide(mu[i][2, 0].value, mu[i][0, 0].value)
            gammak[i-1] = -np.divide(S[i][0, 2].value, S[i][0, 0].value)
            deltak[i-1] = np.divide(S[i][0, 1].value, S[i][0, 0].value)
            gammapk[i-1] = -np.divide(S[i][0, 3].value, S[i][0, 0].value)

    plt.subplot(4, 3, 1)
    plt.plot(np.arange(1, N+2), Q11, "-b"); plt.title("Q11 (coefficient of ||xk-x*||^2)")
    plt.subplot(4, 3, 2)
    plt.plot(np.arange(1, N+2), Q12, "-b"); plt.title("Q12 (coefficient of <f'(xk);xk-x*>)")
    plt.subplot(4, 3, 3)
    plt.plot(np.arange(1, N+2), Q22, "-b"); plt.title("Q22 (coefficient of ||f'(xk)||^2)")
    plt.subplot(4, 3, 4)
    plt.plot(np.arange(1, N+2), ak, "-b"); plt.title("ak (coefficient of ||zk-x*||^2)")
    plt.subplot(4, 3, 5)
    plt.plot(np.arange(1, N+2), dk, "-b"); plt.title("dk (coefficient of f(xk)-f(x*))")
    plt.subplot(4, 3, 6)
    plt.plot(np.arange(1, N+1), tauk[:-1], "-b"); plt.title("tauk")
    plt.subplot(4, 3, 7)
    plt.plot(np.arange(1, N+1), alphak[:-1], "-b"); plt.title("alpha")
    plt.subplot(4, 3, 8)
    plt.plot(np.arange(1, N+1), deltak[:-1], "-b"); plt.title("deltak")
    plt.subplot(4, 3, 9)
    plt.plot(np.arange(1, N+1), gammak[:-1], "-b"); plt.title("gammak")
    plt.subplot(4, 3, 10)
    plt.plot(np.arange(1, N+1), gammapk[:-1], "-b"); plt.title("gamma prime")

    if ssave:
        labels = ["k", "Q11", "Q12", "Q22", "ak", "dk", "alphak", "tauk", "deltak", "gammak", "gammapk"]
        data = np.column_stack((np.arange(1, N+2), Q11, Q12, Q22, ak, dk, alphak, tauk, deltak, gammak, gammapk))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
