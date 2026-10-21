from pathlib import Path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from saveData import saveData
import time

tic = time.perf_counter()
# SOLVER OPTIONS
verbose = 1
tolerance = 1e-8
solver = "MOSEK"  # Original MATLAB solver; requires an available MOSEK license.

# OUTPUT OPTIONS
reoptimize = 1  # Set to 1 for the two-step optimization procedure (see (3) in the paper).
tol_reopt = 1e-5  # Tolerance in the re-optimization process.
ssave = 1  # Save the results?
pplot = 1  # Plot the results?
folder = "SaveData/"  # If results saved, name of the saving folder.
nname = "SGD.dat"  # Name of the file.

# MINIMIZATION PROBLEM SETUP
L = 1  # Smoothness constant.
m = 0  # Strong convexity constant.
n = 2  # Cardinality of the support for the stochastic gradient.
N = 100  # Number of iterations.

# ALGORITHM SETUP: stochastic gradient descent with step-size policy delta_k.
alpha = 0
delta = lambda k: 1 / L / (k + 1) ** alpha  # Step-size (possibly varying function of k).

# POTENTIAL SETUP
# Potential has the form:
#   ak ||x_k-x*||^2 + bk ||f'(x_k)||^2 + 2 ck <f'(x_k); x_k-x*>
#       + dk (f(x_k)-f(x*)).
# INTERMEDIARY POTENTIAL SETUPS (aims at reproducing results from
# Section 3.2.2) via option "relaxation".
# Options:
#   Set relax = 0: use the base potential stated in Section 3.2.2.
#   Set relax = 1: force ck = 0.
#   Set relax = 2: force bk = ck = 0.
#   Set relax = 3: force bk = ck = 0, and ak = L/2.
relax = 0

# INITIAL AND FINAL POTENTIALS SETUP:
a0 = L / 2
b0 = 0
c0 = 0
d0 = 0
tau = cp.Variable(name="tau")  # Variable to be maximized (see problem (3) in the paper).

aN = 0
bN = 0
cN = 0
dN = tau

if relax == 1:
    c0 = 0
    cN = 0
elif relax == 2:
    b0 = 0
    bN = 0
    c0 = 0
    cN = 0
elif relax == 3:
    a0 = L / 2
    aN = L / 2
    b0 = 0
    bN = 0
    c0 = 0
    cN = 0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                              %
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES  %
#                   (end of editable zone)                     %
#                                                              %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0.
# P = [ xk  G(x_k;1) ... G(x_k;n) f'(x_{k+1}^{(1)}) ... f'(x_{k+1}^{(n)})]
# F = [           f(x_k)          f(x_{k+1})  ...       f(x_{k+1}^{(n)})   ]
# NOTE: Symmetry arguments allow simplifying both F and G=P.'*P.
dimG = 1 + 2 * n  # Dimensions of the Gram matrix P.T @ P.
dimF = n + 1  # Dimensions of F.
nbPts = 2 + n  # x*, x_k, x_{k+1}^{(1)}, ..., x_{k+1}^{(n)}.

# Notations for the LMI.
xk = np.zeros(dimG)
xk[0] = 1  # x_k.
xs = np.zeros(dimG)  # x* = 0.
Gxk = np.zeros((n, dimG))
Gxk[:, 1:n+1] = np.eye(n)  # Gxk[i] is G(xk; i).
gxk = np.sum(Gxk, axis=0) / n  # f'(x_k).
gxk1 = np.zeros((n, dimG))
gxk1[:, 1+n:1+2*n] = np.eye(n)  # f'(x_{k+1}^{(i)}).
gxs = np.zeros(dimG)  # f'(x*) = 0.

fxk = np.zeros(dimF)
fxk[0] = 1  # f(x_k).
fxk1 = np.zeros((n, dimF))
fxk1[:, 1:n+1] = np.eye(n)  # f(x_{k+1}^{(i)}).
fxs = np.zeros(dimF)  # f(x*) = 0.

xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gxk

# Matrix encoding interpolation condition for smooth strongly convex functions.
M = np.array([
    [-L*m, L*m, m, -L],
    [L*m, -L*m, -m, L],
    [m, -m, -1, 1],
    [-L, L, 1, -1],
]) / (2 * (L - m))
# Matrix encoding the variance condition.
Avar = np.zeros((dimG, dimG))
for i in range(1, n + 1):
    Avar = Avar + np.outer(Gxk[i-1] - gxk, Gxk[i-1] - gxk) / n
# Notations for potentials.
Q0 = np.array([[a0, c0], [c0, b0]])
q0 = d0
QN = np.array([[aN, cN], [cN, bN]])
qN = dN
statesKG = np.column_stack((xk - xs, gxk))
statesKF = fxk

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                              %
#          SETTING UP THE LINEAR MATRIX INEQUALITIES           %
#     (directly embedded within the larger problem (2))        %
#                                                              %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dictionaries keep the original MATLAB cell indices: 1, ..., N+1.
Q = {1: Q0, N+1: QN}
d = {1: q0}
d[N+1] = qN
# lambda is a Python keyword; lam corresponds to MATLAB lambda{k}.
lam = {}
e = {}
cons_SDP = {}
cons_LIN = {}
cons = [tau >= 0]
for k in range(1, N + 1):  # One LMI per iteration.
    if k < N:
        if relax == 0:  # No constraint on Qk and dk.
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:  # Force ck = 0.
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.bmat([[cp.Variable(name=f"a_{k+1}"), 0], [0, cp.Variable(name=f"b_{k+1}")]])
        elif relax == 2:  # Force bk = ck = 0.
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.bmat([[cp.Variable(name=f"a_{k+1}"), 0], [0, 0]])
        elif relax == 3:  # Force bk = ck = 0 and ak = L/2.
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = np.array([[L/2, 0], [0, 0]])

    # POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.).
    XK1 = xk1(k)
    X = {1: xs, 2: xk}
    G = {1: gxs, 2: gxk}
    F = {1: fxs, 2: fxk}
    for i in range(1, n + 1):
        X[2+i] = XK1[i-1]
        G[2+i] = gxk1[i-1]
        F[2+i] = fxk1[i-1]

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    e[k] = cp.Variable(name=f"e_{k}")
    cons_SDP[k] = -statesKG @ Q[k] @ statesKG.T - e[k] * Avar
    cons_LIN[k] = -d[k] * statesKF

    for i in range(1, n + 1):  # Averaging the states over i.
        statesK1G = np.column_stack((XK1[i-1] - xs, gxk1[i-1]))
        statesK1F = fxk1[i-1]
        cons_SDP[k] = cons_SDP[k] + statesK1G @ Q[k+1] @ statesK1G.T / n
        cons_LIN[k] = cons_LIN[k] + statesK1F * d[k+1] / n

    for i in range(1, nbPts + 1):
        for j in range(1, nbPts + 1):
            if j != i:
                xi = X[i]
                xj = X[j]
                gi = G[i]
                gj = G[j]
                fi = F[i]
                fj = F[j]
                TT = np.vstack((xi, xj, gi, gj))
                cons_SDP[k] = cons_SDP[k] + lam[k][i-1, j-1] * (TT.T @ M @ TT)
                cons_LIN[k] = cons_LIN[k] + lam[k][i-1, j-1] * (fi - fj)

    cons = cons + [cons_SDP[k] << 0]
    cons = cons + [cons_LIN[k] == 0]
    cons = cons + [lam[k] >= 0]
    cons = cons + [e[k] >= 0]
accumulated_ek = 0
for k in range(1, N + 1):
    accumulated_ek = accumulated_ek + e[k]
obj = -tau

solver_opt = {"solver": solver, "verbose": bool(verbose)}
if solver == "CLARABEL":
    solver_opt.update(tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance)
elif solver == "SCS":
    solver_opt.update(eps=tolerance)
elif solver == "SDPA":
    solver_opt.update(epsilonStar=tolerance, epsilonDash=tolerance,
                      mpfPrecision=200, omegaStar=1e12, numThreads=1)
elif solver == "MOSEK":
    solver_opt["mosek_params"] = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance}
solverDetails = cp.Problem(cp.Minimize(obj), cons)
solverDetails.solve(**solver_opt)
if reoptimize:
    cons = cons + [tau >= -float(obj.value) - tol_reopt]
    solver_opt = {"solver": solver, "verbose": bool(verbose)}
    if solver == "SDPA":
        solver_opt.update(epsilonStar=tolerance, epsilonDash=tolerance,
                          mpfPrecision=200, omegaStar=1e12, numThreads=1)
    solverDetails = cp.Problem(cp.Minimize(accumulated_ek), cons)
    solverDetails.solve(**solver_opt)

print(time.perf_counter() - tic)

# Try to grasp what happens by plotting!
if pplot:
    plt.close("all")
    ak = np.zeros(N + 1)
    bk = np.zeros(N + 1)
    ck = np.zeros(N + 1)
    dk = np.zeros(N + 1)
    ek = np.zeros(N + 1)
    ekc = np.zeros(N + 1)
    for i in range(1, N + 2):
        ak[i-1] = float(Q[i][0, 0].value if isinstance(Q[i], cp.Expression) else Q[i][0, 0])
        bk[i-1] = float(Q[i][1, 1].value if isinstance(Q[i], cp.Expression) else Q[i][1, 1])
        ck[i-1] = float(Q[i][1, 0].value if isinstance(Q[i], cp.Expression) else Q[i][1, 0])
        dk[i-1] = float(d[i].value if isinstance(d[i], cp.Expression) else d[i])
    for i in range(2, N + 2):
        ek[i-1] = float(e[i-1].value)
        ekc[i-1] = np.sum(ek[:i])

    plt.subplot(3, 2, 1)
    plt.plot(np.arange(1, N + 2), ak, "-b")
    plt.title('ak (coefficient of ||xk-x*||^2)')
    plt.subplot(3, 2, 2)
    plt.plot(np.arange(1, N + 2), bk, "-b")
    plt.title("bk (coefficient of ||f'(xk)||^2)")
    plt.subplot(3, 2, 3)
    plt.plot(np.arange(1, N + 2), ck, "-b")
    plt.title("ck (coefficient of <f'(xk);xk-x*>)")
    plt.subplot(3, 2, 4)
    plt.plot(np.arange(1, N + 2), dk, "-b")
    plt.title('dk (coefficient of f(xk)-f(x*))')
    plt.subplot(3, 2, 5)
    plt.plot(np.arange(1, N + 2), ek, "-b")
    plt.title('ek (coefficient of sigma^2)')
    plt.subplot(3, 2, 6)
    plt.plot(np.arange(1, N + 2), ekc, "-b")
    plt.title('cumulated ek')

    if ssave:
        labels = ['k', 'ak', 'bk', 'ck', 'dk', 'ek', 'ekc']
        data = np.column_stack((np.arange(1, N + 2), ak, bk, ck, dk, ek, ekc))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
