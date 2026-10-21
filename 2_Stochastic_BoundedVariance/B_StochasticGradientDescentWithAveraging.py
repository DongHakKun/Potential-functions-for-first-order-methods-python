from pathlib import Path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from saveData import saveData, solve

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
nname = "SGD_Averaging.dat"  # Name of the file.

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
#   ak ||z_k-x*||^2 + bk ||f'(z_k)||^2 + 2 ck <f'(z_k); z_k-x*>
#       + dk (f(z_k)-f(x*)) + ak' || x_k-x* ||^2
# INTERMEDIARY POTENTIAL SETUPS (aims at reproducing results from
# Section 3.2.2) via option "relaxation".
# Options:
#   Set relax = 0: use the base potential stated below.
#   Set relax = 1: force ak = bk = ck = 0 and ak' = L/2
relax = 0

# NOTE: this is for the sake of the example; much more states could be
# incorporated in the potentials (e.g., more dependence on x_k's)

# INITIAL AND FINAL POTENTIALS SETUP:
a0 = 0
b0 = 0
c0 = 0
d0 = 0
ap0 = L / 2  # a0'
tau = cp.Variable(name="tau")  # Variable to be maximized (see problem (3) in the paper).

aN = 0
bN = 0
cN = 0
dN = tau
apN = 0  # aN'

if relax == 1:
    a0 = 0
    aN = 0
    b0 = 0
    bN = 0
    c0 = 0
    cN = 0
    ap0 = L / 2  # a0'
    apN = L / 2  # aN'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                              %
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES  %
#                   (end of editable zone)                     %
#                                                              %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0.
# P = [ xk  z_k f'(z_k) G(x_k;1) ... G(x_k;n) f'(z_{k+1}^{(1)}) ... f'(z_{k+1}^{(n)}) ]
# F = [         f(z_k)           f(x_k)       f(z_{k+1})  ...       f(z_{k+1}^{(n)})   ]
# NOTE: Symmetry arguments allow simplifying both F and G=P.'*P.
dimG = 3 + 2 * n  # Dimensions of the Gram matrix P.T @ P.
dimF = n + 2  # Dimensions of F.
nbPts = 3 + n  # x*, x_k, z_k, z_{k+1}^{(1)}, ..., z_{k+1}^{(n)}.

# Notations for the LMI.
xk = np.zeros(dimG)
xk[0] = 1  # x_k.
zk = np.zeros(dimG)
zk[1] = 1  # z_k.
xs = np.zeros(dimG)  # x* = 0.
gzk = np.zeros(dimG)
gzk[2] = 1  # f'(z_k).
Gxk = np.zeros((n, dimG))
Gxk[:, 3:n+3] = np.eye(n)  # G(xk; i).
gxk = np.sum(Gxk, axis=0) / n  # f'(x_k).
gzk1 = np.zeros((n, dimG))
gzk1[:, 3+n:3+2*n] = np.eye(n)  # f'(z_{k+1}^{(i)}).
gxs = np.zeros(dimG)  # f'(x*) = 0.

fzk = np.zeros(dimF)
fzk[0] = 1  # f(z_k).
fxk = np.zeros(dimF)
fxk[1] = 1  # f(x_k).
# The MATLAB allocation says fxk1, but the following assignment uses fzk1.
fzk1 = np.zeros((n, dimF))
fzk1[:, 2:n+2] = np.eye(n)  # f(z_{k+1}^{(i)}).
fxs = np.zeros(dimF)  # f(x*) = 0.

xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gxk
zk1 = lambda k: xk1(k) / k + (k - 1) / k * zk

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
statesKG = np.column_stack((zk - xs, gzk))
statesKF = fzk

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
ap = {1: ap0, N+1: apN}
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
            ap[k+1] = cp.Variable(name=f"ap_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:  # Force ak = bk = ck = 0 and ak' = L/2.
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            ap[k+1] = L / 2
            Q[k+1] = np.zeros((2, 2))

    # POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.).
    XK1 = xk1(k)
    ZK1 = zk1(k)
    X = {1: xs, 2: xk, 3: zk}
    G = {1: gxs, 2: gxk, 3: gzk}
    F = {1: fxs, 2: fxk, 3: fzk}
    for i in range(1, n + 1):
        X[3+i] = ZK1[i-1]
        G[3+i] = gzk1[i-1]
        F[3+i] = fzk1[i-1]

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    e[k] = cp.Variable(name=f"e_{k}")
    cons_SDP[k] = -statesKG @ Q[k] @ statesKG.T - ap[k] * np.outer(xk-xs, xk-xs) - e[k] * Avar
    cons_LIN[k] = -d[k] * statesKF

    for i in range(1, n + 1):  # Averaging the states over i.
        statesK1G = np.column_stack((ZK1[i-1] - xs, gzk1[i-1]))
        statesK1F = fzk1[i-1]
        cons_SDP[k] = cons_SDP[k] + statesK1G @ Q[k+1] @ statesK1G.T / n
        cons_SDP[k] = cons_SDP[k] + ap[k+1] * np.outer(XK1[i-1]-xs, XK1[i-1]-xs) / n
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
solve(solverDetails, **solver_opt)
if reoptimize:
    cons = cons + [tau >= -float(obj.value) - tol_reopt]
    solver_opt = {"solver": solver, "verbose": bool(verbose)}
    if solver == "SDPA":
        solver_opt.update(epsilonStar=tolerance, epsilonDash=tolerance,
                          mpfPrecision=200, omegaStar=1e12, numThreads=1)
    solverDetails = cp.Problem(cp.Minimize(accumulated_ek), cons)
    solve(solverDetails, **solver_opt)

# Try to grasp what happens by plotting!
if pplot:
    plt.close("all")
    ak = np.zeros(N + 1)
    apk = np.zeros(N + 1)
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
        apk[i-1] = float(ap[i].value if isinstance(ap[i], cp.Expression) else ap[i])
    for i in range(2, N + 2):
        ek[i-1] = float(e[i-1].value)
        ekc[i-1] = np.sum(ek[:i])

    plt.subplot(4, 2, 1)
    plt.plot(np.arange(1, N + 2), ak, "-b")
    plt.title('ak (coefficient of ||zk-x*||^2)')
    plt.subplot(4, 2, 2)
    plt.plot(np.arange(1, N + 2), bk, "-b")
    plt.title("bk (coefficient of ||f'(zk)||^2)")
    plt.subplot(4, 2, 3)
    plt.plot(np.arange(1, N + 2), ck, "-b")
    plt.title("ck (coefficient of <f'(zk);zk-x*>)")
    plt.subplot(4, 2, 4)
    plt.plot(np.arange(1, N + 2), dk, "-b")
    plt.title('dk (coefficient of f(zk)-f(x*))')
    plt.subplot(4, 2, 5)
    plt.plot(np.arange(1, N + 2), ek, "-b")
    plt.title('ek (coefficient of sigma^2)')
    plt.subplot(4, 2, 6)
    plt.plot(np.arange(1, N + 2), ekc, "-b")
    plt.title('cumulated ek')
    plt.subplot(4, 2, 7)
    plt.plot(np.arange(1, N + 2), apk, "-b")
    plt.title("ak' (coefficient of ||xk-x*||^2)")

    if ssave:
        labels = ['k', 'ak', 'bk', 'ck', 'dk', 'ek', 'ekc', 'apk']
        data = np.column_stack((np.arange(1, N + 2), ak, bk, ck, dk, ek, ekc, apk))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
