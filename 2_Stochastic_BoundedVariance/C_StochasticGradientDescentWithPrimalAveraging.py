from pathlib import Path
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from saveData import saveData

# This code simply checks feasibility of Theorem 7 for primal averaging,
# (potential function is partially imposed and only the \sum e_k is minimized.)

# SOLVER OPTIONS
verbose = 1
tolerance = 1e-8
solver = "MOSEK"  # Original MATLAB solver; requires an available MOSEK license.

# OUTPUT OPTIONS
ssave = 1  # Save the results?
pplot = 1  # Plot the results?
folder = "SaveData/"  # If results saved, name of the saving folder.
nname = "SGD_Primal_Averaging.dat"  # Name of the file.

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
#   ak ||y_k-x*||^2 + bk ||f'(y_k)||^2 + 2 ck <f'(y_k); y_k-x*>
#       + dk (f(y_k)-f(x*)) + ak' || x_k-x* ||^2
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

aN = 0
bN = 0
cN = 0
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
# P = [ xk  y_k  f'(y_{k}) G(y_{k+1};1) ... G(y_{k+1};n)]
# F = [           f(y_k)         f(y_{k+1})  ]
# NOTE: Symmetry arguments allow simplifying both F and G=P.'*P.
dimG = 3 + n  # Dimensions of the Gram matrix P.T @ P.
dimF = 2  # Dimensions of F.
nbPts = 3  # x*, y_k, y_{k+1}.

# Notations for the LMI.
xk = np.zeros(dimG)
xk[0] = 1  # x_k.
yk = np.zeros(dimG)
yk[1] = 1  # y_k.
xs = np.zeros(dimG)  # x* = 0.
gyk = np.zeros(dimG)
gyk[2] = 1  # f'(y_k).
Gyk1 = np.zeros((n, dimG))
Gyk1[:, 3:n+3] = np.eye(n)  # G(y_{k+1}; i).
gyk1 = np.sum(Gyk1, axis=0) / n  # f'(y_{k+1}).
gxs = np.zeros(dimG)  # f'(x*) = 0.

fyk = np.zeros(dimF)
fyk[0] = 1  # f(y_k).
fyk1 = np.zeros(dimF)
fyk1[1] = 1  # f(y_{k+1}).
fxs = np.zeros(dimF)  # f(x*) = 0.

yk1 = lambda dk, k: delta(k) * L / (dk + delta(k) * L) * xk + dk / (dk + delta(k) * L) * yk
xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gyk1

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
    Avar = Avar + np.outer(Gyk1[i-1] - gyk1, Gyk1[i-1] - gyk1) / n
# Notations for potentials.
Q0 = np.array([[a0, c0], [c0, b0]])
q0 = d0
QN = np.array([[aN, cN], [cN, bN]])
statesKG = np.column_stack((yk - xs, gyk))
statesKF = fyk
statesK1F = fyk1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                              %
#          SETTING UP THE LINEAR MATRIX INEQUALITIES           %
#     (directly embedded within the larger problem (2))        %
#                                                              %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dictionaries keep the original MATLAB cell indices: 1, ..., N+1.
Q = {1: Q0, N+1: QN}
d = {1: q0}
ap = {1: ap0, N+1: apN}
# lambda is a Python keyword; lam corresponds to MATLAB lambda{k}.
lam = {}
e = {}
cons_SDP = {}
cons_LIN = {}
cons = []
for k in range(1, N + 1):  # One LMI per iteration.
    if k < N:
        if relax == 0:  # No constraint on Qk and dk.
            ap[k+1] = cp.Variable(name=f"ap_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:  # Force ak = bk = ck = 0 and ak' = L/2.
            ap[k+1] = L / 2
            Q[k+1] = np.zeros((2, 2))
    d[k+1] = d[k] + delta(k) * L

    # POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.).
    XK1 = xk1(k)
    X = {1: xs, 2: yk, 3: yk1(d[k], k)}
    G = {1: gxs, 2: gyk, 3: gyk1}
    F = {1: fxs, 2: fyk, 3: fyk1}

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    e[k] = cp.Variable(name=f"e_{k}")
    cons_SDP[k] = -statesKG @ Q[k] @ statesKG.T - ap[k] * np.outer(xk-xs, xk-xs) - e[k] * Avar
    cons_LIN[k] = -d[k] * statesKF

    statesK1G = np.column_stack((yk1(d[k], k) - xs, gyk1))
    cons_SDP[k] = cons_SDP[k] + statesK1G @ Q[k+1] @ statesK1G.T
    for i in range(1, n + 1):  # Averaging the states over i.
        cons_SDP[k] = cons_SDP[k] + ap[k+1] * np.outer(XK1[i-1]-xs, XK1[i-1]-xs) / n
    cons_LIN[k] = cons_LIN[k] + statesK1F * d[k+1]

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
obj = accumulated_ek

solver_opt = {"solver": solver, "verbose": bool(verbose)}
if solver == "CLARABEL":
    solver_opt.update(tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance)
elif solver == "SCS":
    solver_opt.update(eps=tolerance)
elif solver == "MOSEK":
    solver_opt["mosek_params"] = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance}
solverDetails = cp.Problem(cp.Minimize(obj), cons)
solverDetails.solve(**solver_opt)

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
    plt.title('ak (coefficient of ||yk-x*||^2)')
    plt.subplot(4, 2, 2)
    plt.plot(np.arange(1, N + 2), bk, "-b")
    plt.title("bk (coefficient of ||f'(yk)||^2)")
    plt.subplot(4, 2, 3)
    plt.plot(np.arange(1, N + 2), ck, "-b")
    plt.title("ck (coefficient of <f'(yk);yk-x*>)")
    plt.subplot(4, 2, 4)
    plt.plot(np.arange(1, N + 2), dk, "-b")
    plt.title('dk (coefficient of f(yk)-f(x*))')
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
