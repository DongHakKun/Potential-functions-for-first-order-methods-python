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
nname  = 'FastGradientD1.dat'  # Name of the file

# MINIMIZATION PROBLEM SETUP
L = 1  # Smoothness constant
m = 0  # Strong convexity constant
N = 100  # Number of iterations

# INTERMEDIARY POTENTIAL SETUPS (aims at reproducing results from Appendix
# C.3; Figure 3) via option "relaxation".
# Options:
#   Set relax = 0: use the base potential stated in Section 3.2.1.
#   Set relax = 1: force ak = L/2; Qk = 0
#   Set relax = 2: force ak = L/2, Qk(1,1)=Qk(1,2)=0, Qk(2,2)=dk/2/L

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
    Q0 = cp.bmat([[0, 0], [0, d0/2/L]]); QN = cp.bmat([[0, 0], [0, dN/2/L]])

################################################################
#                                                              #
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES  #
#                   (end of editable zone)                     #
#                                                              #
################################################################
# Minimization of f(x) with f smooth (possibly strongly) convex.
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0.

# P = [ z_k x_k x_{k+1} y_{k+1} f'(x_k) f'(x_{k+1}) f'(y_{k+1}) ]
# F = [                         f(x_k)  f(x_{k+1})  f(y_{k+1})    ]
dimG = 7  # dimensions of the Gram matrix P.T@P
dimF = 3  # dimensions of F
nbPts = 4  # x*, x_k, x_{k+1}, y_{k+1} in the discrete function f

# Notations for the LMI. Keep MATLAB's row-vector shapes explicitly.
zk = np.zeros((1, dimG)); zk[0, 0] = 1    # this is z_k
xk = np.zeros((1, dimG)); xk[0, 1] = 1    # this is x_k
xk1 = np.zeros((1, dimG)); xk1[0, 2] = 1  # this is x_{k+1}
yk1 = np.zeros((1, dimG)); yk1[0, 3] = 1  # this is y_{k+1}
xs = np.zeros((1, dimG))                  # this is x*

gxk = np.zeros((1, dimG)); gxk[0, 4] = 1    # this is f'(x_k)
gxk1 = np.zeros((1, dimG)); gxk1[0, 5] = 1  # this is f'(x_{k+1})
gyk1 = np.zeros((1, dimG)); gyk1[0, 6] = 1  # this is f'(y_{k+1})
gxs = np.zeros((1, dimG))                   # this is f'(x*) = 0

fxk = np.zeros((1, dimF)); fxk[0, 0] = 1    # this is f(x_k)
fxk1 = np.zeros((1, dimF)); fxk1[0, 1] = 1  # this is f(x_{k+1})
fyk1 = np.zeros((1, dimF)); fyk1[0, 2] = 1  # this is f(y_{k+1})
fxs = np.zeros((1, dimF))                   # this is f(x*)

# POINTS IN THE DISCRETE REPRESENTATION OF FUNCTION f(.)
X = {1: xs, 2: xk, 3: xk1, 4: yk1}  # coordinates (notations w in the paper)
G = {1: gxs, 2: gxk, 3: gxk1, 4: gyk1}  # gradients
F = {1: fxs, 2: fxk, 3: fxk1, 4: fyk1}  # function values

# line-search for y_{k+1}
A = {}
A[1] = gyk1.T @ (yk1-xk); A[1] = 1/2 * (A[1]+A[1].T)  # <f'(y_{k+1}); y_{k+1}-x_k> == 0
A[2] = gyk1.T @ (zk-xk); A[2] = 1/2 * (A[2]+A[2].T)    # <f'(y_{k+1}); z_k-x_k> == 0
# line-search for x_{k+1}
A[3] = gxk1.T @ (yk1-xk1); A[3] = 1/2 * (A[3]+A[3].T)  # <f'(x_{k+1}); y_{k+1}-x_{k+1}> == 0
A[4] = gxk1.T @ gyk1; A[4] = 1/2 * (A[4]+A[4].T)        # <f'(x_{k+1}); f'(y_{k+1})> == 0

# Matrix encoding interpolation condition for smooth strongly convex functions
M = 1/2/(L-m) * np.array([
    [-L*m,  L*m,  m, -L],
    [ L*m, -L*m, -m,  L],
    [    m,   -m, -1,  1],
    [   -L,    L,  1, -1],
])

# Notations for potentials
statesKG = np.vstack((xk-xs, gxk)); statesKF = fxk-fxs
statesK1G = np.vstack((xk1-xs, gxk1)); statesK1F = fxk1-fxs

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
        elif relax == 2:  # force ak = L/2, Qk(1,1)=Qk(1,2)=0, Qk(2,2)=dk/2/L
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            a[k+1] = L/2
            Q[k+1] = cp.bmat([[0, 0], [0, d[k+1]/2/L]])

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    mu[k] = cp.Variable((4, 1), name=f"mu_{k}")  # multipliers for line-search conditions
    S[k] = cp.Variable((3, 3), symmetric=True, name=f"S_{k}")  # this is Sk (see Section C.5)
    cons.append(S[k] >> 0)
    cons.append(S[k][0, 0] == a[k+1])

    # Form the SDP constraint
    cons_SDP[k] = (-a[k] * ((zk-xs).T @ (zk-xs))
                   - statesKG.T @ Q[k] @ statesKG
                   + np.vstack((yk1, zk-yk1, gyk1)).T @ S[k] @ np.vstack((yk1, zk-yk1, gyk1))
                   + statesK1G.T @ Q[k+1] @ statesK1G)
    for i in range(1, 5):
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

# Equivalent solver representation for CLARABEL and SDPA; retain the original cons above.
# At m=0, the x_k and x_next diagonals of M_k are -Q11_k and Q11_next.
# Adjacent LMIs and zero endpoint Q11 force every Q11 to vanish. The y diagonal
# is (1,-1,0).T @ S @ (1,-1,0), which is nonnegative since S >= 0 and must also
# be nonpositive since M <= 0. Thus the first two rows of S coincide. Its [0,2]
# principal block is PSD iff the full S is PSD under these equalities. The x,
# x_next, y rows of M then vanish; z also vanishes when every ak is the same.
solver_cons = cons
# relax1/2 set both Q11 endpoints to zero above; relax0 retains editable Q0/QN.
zero_Q11_endpoints = relax in (1, 2) or (
    relax == 0 and not isinstance(Q0, cp.Expression) and not isinstance(QN, cp.Expression)
    and Q0[0, 0] == 0 and QN[0, 0] == 0)
if solver in ("CLARABEL", "SDPA") and m == 0 and zero_Q11_endpoints:
    solver_cons = [con for con in cons if not isinstance(con, cp.constraints.psd.PSD)]
    fixed_a = relax in (1, 2) and all(
        not isinstance(a[k], cp.Expression) and a[k] == L/2 for k in range(1, N+2))
    positions = [0, 1, 2, 3] if fixed_a else [1, 2, 3]
    principal = [4, 5, 6] if fixed_a else [0, 4, 5, 6]
    for k in range(1, N+1):
        solver_cons.extend([S[k][0, 1] == S[k][0, 0], S[k][1, 1] == S[k][0, 0],
                            S[k][1, 2] == S[k][0, 2], S[k][np.ix_([0, 2], [0, 2])] >> 0,
                            cons_SDP[k][positions, :] == 0,
                            cons_SDP[k][np.ix_(principal, principal)] << 0])
problem = cp.Problem(cp.Minimize(-obj), solver_cons)
solverDetails = solve(problem, **solver_opt)
if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    raise RuntimeError(f"The SDP returned {problem.status}; no solution can be plotted or saved.")
print(f"status={problem.status}, objective={obj.value:.10g}")
if solver_cons is not cons:
    max_original_constraint_violation = max(float(np.max(con.violation())) for con in cons)
    print(f"max_original_constraint_violation={max_original_constraint_violation:.3g}")
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

    tolp = 1e-5
    for i in range(1, N+2):
        Q11[i-1] = float(Q[i][0, 0].value if isinstance(Q[i][0, 0], cp.Expression) else Q[i][0, 0])
        Q12[i-1] = float(Q[i][0, 1].value if isinstance(Q[i][0, 1], cp.Expression) else Q[i][0, 1])
        Q22[i-1] = float(Q[i][1, 1].value if isinstance(Q[i][1, 1], cp.Expression) else Q[i][1, 1])
        ak[i-1] = float(a[i].value if isinstance(a[i], cp.Expression) else a[i])
        dk[i-1] = float(d[i].value if isinstance(d[i], cp.Expression) else d[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, N+1):
            tauk[i-1] = -np.divide(mu[i][1, 0].value, mu[i][0, 0].value)
            alphak[i-1] = np.divide(mu[i][3, 0].value, mu[i][2, 0].value)
            gammak[i-1] = -np.divide(S[i][0, 2].value, S[i][0, 0].value)
            deltak[i-1] = np.divide(S[i][0, 1].value, S[i][0, 0].value)

    plt.subplot(3, 3, 1)
    plt.plot(np.arange(1, N+2), Q11, "-b"); plt.title("Q11 (coefficient of ||xk-x*||^2)")
    plt.subplot(3, 3, 2)
    plt.plot(np.arange(1, N+2), Q12, "-b"); plt.title("Q12 (coefficient of <f'(xk);xk-x*>)")
    plt.subplot(3, 3, 3)
    plt.plot(np.arange(1, N+2), Q22, "-b"); plt.title("Q22 (coefficient of ||f'(xk)||^2)")
    plt.subplot(3, 3, 4)
    plt.plot(np.arange(1, N+2), ak, "-b"); plt.title("ak (coefficient of ||zk-x*||^2)")
    plt.subplot(3, 3, 5)
    plt.plot(np.arange(1, N+2), dk, "-b"); plt.title("dk (coefficient of f(xk)-f(x*))")
    plt.subplot(3, 3, 6)
    plt.plot(np.arange(1, N+1), tauk[:-1], "-b"); plt.title("tauk")
    plt.subplot(3, 3, 7)
    plt.plot(np.arange(1, N+1), alphak[:-1], "-b"); plt.title("alpha")
    plt.subplot(3, 3, 8)
    plt.plot(np.arange(1, N+1), gammak[:-1], "-b"); plt.title("gammak")
    plt.subplot(3, 3, 9)
    plt.plot(np.arange(1, N+1), deltak[:-1], "-b"); plt.title("deltak")

    if ssave:
        labels = ["k", "Q11", "Q12", "Q22", "ak", "dk", "alphak", "tauk", "deltak", "gammak"]
        data = np.column_stack((np.arange(1, N+2), Q11, Q12, Q22, ak, dk, alphak, tauk, deltak, gammak))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
