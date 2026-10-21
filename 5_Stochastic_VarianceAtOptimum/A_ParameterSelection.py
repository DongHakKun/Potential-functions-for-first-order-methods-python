from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from saveData import saveData

# SOLVER OPTIONS
verbose = 1
tolerance = 1e-8
solver = 'MOSEK'  # Same solver as the MATLAB original; requires a MOSEK license.

# OUTPUT OPTIONS
ssave = 1             # Save the results ?
pplot = 1             # Plot the results ?
folder = 'SaveData/'   # If results saved, name of the saving folder
nname = 'ParameterSelection.dat'  # Name of the file

# MINIMIZATION PROBLEM SETUP
L = 1       # Smoothness constant
m = 0       # Strong convexity constant
n = 2       # Cardinality of the support for the stochastic gradient
            # (i.e., number of functions in the finite sum)
N = 10      # Number of iterations

# POTENTIAL SETUP
# Potential has the form:
#   q1k ||x_k-x*||^2 + q2k ||f'(x_k)||^2 + 2 q3k E_i||fi'(xk)-fi'(x*)||^2
#       + q4k <f'(x_k); x_k-x*> + dk (f(x_k)-f(x*)) + ak ||z_k-x*||^2

# In this code, we fix dN to .9*N and minimize sum_k ek.
# INTERMEDIARY POTENTIAL SETUPS
# Options:
#   Set relax = 0: use the base potential stated above.
#   Set relax = 1: force ak = L/2.
#   Set relax = 2: force ak = L/2 and alphak=0 (in the method)
#   Set relax = 3: force ak = L/2, q1k=q2k=q3k=q4k=0 and alphak=0 (in the method)
relax = 3

# INITIAL AND FINAL POTENTIALS SETUP:
a0 = L/2
q10 = 0
q20 = 0
q30 = 0
q40 = 0
d0 = 0
aN = 0
q1N = 0
q2N = 0
q3N = 0
q4N = 0
dN = .9*N
if relax == 1:
    a0 = L/2
    aN = L/2
elif relax == 2:
    a0 = L/2
    aN = L/2
elif relax == 3:
    a0 = L/2
    aN = L/2
    q1N = q10 = 0
    q2N = q20 = 0
    q3N = q30 = 0
    q4N = q40 = 0

################################################################
# SETTING UP THE NOTATIONS FOR THE LINEAR MATRIX INEQUALITIES
#                   (end of editable zone)
################################################################
# Recall that x* (optimum) is set to x* = 0 without loss of generality,
# and so does f(x*) = 0. We also have fi'(x*) = 0 by assumption
# (over-parametrized models).
# P = [xk zk yk1 | xk1 ... xkn | f1'(x_k) ... fn'(x_k) |
#      f1'(y_{k+1}) ... fn'(y_{k+1}) |
#      f1'(x_{k+1}^(1)) ... fn'(x_{k+1}^(1)) | ... |
#      f1'(x_{k+1}^(n)) ... fn'(x_{k+1}^(n)) | f1'(x*) ... fn'(x*)]
# F = [f1(x_k) ... fn(x_k) | f1(y_{k+1}) ... fn(y_{k+1}) |
#      f1(x_{k+1}^(1)) ... fn(x_{k+1}^(1)) | ... |
#      f1(x_{k+1}^(n)) ... fn(x_{k+1}^(n))]
# NOTE: Symmetry arguments allow simplifying both F and G=P.T*P.
dimG = 2 + 4*n + n**2  # dimensions of the Gram matrix P.T*P
dimF = 2*n + n**2      # dimensions of F
nbPts = n + 3         # x*, y_k, x_k, x_{k+1}^(1), ..., x_{k+1}^(n)

xk = np.zeros(dimG)
xk[0] = 1                         # x_k
zk = np.zeros(dimG)
zk[1] = 1                         # z_k
yk1 = np.zeros(dimG)
yk1[2] = 1                        # y_{k+1}
xs = np.zeros(dimG)               # x*
xk1 = np.zeros((n, dimG))
xk1[:, 3:3+n] = np.eye(n)         # xk1[i,:] is x_{k+1}^(i)
gxk = np.zeros((n, dimG))
gxk[:, 3+n:3+2*n] = np.eye(n)     # gxk[i,:] is fi'(x_k)
GXK = np.sum(gxk, axis=0)/n       # f'(x_k)
gyk1 = np.zeros((n, dimG))
gyk1[:, 3+2*n:3+3*n] = np.eye(n)  # fi'(y_{k+1})
GYK1 = np.sum(gyk1, axis=0)/n     # f'(y_{k+1})
gxk1 = np.zeros((n, dimG, n))     # gxk1[i,:,j] is fi'(x_{k+1}^(j))
GXK1 = np.zeros((n, dimG))        # GXK1[j,:] is f'(x_{k+1}^(j))
gxs = np.zeros((n, dimG))         # fi'(x*)
gxs[:n-1, dimG-(n-1):dimG] = np.eye(n-1)
gxs[-1, :] = -np.sum(gxs, axis=0)
GXS = np.sum(gxs, axis=0)/n

fxk = np.zeros((n, dimF))
fxk[:, :n] = np.eye(n)            # fi(x_k)
FXK = np.sum(fxk, axis=0)/n       # f(x_k)
fyk1 = np.zeros((n, dimF))
fyk1[:, n:2*n] = np.eye(n)        # fi(y_{k+1})
FYK1 = np.sum(fyk1, axis=0)/n     # f(y_{k+1})
fxk1 = np.zeros((n, dimF, n))     # fi(x_{k+1}^(j))
FXK1 = np.zeros((n, dimF))        # f(x_{k+1}^(j))
fxs = np.zeros((n, dimF))         # fi(x*)
FXS = np.sum(fxs, axis=0)/n       # f(x*)
Gs_index = 3 + 3*n
Ge_index = Gs_index + n
Fs_index = 2*n
Fe_index = Fs_index + n
for i in range(1, n+1):
    gxk1[:, Gs_index:Ge_index, i-1] = np.eye(n)
    Gs_index += n
    Ge_index = Gs_index + n
    fxk1[:, Fs_index:Fe_index, i-1] = np.eye(n)
    Fs_index += n
    Fe_index = Fs_index + n
    FXK1[i-1, :] = np.sum(fxk1[:, :, i-1], axis=0)/n
    GXK1[i-1, :] = np.sum(gxk1[:, :, i-1], axis=0)/n

# TERMS IN THE POTENTIAL
# term 1: ||x-xs||^2
# term 2: ||f'(x)||^2
# term 3: E_i||fi'(x)-fi'(x*)||^2
# term 4: <f'(x); x-xs>
term1_k = np.outer(xk-xs, xk-xs)
term1_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    term1_k1 += np.outer(xk1[i-1, :]-xs, xk1[i-1, :]-xs)/n
term2_k = np.outer(GXK, GXK)
term2_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    term2_k1 += np.outer(GXK1[i-1, :], GXK1[i-1, :])/n
term3_k = np.zeros((dimG, dimG))
for i in range(1, n+1):
    term3_k += np.outer(gxk[i-1, :]-gxs[i-1, :], gxk[i-1, :]-gxs[i-1, :])/n
term3_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    for j in range(1, n+1):
        term3_k1 += np.outer(gxk1[i-1, :, j-1]-gxs[i-1, :], gxk1[i-1, :, j-1]-gxs[i-1, :])/n**2
term4_k = np.outer(xk-xs, GXK)
term4_k = (term4_k + term4_k.T)/2
term4_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    temp = np.outer(xk1[i-1, :]-xs, GXK1[i-1, :])
    temp = (temp + temp.T)/2
    term4_k1 += temp/n

# Line-searches for finding y_{k+1}.
A = {}
A[1] = np.outer(GYK1, xk-yk1)
A[1] = (A[1] + A[1].T)/2
A[2] = np.outer(GYK1, zk-xk)
A[2] = (A[2] + A[2].T)/2
# Line-searches for x_{k+1}^(i): average them using symmetry.
B = {}
for i in range(1, n+1):
    B[1, i] = np.outer(GXK1[i-1, :], yk1-xk1[i-1, :])
    B[1, i] = (B[1, i] + B[1, i].T)/2
    B[2, i] = np.outer(GXK1[i-1, :], gyk1[i-1, :])
    B[2, i] = (B[2, i] + B[2, i].T)/2

# Matrix encoding interpolation for smooth strongly convex functions.
M = np.array([[-L*m, L*m, m, -L],
              [L*m, -L*m, -m, L],
              [m, -m, -1, 1],
              [-L, L, 1, -1]])/(2*(L-m))
# Matrix encoding the variance condition.
Avar = np.zeros((dimG, dimG))
for i in range(1, n+1):
    Avar += np.outer(gxs[i-1, :]-GXS, gxs[i-1, :]-GXS)/n
statesKF = FXK-FXS
statesK1F = np.sum(FXK1, axis=0)/n-FXS

################################################################
# SETTING UP THE LINEAR MATRIX INEQUALITIES
# (directly embedded within the larger problem (2))
################################################################
# Dictionaries retain the original MATLAB cell indices k=1,...,N+1.
# 'lam' replaces MATLAB's 'lambda', a reserved Python keyword.
Q, d, a = {}, {}, {}
lam, mu_Y, mu_X, S, e = {}, {}, {}, {}, {}
cons_SDP, cons_LIN = {}, {}
Q[1] = np.array([q10, q20, q30, q40])
d[1] = d0
a[1] = a0
Q[N+1] = np.array([q1N, q2N, q3N, q4N])
d[N+1] = dN
a[N+1] = aN
cons = []
for k in range(1, N+1):
    if k < N:
        if relax == 0:  # no additional constraints on potentials
            Q[k+1] = cp.Variable(4, name=f'Q_{k+1}')
            d[k+1] = cp.Variable(name=f'd_{k+1}')
            a[k+1] = cp.Variable(name=f'a_{k+1}')
        elif relax == 1:  # ak = L/2
            Q[k+1] = cp.Variable(4, name=f'Q_{k+1}')
            d[k+1] = cp.Variable(name=f'd_{k+1}')
            a[k+1] = L/2
        elif relax == 2:  # ak = L/2 and alphak=0
            Q[k+1] = cp.Variable(4, name=f'Q_{k+1}')
            d[k+1] = cp.Variable(name=f'd_{k+1}')
            a[k+1] = L/2
        elif relax == 3:  # original code fixes Qk=0 and ak=L/2
            Q[k+1] = np.zeros(4)
            d[k+1] = cp.Variable(name=f'd_{k+1}')
            a[k+1] = L/2
    lam[k] = cp.Variable((nbPts, nbPts, n), name=f'lambda_{k}')
    mu_Y[k] = cp.Variable(2, name=f'mu_Y_{k}')
    mu_X[k] = cp.Variable(2, name=f'mu_X_{k}')
    e[k] = cp.Variable(name=f'e_{k}')  # multiplier for the variance
    # Fix the source declaration-order typo: mu_X is declared before use.
    if relax == 2:
        cons.append(mu_X[k][1] == 0)
    S[k] = cp.Variable((3, 3), symmetric=True, name=f'S_{k}')
    cons.append(S[k] >> 0)
    cons.append(S[k][0, 0] == a[k+1])
    dsstep_avg = np.zeros((dimG, dimG))
    for i in range(1, n+1):
        # E_i||z_{k+1}^(i)-x*||^2, with the original full 3x3 S_k.
        TT = np.column_stack((yk1, zk-yk1, gyk1[i-1, :]))
        dsstep = TT @ S[k] @ TT.T
        dsstep_avg = dsstep_avg + dsstep/n
    cons_SDP[k] = (
        -a[k]*np.outer(zk-xs, zk-xs)-e[k]*Avar + dsstep_avg
        -Q[k][0]*term1_k-Q[k][1]*term2_k-Q[k][2]*term3_k-Q[k][3]*term4_k
        +Q[k+1][0]*term1_k1+Q[k+1][1]*term2_k1
        +Q[k+1][2]*term3_k1+Q[k+1][3]*term4_k1
    )
    for i in range(1, 3):
        cons_SDP[k] = cons_SDP[k] + mu_Y[k][i-1]*A[i]
    for i in range(1, 3):
        for j in range(1, n+1):
            cons_SDP[k] = cons_SDP[k] + mu_X[k][i-1]*B[i, j]/n
    cons_LIN[k] = -d[k]*statesKF+d[k+1]*statesK1F
    for l in range(1, n+1):
        # Points in the discrete representation of f_l.
        X = {1: xs, 2: xk, 3: yk1}
        G = {1: gxs[l-1, :], 2: gxk[l-1, :], 3: gyk1[l-1, :]}
        F = {1: fxs[l-1, :], 2: fxk[l-1, :], 3: fyk1[l-1, :]}
        for i in range(1, n+1):
            X[3+i] = xk1[i-1, :]
            F[3+i] = fxk1[l-1, :, i-1]
            G[3+i] = gxk1[l-1, :, i-1]
        for i in range(1, nbPts+1):
            for j in range(1, nbPts+1):
                if j != i:
                    xi, xj = X[i], X[j]
                    gi, gj = G[i], G[j]
                    fi, fj = F[i], F[j]
                    TT = np.vstack((xi, xj, gi, gj))
                    cons_SDP[k] = cons_SDP[k]+lam[k][i-1, j-1, l-1]*(TT.T @ M @ TT)
                    cons_LIN[k] = cons_LIN[k]+lam[k][i-1, j-1, l-1]*(fi-fj)
    cons.append(cons_SDP[k] << 0)
    cons.append(cons_LIN[k] == 0)
    cons.append(lam[k] >= 0)
accumulated_ek = 0
for k in range(1, N+1):
    accumulated_ek = accumulated_ek + e[k]
obj = accumulated_ek

solver_opt = {'solver': solver, 'verbose': bool(verbose)}
if solver == 'CLARABEL':
    solver_opt.update(tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance)
elif solver == 'SCS':
    solver_opt.update(eps=tolerance)
elif solver == 'SDPA':
    solver_opt.update(epsilonStar=tolerance, epsilonDash=tolerance,
                      mpfPrecision=200, omegaStar=1e12, numThreads=1)
elif solver == 'MOSEK':
    solver_opt['mosek_params'] = {'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tolerance}

solver_cons = list(cons)
if (solver in ('CLARABEL', 'SDPA') and m == 0
        and all(not isinstance(Q[j][0], cp.Expression) and Q[j][0] == 0
                for j in (1, N+1))):
    # The original LMIs force Q1=0 and equal first two rows of S at m=0.
    # Zero diagonal entries of an NSD matrix force their entire rows to zero.
    # Use that equivalent face for solving; retain cons and all variables.
    solver_pos = np.r_[0, 2:3+n]
    solver_keep = np.r_[1, 3+n:dimG]
    if (all(not isinstance(a[j], cp.Expression) for j in range(1, N+2))
            and all(a[j] == a[1] for j in range(1, N+2))):
        solver_pos = np.arange(3+n)
        solver_keep = np.arange(3+n, dimG)
    solver_steps = {S[k].id: k for k in S}
    solver_cons = []
    for solver_constraint in cons:
        if not isinstance(solver_constraint, cp.constraints.psd.PSD):
            solver_cons.append(solver_constraint)
            continue
        solver_k = next(solver_steps[v.id] for v in solver_constraint.variables()
                        if v.id in solver_steps)
        if solver_constraint.shape == (3, 3):
            solver_S = S[solver_k]
            solver_cons.extend([
                solver_S[0, 1] == solver_S[0, 0],
                solver_S[1, 1] == solver_S[0, 0],
                solver_S[1, 2] == solver_S[0, 2],
                solver_S[np.ix_([0, 2], [0, 2])] >> 0,
            ])
        else:
            solver_M = cons_SDP[solver_k]
            solver_cons.extend([solver_M[solver_pos, :] == 0,
                                solver_M[np.ix_(solver_keep, solver_keep)] << 0])
solverDetails = cp.Problem(cp.Minimize(obj), solver_cons)
solverDetails.solve(**solver_opt)
print('Solver status:', solverDetails.status, 'objective:', solverDetails.value)

# Try to grasp what happens...
if pplot:
    plt.close('all')
    apk = np.zeros(N+1)
    Q1 = np.zeros(N+1)
    Q2 = np.zeros(N+1)
    Q3 = np.zeros(N+1)
    Q4 = np.zeros(N+1)
    dk = np.zeros(N+1)
    alphak = np.zeros(N+1)
    tauk = np.zeros(N+1)
    deltak = np.zeros(N+1)
    gammak = np.zeros(N+1)
    ek = np.zeros(N+1)
    ekc = np.zeros(N+1)
    tolp = 1e-5
    for i in range(1, N+2):
        apk[i-1] = a[i].value if isinstance(a[i], cp.Expression) else a[i]
        Q1[i-1] = Q[i][0].value if isinstance(Q[i], cp.Expression) else Q[i][0]
        Q2[i-1] = Q[i][1].value if isinstance(Q[i], cp.Expression) else Q[i][1]
        Q3[i-1] = Q[i][2].value if isinstance(Q[i], cp.Expression) else Q[i][2]
        Q4[i-1] = Q[i][3].value if isinstance(Q[i], cp.Expression) else Q[i][3]
        dk[i-1] = d[i].value if isinstance(d[i], cp.Expression) else d[i]
    for i in range(1, N+1):
        tauk[i-1] = np.divide(mu_Y[i].value[1], mu_Y[i].value[0])
        alphak[i-1] = -np.divide(mu_X[i].value[1], mu_X[i].value[0])
        deltak[i-1] = np.divide(S[i].value[0, 1], S[i].value[0, 0])
        gammak[i-1] = -np.divide(S[i].value[0, 2], S[i].value[0, 0])
    for i in range(2, N+2):
        ek[i-1] = e[i-1].value
        ekc[i-1] = np.sum(ek[:i])
    plt.figure()
    plt.subplot(4, 3, 1)
    plt.plot(np.arange(1, N+2), Q1, '-b')
    plt.title('q1k (coefficient of ||xk-x*||^2)')
    plt.subplot(4, 3, 2)
    plt.plot(np.arange(1, N+2), ek, '-b')
    plt.title('ek (coefficient of sigma^2)')
    plt.subplot(4, 3, 3)
    plt.plot(np.arange(1, N+2), ekc, '-b')
    plt.title('cumulated ek')
    plt.subplot(4, 3, 4)
    plt.plot(np.arange(1, N+2), Q2, '-b')
    plt.title("q2k (coefficient of ||f'(xk)||^2)")
    plt.subplot(4, 3, 5)
    plt.plot(np.arange(1, N+2), Q3, '-b')
    plt.title("q3k (coefficient of Ei||fi'(xk)||^2)")
    plt.subplot(4, 3, 6)
    plt.plot(np.arange(1, N+2), Q4, '-b')
    plt.title("q4k (coefficient of <f'(xk); xk-x*>)")
    plt.subplot(4, 3, 7)
    plt.plot(np.arange(1, N+2), dk, '-b')
    plt.title('dk (coefficient of f(xk)-f(x*))')
    plt.subplot(4, 3, 8)
    plt.plot(np.arange(1, N+2), apk, '-b')
    plt.title('ak (coefficient of ||zk-x*||^2)')
    plt.subplot(4, 3, 9)
    plt.plot(np.arange(1, N+1), tauk[:-1], '-b')
    plt.title('tauk')
    plt.subplot(4, 3, 10)
    plt.plot(np.arange(1, N+1), alphak[:-1], '-b')
    plt.title('alphak')
    plt.subplot(4, 3, 11)
    plt.plot(np.arange(1, N+1), deltak[:-1], '-b')
    plt.title('deltak')
    plt.subplot(4, 3, 12)
    plt.plot(np.arange(1, N+1), gammak[:-1], '-b')
    plt.title('gammak')
    if ssave:
        labels = ['k', 'Q1', 'Q2', 'Q3', 'Q4', 'apk', 'dk',
                  'alphak', 'tauk', 'deltak', 'gammak', 'ek', 'ekc']
        data = np.column_stack((np.arange(1, N+2), Q1, Q2, Q3, Q4, apk, dk,
                                alphak, tauk, deltak, gammak, ek, ekc))
        saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
    plt.show()
