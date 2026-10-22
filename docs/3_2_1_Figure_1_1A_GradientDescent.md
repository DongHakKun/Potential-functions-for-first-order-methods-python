# 실험 1. Gradient Descent — §3.2.1, Figure 1

이 문서는 논문 §3.2.1의 식 (2)–(3), Figure 1, Appendix D.4와 [A_GradientDescent.py](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/1_Deterministic_SmoothConvex/A_GradientDescent.py)의 수치 실험을 설명한다. 논문의 해당 위치는 pp.6–7, 34–36이다. `relax`는 0, 1, 2, 3 중 하나를 선택하며, 한 번 실행할 때 같은 값을 1–9단계 전체에 사용한다. Figure 1은 네 설정을 각각 실행한 결과를 비교한다.

수식의 상태 인덱스는 $k=0,\ldots,N$, 전이 인덱스는 $k=0,\ldots,N-1$이다. Python 딕셔너리의 키와 원본 코드 블록의 반복변수 `k`는 1부터 시작한다. 따라서 $Q_k\leftrightarrow\texttt{Q[k+1]}$, $d_k\leftrightarrow\texttt{d[k+1]}$, $\lambda\leftrightarrow\texttt{lam[k+1]}$로 대응한다. 코드 블록의 식별자는 원본 이름을 유지한다.

수식에서는 점 $x_k$와 gradient $f'(x_k)$를 쓰고, Appendix D.4의 Gram 표현에 사용하는 벡터는 $\mathbf x_k,\mathbf g_k,\mathbf f_k$처럼 굵게 쓴다. 표준기저는 $\mathbf e_j$, 최적점과 최적값은 $x_\star,f_\star$로 쓴다. Interpolation point의 인덱스는 $p,q$로 쓴다. 3–5단계에서는 논문의 Gram 표현과 제약식을 설명하고, 6단계에서 이를 CVXPY로 구현한다.

## 1. Parameterized Potential

논문 식 (2)의 potential family는 다음과 같다. `relax` 값에 따라 일부 계수는 SDP 결정변수로 두고, 나머지는 상수로 고정한다.

$$
\begin{aligned}
\phi_k^f
&=\begin{pmatrix}x_k-x_\star\\f'(x_k)\end{pmatrix}^{\!\top}
(Q_k\otimes I_d)
\begin{pmatrix}x_k-x_\star\\f'(x_k)\end{pmatrix}
+d_k(f(x_k)-f_\star),\\
Q_k&=\begin{pmatrix}a_k&c_k\\c_k&b_k\end{pmatrix}.
\end{aligned}
$$

전개하면 $a_k\|x_k-x_\star\|^2+b_k\|f'(x_k)\|^2+2c_k\langle f'(x_k),x_k-x_\star\rangle+d_k(f(x_k)-f_\star)$이다.

| 수식의 기호 | Python 표현 | 의미 |
| --- | --- | --- |
| $L,\mu,N$ | `L=1`, `m=0`, `N=100` | smoothness, strong convexity, horizon |
| $\delta_k$ | `delta(k+1)` | 기본값 $1/L=1$인 고정 step-size |
| $Q_k$ | `Q[k+1]` | 대칭 2×2 potential 계수 행렬 |
| $a_k,b_k,c_k$ | `Q[k+1][0,0]`, `[1,1]`, `[1,0]` | 거리 제곱, gradient norm 제곱, 교차항의 계수 |
| $d_k$ | `d[k+1]` | 함수값 차이의 계수 |
| $b_N$ | `tau`, `bN` | 최대화할 최종 계수 |

```python
delta = lambda k: 1/L  # Step-size (possibly varying function of k)
```

중간 상태 $k=1,\ldots,N-1$의 계수 조건은 다음과 같다. 초기·최종 계수는 2단계에서 지정한다.

| `relax` | 자유로운 중간 계수 | 고정된 중간 계수 |
| --- | --- | --- |
| 0 | $a_k,b_k,c_k,d_k$ | 없음 |
| 1 | $a_k,b_k,c_k$ | $d_k=(2k+1)L$ |
| 2 | $b_k$ | $a_k=L^2,\ c_k=0,\ d_k=(2k+1)L$ |
| 3 | $a_k,b_k,c_k$ | $d_k=0$ |

$Q_k$는 모든 설정에서 대칭 계수 행렬이다. `relax=0,1,3`에서는 대칭 행렬 변수로, `relax=2`에서는 상수 성분과 스칼라 변수 $b_k$를 조합한 행렬로 구현한다. 별도의 $Q_k\succeq0$ 제약은 두지 않는다. 입력은 $L=1,\mu=0,N=100$, step-size는 $\delta_k=1/L$로 지정하며, `relax`의 선택은 이 입력값과 독립적이다.

## 2. Potential Inequality

`relax` 값과 관계없이 GD 갱신식과 potential inequality는 다음과 같다.

$$
x_{k+1}=x_k-\delta_k f'(x_k),\qquad \delta_k=1/L,
\qquad
\boxed{\phi_{k+1}^f-\phi_k^f\le0},\quad k=0,\ldots,N-1.
$$

초기·최종 계수 조건은 `relax` 값에 따라 아래와 같다. 공통으로 $a_0=L^2$, $b_0=c_0=c_N=0$, $b_N=\texttt{tau}$다.

| `relax` | $d_0$ | $a_N$ | $d_N$ |
| --- | --- | --- | --- |
| 0 | $0$ | $0$ | $0$ |
| 1 | $L$ | $0$ | $(2N+1)L$ |
| 2 | $L$ | $L^2$ | $(2N+1)L$ |
| 3 | $0$ | $0$ | $0$ |

따라서 각 설정에서 초기·최종 potential의 계수 행렬은

$$
Q_0=\begin{pmatrix}L^2&0\\0&0\end{pmatrix},\qquad
Q_N=\begin{pmatrix}a_N&0\\0&b_N\end{pmatrix}
$$

이며, 이에 대응하는 초기·최종 potential은 다음과 같다.

$$
\begin{aligned}
\phi_0^f&=L^2\|x_0-x_\star\|^2+d_0(f(x_0)-f_\star),\\
\phi_N^f&=a_N\|x_N-x_\star\|^2+b_N\|f'(x_N)\|^2+d_N(f(x_N)-f_\star).
\end{aligned}
$$

한 단계 부등식을 모두 더하면 $\phi_N^f\le\phi_0^f$를 얻는다. 이를 `relax` 값에 따라 풀어 쓰면 다음과 같다.

| `relax` | 최종 관계 |
| --- | --- |
| 0, 3 | $b_N\|f'(x_N)\|^2\le L^2\|x_0-x_\star\|^2$ |
| 1 | $b_N\|f'(x_N)\|^2+(2N+1)L(f(x_N)-f_\star)\le L^2\|x_0-x_\star\|^2+L(f(x_0)-f_\star)$ |
| 2 | $L^2\|x_N-x_\star\|^2+b_N\|f'(x_N)\|^2+(2N+1)L(f(x_N)-f_\star)\le L^2\|x_0-x_\star\|^2+L(f(x_0)-f_\star)$ |

위 중간 계수 조건과 초기·최종 계수 조건 아래에서 $b_N$을 최대화한다. Objective는 $b_N$이며 코드의 `tau`에 해당한다. Step-size는 모든 단계에서 $\delta_k=1/L$이며, 기본 설정 `L=1`에서는 1이다.

## 3. Finite-dimensional Gram representation

일반성을 잃지 않고 $x_\star=f_\star=0$으로 두고, 미분 가능한 볼록함수의 최적성 조건에 따라 최적점에서 $f'(x_\star)=0$임을 이용한다. 한 단계의 계산에 필요한 벡터와 함수값을 다음과 같이 모은다.

$$
P=[x_k\ f'(x_k)\ f'(x_{k+1})],\quad
F=[f(x_k)\ f(x_{k+1})],\quad G=P^\top P\in\mathbb S_+^3
$$

$P\in\mathbb R^{d\times3}$는 세 벡터를 열로 모은 행렬이고, $F\in\mathbb R^{1\times2}$는 두 함수값을 모은 행렬이다. Gram 표현에 사용할 기호를 다음과 같이 둔다.

$$
\mathbf x_k=\mathbf e_1,\quad \mathbf g_k=\mathbf e_2,\quad
\mathbf g_{k+1}=\mathbf e_3,\quad
\mathbf x_{k+1}=\mathbf e_1-L^{-1}\mathbf e_2.
$$

예를 들어 $P\mathbf x_k=x_k$이고 $\|x_k\|^2=\operatorname{Tr}(G\mathbf x_k\mathbf x_k^\top)$이다.

```python
dimG = 3
dimF = 2
nbPts = 3
xk = np.zeros((1, dimG)); xk[0, 0] = 1  # this is x_k
xs = np.zeros((1, dimG))                # this is x* = 0

gxk = np.zeros((1, dimG)); gxk[0, 1] = 1    # this is f'(x_k)
gxk1 = np.zeros((1, dimG)); gxk1[0, 2] = 1  # this is f'(x_{k+1})
gxs = np.zeros((1, dimG))                   # this is f'(x*) = 0

fxk = np.zeros((1, dimF)); fxk[0, 0] = 1    # this is f(x_k)
fxk1 = np.zeros((1, dimF)); fxk1[0, 1] = 1  # this is f(x_{k+1})
fxs = np.zeros((1, dimF))                   # this is f(x*) = 0

xk1 = lambda k: xk - delta(k) * gxk  # this is x_{k+1}

statesKG = np.hstack(((xk-xs).T, gxk.T))
statesK1G = lambda k: np.hstack(((xk1(k)-xs).T, gxk1.T))
statesKF = fxk
statesK1F = fxk1
```

Appendix D.4, p.36의 $V_k,V_{k+1}$를 직접 쓰면 다음과 같다.

$$
\begin{aligned}
V_k&=[\mathbf x_k-\mathbf x_\star\ \mathbf g_k]
Q_k[\mathbf x_k-\mathbf x_\star\ \mathbf g_k]^\top,\\
V_{k+1}&=[\mathbf x_{k+1}-\mathbf x_\star\ \mathbf g_{k+1}]
Q_{k+1}[\mathbf x_{k+1}-\mathbf x_\star\ \mathbf g_{k+1}]^\top.
\end{aligned}
$$

코드의 `statesKG`와 `statesK1G(k)`는 위 행렬곱에서 $Q_k,Q_{k+1}$의 왼쪽에 놓이는 두 열을 만든다. 각각 3×2이며, potential 이차항의 차이는 $\operatorname{Tr}(G(V_{k+1}-V_k))$이다.

실제 코드는 Gram matrix를 `cp.Variable`로 선언하지 않고, dualization으로 얻은 LMI 조건을 구성해 potential의 감소를 보장할 계수를 찾는다. 코드의 `G`와 `F`는 interpolation에 쓸 $\mathbf g_p,\mathbf f_p$를 저장한 딕셔너리로, 수식에서 정의한 Gram matrix $G$와 함수값 행렬 $F$와는 역할이 다르다.

수식의 $\mathbf x_k,\mathbf g_k,\mathbf f_k$는 열벡터이고, 위 코드의 `xk`, `gxk`, `fxk` 등은 행벡터다. 따라서 $\mathbf x_k\leftrightarrow\texttt{xk.T}$, $\mathbf g_k\leftrightarrow\texttt{gxk.T}$, $\mathbf f_k\leftrightarrow\texttt{fxk.T}$로 대응한다.

## 4. Interpolation matrices + multipliers

함수 $f\in\mathcal F_L$의 두 점에서 얻은 함수값과 gradient는 다음 smooth convex interpolation 조건을 만족한다.

$$
f(x_p)-f(x_q)-\langle f'(x_q),x_p-x_q\rangle
-\frac{1}{2L}\|f'(x_p)-f'(x_q)\|^2\ge0.
$$

여기서 $p,q$는 interpolation point의 인덱스이다. $P\mathbf x_p=x_p$, $P\mathbf g_p=f'(x_p)$, $F\mathbf f_p=f(x_p)$를 대입하면 위 부등식을 다음과 같이 쓸 수 있다.

$$
\begin{aligned}
M_{p,q}&=[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]
M[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]^\top,\\
F(\mathbf f_p-\mathbf f_q)+\operatorname{Tr}(GM_{p,q})&\ge0.
\end{aligned}
$$

논문 Appendix D.4의 식 (19)는 $\mu=0$인 경우를 다룬다. Python 구현은 strong convexity까지 지원하는 다음 행렬을 사용하며, 기본 설정인 `m=0`을 대입하면 논문과 같은 식이 된다.

$$
M=\frac{1}{2(L-\mu)}
\begin{pmatrix}
-L\mu&L\mu&\mu&-L\\
L\mu&-L\mu&-\mu&L\\
\mu&-\mu&-1&1\\
-L&L&1&-1
\end{pmatrix}.
$$

코드에서는 `TT = np.vstack((xi, xj, gi, gj))`로 $[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]^\top$을 만든다. 따라서 `TT.T @ M @ TT`를 계산하면 위 식의 $M_{p,q}$를 얻는다.

각 interpolation inequality에는 nonnegative multiplier $\lambda_{p,q}\ge0$를 곱해 더한다. 행렬 항과 함수값 항이 같은 부등식에서 나오므로, 6단계의 코드에서도 두 항에 동일한 `lam[k][i-1, j-1]`을 곱한다. 이때 `lam[k] >= 0`은 PSD 제약이 아니라 각 multiplier의 부호를 제한하는 원소별 조건이다.

이 실험에서는 $x_\star,x_k,x_{k+1}$의 세 점을 사용하며, 그 인덱스 집합은 $I_k=\{\star,k,k+1\}$이다. 서로 다른 두 점을 순서까지 구분해 고르면 6쌍이므로, 한 단계마다 interpolation inequality 6개를 더하게 된다. 각 $M_{p,q}$는 앞에서 정한 Gram matrix와 같은 3×3 크기이다.

`lam[k]`는 3×3 행렬이다. 6단계의 `if j != i`는 같은 점끼리의 조합을 제외하므로, 9개 원소 중 대각 원소 3개를 제외한 6개만 interpolation 합에 사용한다. 수식의 점 라벨 $p,q$는 코드의 점 번호 `i,j`에 대응한다.

행렬 $M$은 다음과 같이 구현한다.

```python
M = 1/2/(L-m) * np.array([
    [-L*m,  L*m,  m, -L],
    [ L*m, -L*m, -m,  L],
    [    m,   -m, -1,  1],
    [   -L,    L,  1, -1],
])
```

## 5. Linear equality + semidefinite inequality

Appendix D.4, p.36의 조건은 다음과 같다. 아래 식의 $\lambda_{p,q}$는 한 전이에서 사용하는 multiplier이다.

$$
\begin{aligned}
V_{k+1}-V_k+\sum_{p,q\in I_k}\lambda_{p,q}M_{p,q}&\preceq0,\\
d_{k+1}(\mathbf f_{k+1}-\mathbf f_\star)
-d_k(\mathbf f_k-\mathbf f_\star)
+\sum_{p,q\in I_k}\lambda_{p,q}(\mathbf f_p-\mathbf f_q)&=0,\\
\lambda_{p,q}&\ge0.
\end{aligned}
$$

함수값에 곱해지는 계수를 0으로 두면 함수값 항이 사라진다. 남은 이차항은 Gram matrix $G\succeq0$와 NSD 행렬의 trace이므로 0 이하가 된다.

코드의 `cons_SDP[k]`는 첫 번째 식의 좌변이고, `cons_LIN[k]`는 두 번째 식의 좌변을 행벡터로 저장한다. `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`이 위 조건에 해당한다. 함수값을 0으로 두는 것이 아니라 함수값에 곱해지는 계수를 0으로 둔다.

$L=1$일 때 interpolation 항을 더하기 전의 행렬을 전개하면 다음과 같다.

$$
V_{k+1}-V_k=
\begin{pmatrix}
a_{k+1}-a_k&-a_{k+1}-c_k&c_{k+1}\\
-a_{k+1}-c_k&a_{k+1}-b_k&-c_{k+1}\\
c_{k+1}&-c_{k+1}&b_{k+1}
\end{pmatrix}.
$$

여기에 $\sum_{p,q\in I_k}\lambda_{p,q}M_{p,q}$를 더해 NSD 조건을 적용한다. $p=q$인 interpolation 항은 0이므로 코드에서는 제외한다. 두 함수값에 대한 선형 등식은 다음과 같이 풀어 쓸 수 있다.

$$
\begin{aligned}
0&=-d_k-\lambda_{\star,k}+\lambda_{k,\star}
+\lambda_{k,k+1}-\lambda_{k+1,k},\\
0&=d_{k+1}-\lambda_{\star,k+1}-\lambda_{k,k+1}
+\lambda_{k+1,\star}+\lambda_{k+1,k}.
\end{aligned}
$$

각 전이에서 1·2단계의 중간 계수 조건과 초기·최종 계수 조건을 적용한다.

## 6. SDP formulation in CVXPY

1·2단계의 계수 조건과 모든 전이의 5단계 부등식·등식을 함께 적용하여 $b_N$을 최대화한다. 모든 단계에서 step-size $\delta_k=1/L$을 사용하는 GD에 대해 potential 계수와 interpolation multiplier를 최적화한다.

아래 코드는 3·4단계에서 정의한 벡터와 행렬을 사용한다. 초기·최종 계수를 정한 뒤 각 전이에서 potential 차이와 interpolation 항을 계산하고, 이를 제약식에 넣는다. `cp`는 `cvxpy`, `np`는 `numpy`다.

```python
# Initial/final potential coefficients: choose one relax before building Q0 and QN.
a0, b0, c0, d0 = L**2, 0, 0, 0
tau = cp.Variable(name="tau")
aN, bN, cN, dN = 0, tau, 0, 0
if relax == 1:
    d0 = L
    dN = (2*N+1)*L
elif relax == 2:
    d0 = L
    dN = (2*N+1)*L
    aN = L**2
elif relax == 3:
    d0 = 0
    dN = 0

Q0 = cp.bmat([[a0, c0], [c0, b0]])
QN = cp.bmat([[aN, cN], [cN, bN]])
Q = {1: Q0, N+1: QN}
d = {1: d0, N+1: dN}
lam, cons_SDP, cons_LIN = {}, {}, {}
cons = [tau >= 0]

for k in range(1, N+1):
    if k < N:
        if relax == 0:
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:
            d[k+1] = (2*k+1)*L
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 2:
            d[k+1] = (2*k+1)*L
            Q[k+1] = cp.bmat([[L**2, 0], [0, cp.Variable(name=f"b_{k+1}")]])
        elif relax == 3:
            d[k+1] = 0
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")

    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    cons_SDP[k] = (-statesKG @ Q[k] @ statesKG.T
                   + statesK1G(k) @ Q[k+1] @ statesK1G(k).T)
    cons_LIN[k] = -d[k]*statesKF + d[k+1]*statesK1F

    X = {1: xs, 2: xk, 3: xk1(k)}
    G = {1: gxs, 2: gxk, 3: gxk1}
    F = {1: fxs, 2: fxk, 3: fxk1}
    for i in range(1, nbPts+1):
        for j in range(1, nbPts+1):
            if j != i:
                xi, xj = X[i], X[j]
                gi, gj = G[i], G[j]
                fi, fj = F[i], F[j]
                TT = np.vstack((xi, xj, gi, gj))
                cons_SDP[k] += lam[k][i-1, j-1]*(TT.T @ M @ TT)
                cons_LIN[k] += lam[k][i-1, j-1]*(fi-fj)
    cons.append(cons_SDP[k] << 0)
    cons.append(cons_LIN[k] == 0)
    cons.append(lam[k] >= 0)

obj = tau
problem = cp.Problem(cp.Minimize(-obj), cons)
```

코드의 `k=1`은 수식의 $0\to1$, `k=N`은 $N-1\to N$ 전이다. 마지막 전이에서는 중간 계수를 생성하지 않고 이미 지정한 `Q[N+1]`과 `d[N+1]`을 사용한다.

각 전이의 제약은 인접한 potential 계수를 공유한다. 따라서 100개 제약 묶음을 하나의 SDP로 구성하고 `cp.Minimize(-obj)`로 $b_N$을 최대화한다. objective와 제약 행렬의 각 원소는 결정변수에 대해 affine이다.

| `relax` | 중간 $Q_k$ 독립 원소 | 중간 $d_k$ | 선언된 $\lambda$ | $b_N$ | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 297 | 99 | 900 | 1 | 1297 |
| 1 | 297 | 0 | 900 | 1 | 1198 |
| 2 | 99 | 0 | 900 | 1 | 1000 |
| 3 | 297 | 0 | 900 | 1 | 1198 |

표의 개수는 코드에서 선언한 scalar 변수의 수이다. Symmetric 변수는 독립 원소만 세었고, `lam`은 interpolation에 쓰이지 않는 대각 원소까지 포함했다. 각 단계에서 3×3 NSD LMI 하나와 scalar linear equality 두 개가 생기므로, 100단계 전체에는 각각 100개와 200개의 제약이 있다.

`cons`는 전체 제약 목록이고, `obj=tau`는 objective이다. 이들로 CVXPY 문제 `problem`을 만든다.

## 7. MOSEK numerical optimization

기본 설정에서는 `solver="MOSEK"`, `tolerance=1e-8`을 사용한다. `MSK_DPAR_INTPNT_CO_TOL_PFEAS` 옵션으로 이 값을 MOSEK의 primal feasibility tolerance에 전달한다.

아래의 `solve`는 원본 Python 파일과 같은 디렉터리의 `saveData.py`에서 가져온 함수이다. MOSEK를 선택하면 `_MatlabOrderMOSEK` adapter가 변수 순서와 PSD 계수 전달 방식을 MATLAB/YALMIP에 맞춘다. 이때 canonicalization과 해 복원은 CVXPY가 그대로 담당하며, 다른 solver를 선택하면 일반 CVXPY 경로로 계산한다.

계산은 다음 순서로 진행된다.

1. CVXPY가 affine objective와 선형·PSD 제약을 cone problem으로 변환한다.
2. MOSEK가 이 유한 차원 SDP의 수치해를 계산한다.
3. CVXPY가 계산된 해를 원래 `cp.Variable`의 `.value`에 저장한다.

이 과정에서 얻는 것은 potential의 계수와 제약식에 쓰이는 multiplier이다. 특정 데이터셋에서 GD를 실행해 궤적을 구하는 실험은 아니며, 주어진 함수 클래스에 대해 potential inequality를 보장할 계수를 계산한다.

```python
solver = "MOSEK"
tolerance = 1e-8
solver_opt = {
    "solver": solver,
    "verbose": bool(verbose),
    "mosek_params": {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance},
}
solverDetails = solve(problem, **solver_opt)
if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    raise RuntimeError(f"The SDP returned {problem.status}; no solution can be plotted or saved.")
print(f"status={problem.status}, objective={obj.value:.10g}")
print(f"Elapsed time: {perf_counter()-t_start:.3f} seconds")
```

코드는 `problem.status`가 `optimal` 또는 `optimal_inaccurate`일 때 결과를 사용한다. 음수를 최소화하도록 문제를 정의했기 때문에 `problem.value`에는 $-b_N$이, `tau.value`에는 구하려던 $b_N$이 들어 있다.

각 CVXPY 결정변수의 수치해는 `.value`에 저장된다.

## 8. Optimal coefficients / algorithm parameters

각 상태의 수치 계수는 다음 코드 위치에서 추출한다. 아래 대응식의 코드 표현은 저장 위치를 나타내며, 실제 수치 추출에서는 CVXPY expression과 상수를 구분한다.

$$
a_k\leftrightarrow\texttt{Q[k+1][0,0]},\quad
b_k\leftrightarrow\texttt{Q[k+1][1,1]},\quad
c_k\leftrightarrow\texttt{Q[k+1][1,0]},\quad d_k\leftrightarrow\texttt{d[k+1]}
$$

일부 항은 처음부터 상수로 고정되어 있고, 나머지는 `cp.Expression`으로 표현되어 있다. 코드는 이 둘을 구분해 수치값을 가져온다.

```python
ak = np.zeros(N+1)
bk = np.zeros(N+1)
ck = np.zeros(N+1)
dk = np.zeros(N+1)
for i in range(1, N+2):
    ak[i-1] = float(Q[i][0, 0].value if isinstance(Q[i][0, 0], cp.Expression) else Q[i][0, 0])
    bk[i-1] = float(Q[i][1, 1].value if isinstance(Q[i][1, 1], cp.Expression) else Q[i][1, 1])
    ck[i-1] = float(Q[i][1, 0].value if isinstance(Q[i][1, 0], cp.Expression) else Q[i][1, 0])
    dk[i-1] = float(d[i].value if isinstance(d[i], cp.Expression) else d[i])
```

출력 배열에는 자유 계수의 수치해와 고정 계수의 지정값을 함께 담는다. 단계별 interpolation multiplier의 수치해는 `lam[k+1].value`에 있다. Step-size는 모든 단계에서 $\delta_k=1/L$을 사용하며, SDP의 해에서 얻는 것은 potential 계수와 multiplier이다.

8단계의 계수 배열은 길이 $N+1$이며 $k=0,\ldots,N$ 순서다. 원본 스크립트는 이 추출을 `if pplot:` 내부에서 실행하므로, 전체 계수 출력에는 `pplot=1`을 사용한다. 그림과 저장에 대한 플래그의 동작은 9단계를 따른다.

## 9. Plotting, numerical analysis

### Figure 1과 결과 저장

코드는 $a_k,b_k,c_k,d_k$를 각각 하나의 패널에 그린다. 논문 Figure 1의 갈색 실선, 빨간 점선, 파란 점선, 보라 점선은 각각 `relax=0,1,2,3`의 결과이다. Python 코드는 한 번 실행할 때 선택한 설정의 결과만 파란 선으로 그린다. 네 설정을 비교하려면 각 실행 결과를 같은 계수의 축에 겹쳐 그린다. 기본 저장 파일명은 같으므로, 각 결과를 별도 파일로 보관한다.

`pplot=1`이면 8단계의 계수를 추출하고 그래프를 그린다. 이 블록 안에서 `ssave=1`이면 [SaveData/GradientD.dat](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/1_Deterministic_SmoothConvex/SaveData/GradientD.dat)에 `k, ak, bk, ck, dk`를 저장한다. `pplot=0`이면 원본 스크립트의 추출·plotting·저장 블록 전체를 생략한다. 저장 파일과 그래프의 인덱스 1,…,101은 수식의 상태 인덱스 0,…,100에 대응한다.

### 논문에 보고된 값과의 비교

논문 p.6에서는 기본 SDP의 수치해가 $b_N^{\mathrm{opt}}\approx(N+1)^2$라고 보고한다. $N=100$을 대입하면 10201이므로, 코드를 실행해 얻은 최적값을 이 값과 비교할 수 있다. 코드에 10201을 미리 넣어 두고 계산하는 것은 아니다.

`relax=2`에 해당하는 해석적 수열은 $a_k=L^2,c_k=0,d_k=(2k+1)L,b_k=k(k+2)$이다. 따라서 이 설정의 결과는 $b_{100}=10200$과 비교할 수 있다. 다만 이 수열은 기본 `relax=0`과 초기·최종 계수 조건이 다르다. 논문에서는 `relax=3`도 함께 비교하여, 함수값 항을 제거하면 원하는 gradient norm rate를 얻기 어렵다는 점을 보여 준다.
