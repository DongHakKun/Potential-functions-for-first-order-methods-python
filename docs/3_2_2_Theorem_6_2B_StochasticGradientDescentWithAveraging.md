# 실험 3. SGD + Polyak–Ruppert Averaging — §3.2.2, Theorem 6

이 문서는 논문 §3.2.2, Theorem 6과 Appendix E.2의 averaging 실험을 설명한다. [B_StochasticGradientDescentWithAveraging.py](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/B_StochasticGradientDescentWithAveraging.py)를 따라 논문의 수식이 Python 코드로 어떻게 구현되는지 살펴본다.

1–5단계에서는 potential, 초기·최종 계수 조건, Gram 표현과 제약식을 설명한다. 6단계에서 이를 CVXPY로 구현하고, 7–9단계에서 SDP를 풀어 계수를 추출하고 결과를 확인한다.

수식의 상태 인덱스는 $k=0,\ldots,N$, 전이 인덱스는 $k=0,\ldots,N-1$이다. Python 딕셔너리는 1부터 시작하므로 $Q_k,d_k,a_k'$는 각각 `Q[k+1]`, `d[k+1]`, `ap[k+1]`에, $e_k,\lambda$는 `e[k+1]`, `lam[k+1]`에 대응한다. 코드 블록의 루프 변수 `k`는 원본과 같이 1부터 시작한다.

논문의 iterate는 $x_k$와 $z_k$로 쓰고, Python 식별자는 `Q[k]`처럼 표시한다. Interpolation point의 인덱스는 $p,q$로 쓴다. 아래 3–6단계는 원본 Python 코드의 SDP 구성을 설명한다.

## 1. Parameterized Potential

각 전이의 outcome $i\in\{1,\ldots,n\}$는 uniform distribution을 사용한다. 따라서 $\mathbb E_i$는 $1/n$을 곱한 outcome 합으로 구현한다.

$f\in\mathcal F_L$이고, oracle은 unbiased 조건 $\mathbb E_iG(x;i)=f'(x)$와 uniformly bounded variance 조건 $\mathbb E_i\|G(x;i)-f'(x)\|^2\le\sigma^2$를 만족한다고 가정한다. 이 실험에서는 SGD iterate $x_k$와 averaged iterate $z_k$를 함께 다루므로, potential에도 두 점에 관한 항을 둔다.

$$
\begin{aligned}
\phi_k^f={}&
\begin{pmatrix}z_k-x_\star\\f'(z_k)\end{pmatrix}^{\!\top}
(Q_k\otimes I_d)
\begin{pmatrix}z_k-x_\star\\f'(z_k)\end{pmatrix}
+d_k(f(z_k)-f_\star)+a_k'\|x_k-x_\star\|^2,\\
Q_k={}&\begin{pmatrix}a_k&c_k\\c_k&b_k\end{pmatrix}.
\end{aligned}
$$

| 수식의 기호 | Python 표현 | 의미 |
| --- | --- | --- |
| $L,\mu,n,N$ | `L=1`, `m=0`, `n=2`, `N=100` | 기본 실험 설정 |
| $a_k,b_k,c_k$ | `Q[k+1]`의 `[0,0]`, `[1,1]`, `[1,0]` | $z_k,f'(z_k)$에 대한 이차항 |
| $a_k'$ | `ap[k+1]` | $\|x_k-x_\star\|^2$의 계수 |
| $d_k$ | `d[k+1]` | $f(z_k)-f_\star$의 계수 |
| $e_k$ | `e[k+1]` | variance에 따른 오차의 계수 |
| $d_N$ | `tau` | 첫 번째 SDP에서 최대화할 계수 |

`relax`는 0 또는 1이며 기본값은 0이다. 중간 상태 $k=1,\ldots,N-1$에서 자유롭게 정할 계수와 고정할 계수는 다음과 같다. $Q_k$는 두 설정 모두 대칭 행렬이며, 별도의 PSD 제약은 없다.

| `relax` | $Q_k$ | $a'_k$ | $d_k$ |
| --- | --- | --- | --- |
| 0 | 대칭 2×2 decision variable | decision variable | decision variable |
| 1 | $0$으로 고정 | $L/2$로 고정 | decision variable |

두 설정 모두 $e_k,\lambda$를 최적화한다. 초기·최종 계수는 2단계에서 지정한다. 이차항을 전개하면 다음과 같다.

$$
a_k\|z_k-x_\star\|^2+b_k\|f'(z_k)\|^2
+2c_k\langle f'(z_k),z_k-x_\star\rangle
+a_k'\|x_k-x_\star\|^2
$$

`Q`의 $a_k$는 averaged iterate $z_k$의 거리항에, 별도의 `ap`에 저장한 $a_k'$는 SGD iterate $x_k$의 거리항에 곱해진다. 같은 거리항 계수라도 서로 다른 점에 적용된다.

```python
alpha = 0
delta = lambda k: 1 / L / (k + 1) ** alpha  # Step-size (possibly varying function of k).
```

## 2. Potential Inequality

한 단계에서는 먼저 SGD로 $x_k$를 갱신한 뒤, 새로 얻은 점을 기존 평균 $z_k$와 합쳐 $z_{k+1}$을 구한다.

$$
\begin{aligned}
x_{k+1}^{(i)}&=x_k-\delta_kG(x_k;i),\\
z_{k+1}^{(i)}&=\frac1{k+1}x_{k+1}^{(i)}+\frac{k}{k+1}z_k.
\end{aligned}
$$

```python
xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gxk
zk1 = lambda k: xk1(k) / k + (k - 1) / k * zk
```

코드의 `k=1`에서는 `zk1(1)=xk1(1)`이므로 첫 averaged iterate가 첫 SGD iterate와 같다. 기본 설정은 `alpha=0`이어서 $\delta_k=1/L=1$이다.

수식의 $\delta_k$는 코드의 `delta(k+1)`에 대응한다. `alpha`를 바꾸면 그 값은 $1/[L(k+2)^\alpha]$가 되므로, 논문과 비교할 때 이 인덱스 차이를 반영한다.

두 iterate가 stochastic outcome에 따라 달라지므로, 한 단계의 potential inequality는 expectation에 대해 쓴다.

$$
\boxed{\mathbb E_i[\phi_{k+1}^f(x_{k+1}^{(i)},z_{k+1}^{(i)})]
\le\phi_k^f(x_k,z_k)+e_k\sigma^2}.
$$

공통 초기조건은 $Q_0=0$, $d_0=0$, $a'_0=L/2$이다. 최종 함수값 계수는 $d_N=\texttt{tau}\ge0$으로 두고, 최종 거리항은 `relax` 값에 따라 다음과 같이 달라진다.

| `relax` | $Q_N$ | $a'_N$ | 최종 potential $\phi_N^f$ |
| --- | --- | --- | --- |
| 0 | $0$ | $0$ | $d_N(f(z_N)-f_\star)$ |
| 1 | $0$ | $L/2$ | $d_N(f(z_N)-f_\star)+(L/2)\|x_N-x_\star\|^2$ |

각 단계의 부등식에 expectation을 취하고 더하면 다음 telescoping 부등식을 얻는다. $a'_N$에는 위 표의 값을 사용한다.

$$
d_N\mathbb E[f(z_N)-f_\star]
+a'_N\mathbb E\|x_N-x_\star\|^2
\le\frac L2\|x_0-x_\star\|^2+\sigma^2\sum_{k=0}^{N-1}e_k.
$$

`relax=0`은 최종 거리항이 0이고, `relax=1`은 비음수인 최종 거리항을 포함한다. 함수값만의 상한을 표시할 때에는 두 설정 모두 그 항을 제외하여 다음 식을 사용한다.

$$
d_N\mathbb E[f(z_N)-f_\star]
\le\frac L2\|x_0-x_\star\|^2+\sigma^2\sum_{k=0}^{N-1}e_k.
$$

초기·최종 계수 조건의 Python 대응은 `a0=b0=c0=d0=0`, `ap0=L/2`, `aN=bN=cN=0`, `dN=tau`이며, `apN`은 위 표에 따라 0 또는 `L/2`이다. 이 선택을 마친 뒤 6단계에서 `Q`, `d`, `ap`의 초기·최종 원소를 구성한다.

## 3. Finite-dimensional Gram representation

논문 Appendix E.2는 부등식의 가중합으로 정리를 증명한다. 아래 $P,F$의 배치와 배열 구성은 원본 Python 코드의 주석과 연산을 따른다. 일반성을 잃지 않고 $x_\star=f_\star=0$으로 두며, 최적점에서 $f'(x_\star)=0$이다.

$$
\begin{aligned}
P&=[x_k\ z_k\ f'(z_k)\ G(x_k;1)\ \cdots\ G(x_k;n)\ f'(z_{k+1}^{(1)})\ \cdots\ f'(z_{k+1}^{(n)})],\\
F&=[f(z_k)\ f(x_k)\ f(z_{k+1}^{(1)})\ \cdots\ f(z_{k+1}^{(n)})],\\
G&=P^\top P\in\mathbb S_+^{3+2n}.
\end{aligned}
$$

$n=2$에서는 벡터 7개와 함수값 4개를 사용하므로 `dimG=7`, `dimF=4`이고, interpolation point는 다섯 개이다.

```python
dimG = 3 + 2 * n
dimF = n + 2
nbPts = 3 + n
```

`xk`, `zk`, `gzk`는 각각 $P$의 첫 번째, 두 번째, 세 번째 열을 고른다. `Gxk`의 각 행은 oracle을 고르고, `gxk`는 그 평균으로 $f'(x_k)$를 나타낸다. `gzk1`의 각 행은 다음 averaged iterate의 gradient를 고른다.

```python
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
```

`fzk`, `fxk`, `fzk1`은 위 $F$의 순서대로 함수값을 고른다. `fzk1`과 `gzk1`은 같은 outcome 순서를 사용한다.

```python
fzk = np.zeros(dimF)
fzk[0] = 1  # f(z_k).
fxk = np.zeros(dimF)
fxk[1] = 1  # f(x_k).
# The MATLAB allocation says fxk1, but the following assignment uses fzk1.
fzk1 = np.zeros((n, dimF))
fzk1[:, 2:n+2] = np.eye(n)  # f(z_{k+1}^{(i)}).
fxs = np.zeros(dimF)  # f(x*) = 0.
```

`statesKG`는 `zk-xs`와 `gzk`를 두 열로 모은 7×2 배열이다. `statesKG @ Q[k] @ statesKG.T`로 averaged iterate의 이차항을 계산하고, `ap[k] * np.outer(xk-xs, xk-xs)`로 SGD iterate의 거리항을 따로 계산한다. 다음 단계에서는 `ZK1`과 `gzk1`으로 averaged iterate의 항을, `XK1`으로 SGD iterate의 거리항을 구한 뒤 각각 outcome별로 평균 낸다.

```python
statesKG = np.column_stack((zk - xs, gzk))
statesKF = fzk
```

Gram matrix 자체를 decision variable로 선언하지 않고, dualization으로 얻은 LMI 조건의 계수를 구한다. 코드의 `G`, `F`는 interpolation용 딕셔너리이며 위 수식의 Gram matrix와 함수값 행렬과 구분한다.

## 4. Interpolation matrices + multipliers

Appendix D.4의 smooth convex interpolation 조건을 사용한다.

$$
f(x_p)-f(x_q)-\langle f'(x_q),x_p-x_q\rangle
-\frac1{2L}\|f'(x_p)-f'(x_q)\|^2\ge0.
$$

원본 코드에서는 $\{x_\star,x_k,z_k,z_{k+1}^{(1)},\ldots,z_{k+1}^{(n)}\}$의 점들에 적용한다. 각 점의 위치·gradient·함수값 배열을 `X`, `G`, `F`의 같은 키에 넣는다.

`XK1`과 `ZK1`은 각각 다음 SGD iterate와 averaged iterate의 배열이다. Interpolation에는 `ZK1`의 점들을 넣고 `XK1`의 점들은 넣지 않는다. $n=2$에서는 서로 다른 순서쌍 20개를 사용한다. $F$의 함수값 순서와 딕셔너리의 점 번호는 별개이므로 `X`, `G`, `F`에서 같은 키를 함께 가져온다.

`TT = np.vstack((xi, xj, gi, gj))`로 네 벡터를 쌓고, `TT.T @ M @ TT`로 interpolation의 이차항을 계산한다. 함수값 계수는 `fi-fj`이다. 같은 `lam[k][i-1,j-1]`을 두 항에 곱해 각각 `cons_SDP[k]`, `cons_LIN[k]`에 더한다. 같은 점끼리의 쌍은 `if j != i`에서 제외하며, `lam[k] >= 0`은 원소별 부호 조건이다.

행렬 `M`은 다음과 같다. 기본 설정 `m=0`에서는 논문 식 (19)의 행렬과 일치한다.

```python
M = np.array([
    [-L*m, L*m, m, -L],
    [L*m, -L*m, -m, L],
    [m, -m, -1, 1],
    [-L, L, 1, -1],
]) / (2 * (L - m))
```

Variance 조건에는 $x_k$에서 호출하는 `Gxk`와 그 평균 `gxk`의 차이를 사용한다. Appendix E.7의 $A_{\mathrm{var}}$를 계산하는 방식과 같으며, 코드의 `Avar`에 저장한다.

```python
Avar = np.zeros((dimG, dimG))
for i in range(1, n + 1):
    Avar = Avar + np.outer(Gxk[i-1] - gxk, Gxk[i-1] - gxk) / n
```

이 행렬과 Gram matrix의 trace는 oracle의 variance이므로 상한이 $\sigma^2$이다. 제약식에는 `-e[k] * Avar`를 더하고 `e[k] >= 0`을 적용한다.

## 5. Linear equality + semidefinite inequality

다음 potential의 expectation에서 현재 potential을 빼고, interpolation 부등식과 variance 조건의 가중합을 더한다. 원본 코드의 `cons_SDP[k]`에는 이차항 행렬을, `cons_LIN[k]`에는 함수값에 곱해지는 계수를 모은다.

다음 averaged iterate와 SGD iterate의 항은 모두 outcome별로 계산해 평균 낸다. 함수값 계수의 순서는 $f(z_k),f(x_k),f(z_{k+1}^{(1)}),f(z_{k+1}^{(2)})$이다. Potential에는 $f(x_k)$가 없지만 interpolation에는 있으므로 이 함수값의 계수도 소거한다.

$n=2$에서 `cons_SDP[k]`는 7×7 대칭행렬이고 `cons_LIN[k]`는 함수값 4개에 대응하는 행벡터이다. 모든 항을 더한 뒤 `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`, `e[k] >= 0`을 적용한다.

선형 등식으로 함수값 항을 소거하고, 남은 이차항이 모든 Gram matrix에 대해 0 이하가 되도록 NSD 조건을 둔다. Interpolation 항은 비음수이고 oracle의 variance는 $\sigma^2$ 이하이므로 2단계의 potential inequality가 성립한다. 6단계에 이 계산과 제약 등록 코드가 나와 있다.

## 6. SDP formulation in CVXPY

### objective와 `reoptimize` 분기

모든 `relax`에서 1차 SDP는 $\max d_N$을 푼다. `reoptimize=0`이면 이 해를 사용한다. `reoptimize=1`이면 첫 번째 문제의 objective `obj=-tau`에서 얻은 `-float(obj.value)`를 사용해 `tau >= -float(obj.value) - tol_reopt`를 추가하고, 같은 제약 아래에서 `accumulated_ek`를 최소화한다. 기본값은 `reoptimize=1`, `tol_reopt=1e-5`이다. 두 번째 SDP에서는 이 하한을 유지하면서 나머지 potential 계수와 multiplier를 다시 최적화한다.

| `relax` | 중간 상태 $k=1,\ldots,N-1$ | 최종 potential |
| --- | --- | --- |
| 0 | $a_k,b_k,c_k,d_k,a'_k$ 자유 | $d_N(f(z_N)-f_\star)$ |
| 1 | $Q_k=0,\ a'_k=L/2$, $d_k$ 자유 | $d_N(f(z_N)-f_\star)+(L/2)\|x_N-x_\star\|^2$ |

### 초기·최종 계수 조건을 적용한 뒤 모든 decision variable 선언

초기·최종 계수를 먼저 정하고, 중간 계수와 각 전이의 $\lambda,e_k$를 선언한다. 중간 계수를 생성하는 루프는 $k=1,\ldots,N-1$만 다룬다.

```python
a0 = b0 = c0 = d0 = 0
ap0 = L / 2
tau = cp.Variable(name="tau")
aN = bN = cN = 0
dN = tau
apN = 0 if relax == 0 else L / 2

Q0 = np.array([[a0, c0], [c0, b0]])
QN = np.array([[aN, cN], [cN, bN]])
q0 = d0
qN = dN
Q = {1: Q0, N+1: QN}
d = {1: q0, N+1: qN}
ap = {1: ap0, N+1: apN}
for k in range(1, N):
    if relax == 0:
        d[k+1] = cp.Variable(name=f"d_{k+1}")
        ap[k+1] = cp.Variable(name=f"ap_{k+1}")
        Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
    elif relax == 1:
        d[k+1] = cp.Variable(name=f"d_{k+1}")
        ap[k+1] = L / 2
        Q[k+1] = np.zeros((2, 2))

lam = {}
e = {}
for k in range(1, N + 1):
    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    e[k] = cp.Variable(name=f"e_{k}")
```

위 선언은 원본의 `if k < N` 안에서 수행하는 선언과 같은 계수·이름·자유도를 사용한다.

### Potential 차이의 초기화, interpolation 가산, 제약 등록

다음 코드는 앞서 정의한 벡터와 행렬을 사용한다. 각 전이에서 현재 potential을 빼고 다음 potential의 outcome 평균을 더한 뒤 interpolation 항을 더한다. 완성된 행렬과 벡터만 `cons`에 등록한다.

```python
cons_SDP = {}
cons_LIN = {}
cons = [tau >= 0]
for k in range(1, N + 1):
    XK1 = xk1(k)
    ZK1 = zk1(k)
    X = {1: xs, 2: xk, 3: zk}
    G = {1: gxs, 2: gxk, 3: gzk}
    F = {1: fxs, 2: fxk, 3: fzk}
    for i in range(1, n + 1):
        X[3+i] = ZK1[i-1]
        G[3+i] = gzk1[i-1]
        F[3+i] = fzk1[i-1]

    cons_SDP[k] = -statesKG @ Q[k] @ statesKG.T - ap[k] * np.outer(xk-xs, xk-xs) - e[k] * Avar
    cons_LIN[k] = -d[k] * statesKF
    for i in range(1, n + 1):
        statesK1G = np.column_stack((ZK1[i-1] - xs, gzk1[i-1]))
        statesK1F = fzk1[i-1]
        cons_SDP[k] = cons_SDP[k] + statesK1G @ Q[k+1] @ statesK1G.T / n
        cons_SDP[k] = cons_SDP[k] + ap[k+1] * np.outer(XK1[i-1]-xs, XK1[i-1]-xs) / n
        cons_LIN[k] = cons_LIN[k] + statesK1F * d[k+1] / n

    for i in range(1, nbPts + 1):
        for j in range(1, nbPts + 1):
            if j != i:
                xi, xj = X[i], X[j]
                gi, gj = G[i], G[j]
                fi, fj = F[i], F[j]
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
```

코드의 `k=1`은 수식의 전이 $0\to1$이고, `k=N`은 $N-1\to N$이다. `statesKG`, `statesKF`는 3단계의 배열이며, `X`, `G`, `F`는 4단계에서 설명한 점 순서를 사용한다. 위 코드가 5단계의 NSD·선형 등식·부호 조건과 $d_N\ge0$ 및 `relax`별 계수 조건을 함께 등록한다.

### 문제 규모

$N=100,n=2$에서 7×7 NSD LMI는 100개이고, 함수값 계수의 scalar 등식은 $4\times100=400$개이다. 설정별 독립 scalar decision variable 수는 다음과 같다.

| `relax` | 중간 $Q_k$ | 중간 $d_k$ | 중간 $a'_k$ | $\lambda$ | $e_k$ | $d_N$ | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 297 | 99 | 99 | 2500 | 100 | 1 | 3096 |
| 1 | 0 | 99 | 0 | 2500 | 100 | 1 | 2700 |

이 수에는 interpolation 합에 사용하지 않는 `lam`의 대각 원소도 포함한다. 1차 SDP의 $e_k$는 제약식을 만족하도록 정해지며, $\sum e_k$를 objective로 최소화하는 것은 `reoptimize=1`의 2차 SDP이다.

## 7. MOSEK numerical optimization

기본 설정은 `solver="MOSEK"`, `tolerance=1e-8`이다. 첫 번째 SDP를 풀 때 이 값을 `MSK_DPAR_INTPNT_CO_TOL_PFEAS` 옵션으로 전달해 MOSEK의 primal feasibility tolerance를 지정한다.

아래의 `solve`는 원본 Python 파일과 같은 디렉터리의 `saveData.py`에서 가져온 함수이다. MOSEK를 선택하면 `_MatlabOrderMOSEK` adapter가 변수 순서와 PSD 계수 전달 방식을 MATLAB/YALMIP에 맞춘다. 이때 canonicalization과 해 복원은 CVXPY가 그대로 담당하며, 다른 solver를 선택하면 일반 CVXPY 경로로 계산한다.

계산은 다음 순서로 진행된다.

1. CVXPY가 affine objective와 선형·PSD 제약을 cone problem으로 변환한다.
2. MOSEK가 이 유한 차원 SDP의 수치해를 계산한다.
3. CVXPY가 계산된 해를 원래 `cp.Variable`의 `.value`에 저장한다.

이 계산에서 구하는 것은 potential의 계수와 제약식을 구성하는 multiplier이다. 특정 데이터셋에 SGD를 실행해 iterate의 궤적을 얻는 과정은 포함되지 않는다.

```python
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
```

`reoptimize`와 무관하게 첫 번째 호출은 `cp.Minimize(-tau)`를 푼다. `reoptimize=1`인 경우에만 두 번째 호출에서 `cp.Minimize(accumulated_ek)`를 푼다. 두 결과를 비교하려면 두 번째 호출 전에 첫 번째 계산에서 얻은 `tau.value`를 따로 보관한다.

두 번째 호출에서는 `solver_opt`를 새로 만들면서 MOSEK의 `mosek_params`를 다시 지정하지 않는다. 따라서 첫 번째 호출에 사용한 `tolerance=1e-8`은 두 번째 호출에 그대로 이어지지 않는다. `reoptimize=0`이면 첫 번째 계산에서 끝나며, 누적 오차를 최소화하는 두 번째 계산은 생략된다.

## 8. Optimal coefficients / algorithm parameters

`reoptimize=0`이면 1차 해, `reoptimize=1`이면 2차 해를 최종 해로 사용한다. 원본과 같이 `pplot=1`인 실행에서 아래 계수 추출을 수행한다. `pplot=0`이면 배열 추출과 9단계의 그림·파일 출력을 생략한다. free variable은 `.value`를 읽고, 고정 계수는 상수 자체를 읽는다. Multiplier는 최종 `lam[k].value`에 있으며 저장 열에는 포함하지 않는다.

```python
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
```

기본 설정 `alpha=0, L=1`에서는 모든 단계에서 SGD step-size $\delta_k=1/L=1$을 사용한다. 새 SGD iterate와 기존 평균의 가중치는 각각 $1/(k+1)$과 $k/(k+1)$이다. SDP는 이 갱신식의 potential 계수를 구하며, `tau.value`는 최종 함수값 계수 $d_N$이다.

`Q[i][1,0]`은 $c_{i-1}$이며, potential의 교차항은 $2c_{i-1}\langle f'(z_{i-1}),z_{i-1}-x_\star\rangle$이다. `apk[i-1]`은 $x_{i-1}$의 거리 계수 $a_{i-1}'$이며, $z_{i-1}$의 거리 계수를 담은 `ak`와 구분된다. `relax=1`처럼 계수를 상수로 고정한 경우에는 `.value` 대신 상수 자체를 읽는다.

## 9. Plotting, numerical analysis

### 출력 설정과 실행

출력 관련 변수는 `pplot`, `ssave`, `folder`, `nname`이며 기본값은 각각 `1`, `1`, `"SaveData/"`, `"SGD_Averaging.dat"`이다. `pplot`과 `ssave`에 따른 실행 범위는 다음과 같다.

| `pplot` | `ssave` | 8–9단계의 실행 |
| --- | --- | --- |
| 0 | 0 또는 1 | 계수 배열 추출, plotting, 저장을 모두 생략 |
| 1 | 0 | 계수 배열 추출과 plotting 수행, 파일 저장 생략 |
| 1 | 1 | 계수 배열 추출, plotting, 파일 저장 수행 |

각 `relax` 값에 대해 SDP를 따로 푼다. 결과를 함께 보관하려면 `folder` 또는 `nname`을 다르게 설정한다. `relax` 값을 바꾸면 초기·최종 계수 조건과 decision variable도 다시 구성한다.

### 계수 출력과 저장

코드는 $a_k,b_k,c_k,d_k,e_k$, 누적합, $a_k'$를 일곱 패널로 출력한다. `pplot=1` 안에서 계수를 추출하고, `ssave=1`이면 [SaveData/SGD_Averaging.dat](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/SaveData/SGD_Averaging.dat)에 `k, ak, bk, ck, dk, ek, ekc, apk`를 저장한다.

논문의 단계 $k$는 저장 파일의 인덱스 열에서 $k+1$로 표시된다. `ek[0]=0`은 표시용이고 `ek[k+1]=e_k`, `ekc[k]=\sum_{t=0}^{k-1}e_t`이다. 논문 Figure 2는 averaging을 적용하지 않은 SGD의 결과이며, 이 파일에서 그리는 그래프는 averaging을 적용한 실험의 결과이다.

### Theorem 6과 함수값 계수의 대응

이 비교에서 Theorem 6 원문의 $d_k$는 averaging weight를 정하는 보조 수열이다. 본문 1–8단계와 Python `d[k+1]`의 $d_k$는 potential의 함수값 계수이다. 두 의미는 이 절에서 구분하여 대응시키며, 본문의 기호 정의는 바꾸지 않는다. 코드에 맞춰 그 보조 수열의 초기값을 0으로 두면 값이 $k$가 되므로, 이를 정리의 식에 대입한다.

그러면 averaging 식은 코드와 같은 $z_{k+1}^{(i)}=x_{k+1}^{(i)}/(k+1)+kz_k/(k+1)$이 된다. 정리의 부등식에서 현재와 다음 함수값의 계수는 각각 $\delta_kLk$, $\delta_kL(k+1)$이다.

Step-size가 감소하면 다음 함수값의 계수를 $\delta_{k+1}L(k+1)$로 줄여도 부등식이 유지된다. 따라서 다음 단계와 연결할 수 있으며, 이 문서의 potential 표기로는 다음 계수 선택에 해당한다.

$$
Q_k=0,\qquad a_k'=L/2,\qquad d_k=\delta_kLk
$$

기본 step-size $\delta_k=1/L$에서는 $d_k=k$이다. 같은 초기값을 정리의 오차 계수와 step-size 조건에 대입하면 다음과 같다.

$$
e_k=\frac{\delta_k^2L(1+k+L\delta_k)}{2(1+k)},\qquad
0\le\delta_k\le\frac{1+\sqrt5}{2L},\qquad\delta_{k+1}\le\delta_k.
$$

이 선택은 $Q_k=0$, $a_k'=L/2$를 사용하는 `relax=1`과 연결된다. 기본 `relax=0`은 더 넓은 potential family를 탐색하고 최종 거리항도 제거하므로, 위 수열을 기본 설정의 수치 최적해로 보지는 않는다.

논문 Table 2에서 averaging의 최적 decay exponent는 $\alpha=1/2$이다. 이 코드의 기본값은 $\alpha=0$이므로, 기본 실행 결과는 Table 2의 최적 decay policy를 사용한 결과와 구분해야 한다.

### Plotting 배열과 저장 열의 대응

| 패널 | Python 배열 | 논문의 양 |
| --- | --- | --- |
| 1–4 | `ak,bk,ck,dk` | averaged iterate에 대한 계수 $a_k,b_k,c_k,d_k$ |
| 5 | `ek` | 각 단계의 variance 오차 계수 |
| 6 | `ekc` | 상태 $k$까지의 누적 오차 |
| 7 | `apk` | SGD iterate의 거리 계수 $a_k'$ |

파일에는 아래 `labels`에 나열된 순서로 각 배열을 저장한다.

```python
if pplot:
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
```
