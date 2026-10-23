# 실험 4. SGD + Primal Averaging — §3.2.2, Theorem 7

이 문서는 논문 §3.2.2, Theorem 7과 Appendix E.3의 primal averaging 실험을 설명한다. [C_StochasticGradientDescentWithPrimalAveraging.py](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/C_StochasticGradientDescentWithPrimalAveraging.py)를 따라 논문의 수식이 Python 코드로 어떻게 구현되는지 살펴본다.

1–5단계에서는 potential, 초기·최종 계수 조건, Gram 표현과 제약식을 설명한다. 6단계에서 이를 CVXPY로 구현하고, 7–9단계에서 SDP를 풀어 계수를 추출하고 결과를 확인한다.

수식의 상태 인덱스는 $k=0,\ldots,N$, 전이 인덱스는 $k=0,\ldots,N-1$이다. Python 딕셔너리는 1부터 시작하므로 $Q_k,d_k,a_k'$는 각각 `Q[k+1]`, `d[k+1]`, `ap[k+1]`에, $e_k,\lambda$는 `e[k+1]`, `lam[k+1]`에 대응한다. 코드 블록의 루프 변수 `k`는 원본과 같이 1부터 시작한다.

논문의 iterate는 $x_k$와 $y_k$로 쓰고, Python 식별자는 `Q[k]`처럼 표시한다. Interpolation point의 인덱스는 $p,q$로 쓴다. 아래 3–6단계는 원본 Python 코드의 SDP 구성을 설명한다.

## 1. Parameterized Potential

각 전이의 outcome $i\in\{1,\ldots,n\}$는 uniform distribution을 사용한다. 따라서 $\mathbb E_i$는 $1/n$을 곱한 outcome 합으로 구현한다.

$f\in\mathcal F_L$이고 oracle $G(y;i)$가

$$
\mathbb E_iG(y;i)=f'(y),\qquad
\mathbb E_i\|G(y;i)-f'(y)\|^2\le\sigma^2
$$

를 만족한다고 가정한다. 이 실험에서 계수를 찾을 potential의 형태는 다음과 같다.

$$
\begin{aligned}
\phi_k^f={}&
\begin{pmatrix}y_k-x_\star\\f'(y_k)\end{pmatrix}^{\!\top}
(Q_k\otimes I_d)
\begin{pmatrix}y_k-x_\star\\f'(y_k)\end{pmatrix}
+d_k(f(y_k)-f_\star)+a_k'\|x_k-x_\star\|^2,\\
Q_k={}&\begin{pmatrix}a_k&c_k\\c_k&b_k\end{pmatrix}.
\end{aligned}
$$

| 수식의 기호 | Python 표현 | 의미 |
| --- | --- | --- |
| $L,\mu,n,N$ | `L=1`, `m=0`, `n=2`, `N=100` | 기본 실험 설정 |
| $a_k,b_k,c_k$ | `Q[k+1]`의 `[0,0]`, `[1,1]`, `[1,0]` | $y_k,f'(y_k)$에 대한 이차항 |
| $a_k'$ | `ap[k+1]` | $\|x_k-x_\star\|^2$의 계수 |
| $d_k$ | `d[k+1]` | recurrence로 정해지는 함수값 계수 |
| $e_k$ | `e[k+1]` | variance에 곱해지는 오차 계수 |

`relax`는 0 또는 1이며 기본값은 0이다. 중간 상태 $k=1,\ldots,N-1$에서 자유롭게 정할 계수와 고정할 계수는 다음과 같다. $Q_k$는 두 설정 모두 대칭 행렬이며, 별도의 PSD 제약은 없다.

| `relax` | $Q_k$ | $a'_k$ | $d_k$ |
| --- | --- | --- | --- |
| 0 | 대칭 2×2 decision variable | decision variable | $d_0=0$, $d_{k+1}=d_k+\delta_kL$로 고정 |
| 1 | $0$으로 고정 | $L/2$로 고정 | 같은 recurrence로 고정 |

두 설정 모두 $e_k,\lambda$를 최적화한다. 초기·최종 계수는 2단계에서 지정한다. $d_k$는 recurrence로 정해지므로 decision variable이 아니다. 최종 계수를 최대화하기 위한 `tau`와 2차 최적화용 `reoptimize`도 사용하지 않는다.

이차항을 전개하면 $a_k\|y_k-x_\star\|^2+b_k\|f'(y_k)\|^2+2c_k\langle f'(y_k),y_k-x_\star\rangle+a_k'\|x_k-x_\star\|^2$이다. `Q`는 averaged iterate $y_k$와 그 gradient에 관한 계수를 담고, `ap`는 SGD iterate $x_k$의 거리항 계수 $a_k'$를 담는다.

```python
alpha = 0
delta = lambda k: 1 / L / (k + 1) ** alpha  # Step-size (possibly varying function of k).
```

## 2. Potential Inequality

Primal averaging에서는 먼저 $x_k$와 $y_k$를 평균 내고, 그 점에서 oracle을 호출해 $x_k$를 갱신한다. Theorem 7의 갱신식은 다음과 같다.

$$
\begin{aligned}
y_{k+1}&=\frac{d_k}{d_k+\delta_kL}y_k
+\frac{\delta_kL}{d_k+\delta_kL}x_k,\\
x_{k+1}^{(i)}&=x_k-\delta_kG(y_{k+1};i),\\
d_{k+1}&=d_k+\delta_kL.
\end{aligned}
$$

```python
yk1 = lambda dk, k: delta(k) * L / (dk + delta(k) * L) * xk + dk / (dk + delta(k) * L) * yk
xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gyk1
```

기본 설정에서는 $\delta_k=1/L=1$, $d_0=0$이므로 $d_k=k$이다. 이를 averaging 식에 대입하면 다음과 같다.

$$
y_{k+1}=\frac{k}{k+1}y_k+\frac1{k+1}x_k.
$$

$y_{k+1}$는 oracle 호출 전에 정해지므로 stochastic outcome에 의존하지 않는다. 그 점에서 얻는 oracle 값에 따라 $x_{k+1}^{(i)}$가 달라진다.

이 갱신에 대한 potential inequality는 다음과 같다. 다음 상태 중 $x_{k+1}^{(i)}$에만 outcome이 표시되는 것은 앞의 계산 순서 때문이다.

$$
\boxed{\mathbb E_i[\phi_{k+1}^f(x_{k+1}^{(i)},y_{k+1})]
\le\phi_k^f(x_k,y_k)+e_k\sigma^2}.
$$

공통 초기조건은 $Q_0=0$, $d_0=0$, $a'_0=L/2$이다. $d_N$은 앞의 recurrence로 정하며, 최종 거리항은 `relax` 값에 따라 다음과 같이 달라진다.

| `relax` | $Q_N$ | $a'_N$ | 최종 potential $\phi_N^f$ |
| --- | --- | --- | --- |
| 0 | $0$ | $0$ | $d_N(f(y_N)-f_\star)$ |
| 1 | $0$ | $L/2$ | $d_N(f(y_N)-f_\star)+(L/2)\|x_N-x_\star\|^2$ |

각 단계의 부등식에 expectation을 취하고 더하면 다음 telescoping 부등식을 얻는다. $a'_N$에는 위 표의 값을 사용한다.

$$
d_N\mathbb E[f(y_N)-f_\star]
+a'_N\mathbb E\|x_N-x_\star\|^2
\le\frac L2\|x_0-x_\star\|^2+\sigma^2\sum_{k=0}^{N-1}e_k.
$$

함수값만의 상한은 비음수인 최종 거리항을 제외한 다음 식으로 표시한다.

$$
d_N\mathbb E[f(y_N)-f_\star]
\le\frac L2\|x_0-x_\star\|^2+\sigma^2\sum_{k=0}^{N-1}e_k.
$$

$d_N$은 이미 정해져 있으므로 두 설정 모두 $\sum e_k$를 최소화한다. 초기·최종 계수 조건의 Python 대응은 `a0=b0=c0=d0=0`, `ap0=L/2`, `aN=bN=cN=0`이며, `apN`은 위 표에 따라 0 또는 `L/2`이다. 별도의 `dN`이나 `tau`를 선언하지 않고 6단계에서 `d[N+1]`까지 recurrence로 계산한다.

기본 설정인 `alpha=0`에서는 모든 step-size가 1이다. `alpha`를 바꾸면 코드가 `delta(1)`부터 호출하므로, 첫 step-size는 $1/(L2^\alpha)$가 된다.

## 3. Finite-dimensional Gram representation

논문 Appendix E.3는 부등식의 가중합으로 정리를 증명한다. 아래 $P,F$의 배치와 배열 구성은 원본 Python 코드의 주석과 연산을 따른다. 일반성을 잃지 않고 $x_\star=f_\star=0$으로 두며, 최적점에서 $f'(x_\star)=0$이다.

$$
\begin{aligned}
P&=[x_k\ y_k\ f'(y_k)\ G(y_{k+1};1)\ \cdots\ G(y_{k+1};n)],\\
F&=[f(y_k)\ f(y_{k+1})],\qquad G=P^\top P\in\mathbb S_+^{3+n}.
\end{aligned}
$$

$n=2$에서는 벡터 5개와 함수값 2개를 사용하므로 `dimG=5`, `dimF=2`이고, interpolation point는 세 개이다.

```python
dimG = 3 + n
dimF = 2
nbPts = 3
```

`xk`, `yk`, `gyk`는 각각 $P$의 첫 번째, 두 번째, 세 번째 열을 고른다. `Gyk1`의 각 행은 oracle을 고르고, `gyk1`은 그 평균으로 $f'(y_{k+1})$를 나타낸다.

```python
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
```

`fyk`, `fyk1`은 각각 $F$의 첫 번째와 두 번째 함수값을 고른다. $y_{k+1}$는 oracle 호출 전에 정해지므로 함수값과 gradient를 outcome별로 나누지 않는다.

```python
fyk = np.zeros(dimF)
fyk[0] = 1  # f(y_k).
fyk1 = np.zeros(dimF)
fyk1[1] = 1  # f(y_{k+1}).
fxs = np.zeros(dimF)  # f(x*) = 0.
```

`statesKG`는 `yk-xs`와 `gyk`를 두 열로 모은 5×2 배열이다. `statesKG @ Q[k] @ statesKG.T`로 averaged iterate의 이차항을 계산하고, `ap[k] * np.outer(xk-xs, xk-xs)`로 SGD iterate의 거리항을 따로 계산한다. 다음 averaged iterate의 항은 `yk1(d[k], k)`와 `gyk1`으로 한 번 계산한다. `XK1`으로 구한 다음 SGD iterate의 거리항만 outcome별로 평균 낸다.

```python
statesKG = np.column_stack((yk - xs, gyk))
statesKF = fyk
statesK1F = fyk1
```

Gram matrix 자체를 decision variable로 선언하지 않고, dualization으로 얻은 LMI 조건의 계수를 구한다. 코드의 `G`, `F`는 interpolation용 딕셔너리이며 위 수식의 Gram matrix와 함수값 행렬과 구분한다.

## 4. Interpolation matrices + multipliers

Appendix D.4의 smooth convex interpolation 조건을 사용한다.

$$
f(x_p)-f(x_q)-\langle f'(x_q),x_p-x_q\rangle
-\frac1{2L}\|f'(x_p)-f'(x_q)\|^2\ge0.
$$

원본 코드에서는 $\{x_\star,y_k,y_{k+1}\}$의 세 점에 적용한다. 각 점의 위치·gradient·함수값 배열을 `X`, `G`, `F`의 같은 키에 넣는다.

`X[3]`은 `yk1(d[k], k)`이고 `G[3]`은 oracle 평균 `gyk1`이다. `XK1`의 점들은 거리항에만 쓰므로 interpolation에 추가하지 않는다. $n=2$에서는 서로 다른 순서쌍 6개를 사용한다.

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

Variance 조건에는 $y_{k+1}$에서 호출하는 `Gyk1`과 그 평균 `gyk1`의 차이를 사용한다. Appendix E.7의 $A_{\mathrm{var}}$를 계산하는 방식과 같으며, 코드의 `Avar`에 저장한다.

```python
Avar = np.zeros((dimG, dimG))
for i in range(1, n + 1):
    Avar = Avar + np.outer(Gyk1[i-1] - gyk1, Gyk1[i-1] - gyk1) / n
```

이 행렬과 Gram matrix의 trace는 oracle의 variance이므로 상한이 $\sigma^2$이다. 제약식에는 `-e[k] * Avar`를 더하고 `e[k] >= 0`을 적용한다.

## 5. Linear equality + semidefinite inequality

다음 potential의 expectation에서 현재 potential을 빼고, interpolation 부등식과 variance 조건의 가중합을 더한다. 원본 코드의 `cons_SDP[k]`에는 이차항 행렬을, `cons_LIN[k]`에는 함수값에 곱해지는 계수를 모은다.

$y_{k+1}$에 관한 이차항과 함수값 항은 한 번만 더한다. Outcome에 따라 달라지는 $x_{k+1}^{(i)}$의 거리항만 평균 낸다.

$n=2$에서 `cons_SDP[k]`는 5×5 대칭행렬이고 `cons_LIN[k]`는 함수값 2개에 대응하는 행벡터이다. 모든 항을 더한 뒤 `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`, `e[k] >= 0`을 적용한다.

선형 등식으로 함수값 항을 소거하고, 남은 이차항이 모든 Gram matrix에 대해 0 이하가 되도록 NSD 조건을 둔다. Interpolation 항은 비음수이고 oracle의 variance는 $\sigma^2$ 이하이므로 2단계의 potential inequality가 성립한다. 6단계에 이 계산과 제약 등록 코드가 나와 있다.

## 6. SDP formulation in CVXPY

모든 `relax`에서 다음 단일 SDP를 구성한다.

$$
\boxed{\min\sum_{k=0}^{N-1}e_k}
$$

모든 전이에 5단계의 `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`, `e[k] >= 0`과 1·2단계의 계수 조건을 적용한다. $d_k$는 두 설정에서 같은 recurrence로 정하며 `tau`, `reoptimize`는 사용하지 않는다.

| `relax` | 중간 상태 $k=1,\ldots,N-1$ | 고정된 $d_k$ | 최종 $a'_N$ |
| --- | --- | --- | --- |
| 0 | $Q_k,a'_k$ 자유 | $d_{k+1}=d_k+\delta_kL$ | $0$ |
| 1 | $Q_k=0,\ a'_k=L/2$ | 같은 recurrence | $L/2$ |

### 초기·최종 계수와 고정 수열을 지정한 뒤 decision variable 선언

2단계에서 지정한 초기·최종 계수를 적용하고, $d_0$에서 $d_N$까지 계산한 뒤 중간 계수와 각 전이의 $\lambda,e_k$를 선언한다. `yk1(d[k], k)`의 averaging 계수는 step-size 수열 `delta(1),...,delta(N)`과 그 수열로 먼저 계산한 $d_0,\ldots,d_N$을 대입해 구한다.

```python
a0 = b0 = c0 = d0 = 0
ap0 = L / 2
aN = bN = cN = 0
apN = 0 if relax == 0 else L / 2

Q0 = np.array([[a0, c0], [c0, b0]])
QN = np.array([[aN, cN], [cN, bN]])
q0 = d0
Q = {1: Q0, N+1: QN}
d = {1: q0}
ap = {1: ap0, N+1: apN}
for k in range(1, N + 1):
    d[k+1] = d[k] + delta(k) * L
for k in range(1, N):
    if relax == 0:
        ap[k+1] = cp.Variable(name=f"ap_{k+1}")
        Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
    elif relax == 1:
        ap[k+1] = L / 2
        Q[k+1] = np.zeros((2, 2))

lam = {}
e = {}
for k in range(1, N + 1):
    lam[k] = cp.Variable((nbPts, nbPts), name=f"lambda_{k}")
    e[k] = cp.Variable(name=f"e_{k}")
```

`d`의 루프는 마지막 전이도 포함한다. 따라서 기본 설정에서는 `d[1]=0`, `d[2]=1`, …, `d[101]=100`이며, 마지막 계수 $d_N$도 SDP 조립 전에 존재한다. 위 선언은 원본의 `if k < N` 안에 있는 계수 선언 및 그 밖에 있는 `d[k+1]` 계산과 같은 계수·이름·자유도를 사용한다.

### Potential 차이의 초기화, interpolation 가산, 제약 등록

각 전이에서 $y_{k+1}$의 potential 항은 한 번만 더하고, outcome에 따라 달라지는 $x_{k+1}^{(i)}$의 거리항만 평균 낸다. 그 다음 interpolation 항을 더하고 완성된 제약을 등록한다.

```python
cons_SDP = {}
cons_LIN = {}
cons = []
for k in range(1, N + 1):
    XK1 = xk1(k)
    X = {1: xs, 2: yk, 3: yk1(d[k], k)}
    G = {1: gxs, 2: gyk, 3: gyk1}
    F = {1: fxs, 2: fyk, 3: fyk1}

    cons_SDP[k] = -statesKG @ Q[k] @ statesKG.T - ap[k] * np.outer(xk-xs, xk-xs) - e[k] * Avar
    cons_LIN[k] = -d[k] * statesKF
    statesK1G = np.column_stack((yk1(d[k], k) - xs, gyk1))
    cons_SDP[k] = cons_SDP[k] + statesK1G @ Q[k+1] @ statesK1G.T
    for i in range(1, n + 1):
        cons_SDP[k] = cons_SDP[k] + ap[k+1] * np.outer(XK1[i-1]-xs, XK1[i-1]-xs) / n
    cons_LIN[k] = cons_LIN[k] + statesK1F * d[k+1]

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
obj = accumulated_ek
```

이 조립 순서에서 모든 제약은 decision variable에 대해 affine이다. $d_k$를 free variable로 바꾸면 2단계의 averaging 계수와 이 SDP 정의가 달라지므로, 두 `relax` 모두 위의 고정 recurrence를 사용한다.

### 문제 규모

$N=100,n=2$에서 5×5 NSD LMI는 100개이고, 함수값 계수의 scalar 등식은 $2\times100=200$개이다.

| `relax` | 중간 $Q_k$ 독립 원소 | 중간 $a'_k$ | 자유 $d_k$ | $\lambda$ | $e_k$ | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 297 | 99 | 0 | 900 | 100 | 1396 |
| 1 | 0 | 0 | 0 | 900 | 100 | 1000 |

`relax=1`의 decision variable은 $\lambda,e$뿐이다. 위 수에는 interpolation 합에 사용하지 않는 `lam`의 대각 원소도 포함한다.

## 7. MOSEK numerical optimization

기본 설정은 `solver="MOSEK"`, `tolerance=1e-8`이다. 이 값을 `MSK_DPAR_INTPNT_CO_TOL_PFEAS` 옵션으로 전달해 MOSEK의 primal feasibility tolerance를 지정한다.

아래의 `solve`는 원본 Python 파일과 같은 디렉터리의 `saveData.py`에서 가져온 함수이다. MOSEK를 선택하면 `_MatlabOrderMOSEK` adapter가 변수 순서와 PSD 계수 전달 방식을 MATLAB/YALMIP에 맞춘다. 이때 canonicalization과 해 복원은 CVXPY가 그대로 담당하며, 다른 solver를 선택하면 일반 CVXPY 경로로 계산한다.

계산은 다음 순서로 진행된다.

1. CVXPY가 affine objective와 선형·PSD 제약을 cone problem으로 변환한다.
2. MOSEK가 이 유한 차원 SDP의 수치해를 계산한다.
3. CVXPY가 계산된 해를 원래 `cp.Variable`의 `.value`에 저장한다.

이 계산에서 구하는 것은 potential의 계수와 제약식을 구성하는 multiplier이다. 특정 데이터셋에 SGD를 실행해 iterate의 궤적을 얻는 과정은 포함되지 않는다.

```python
accumulated_ek = 0
for k in range(1, N + 1):
    accumulated_ek = accumulated_ek + e[k]
obj = accumulated_ek

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
```

앞의 두 stochastic 실험에서는 $d_N$을 최대화한 뒤 그 하한을 유지하며 다시 최적화했지만, 이 실험은 한 번의 최소화로 끝난다. 따라서 `solverDetails.value`와 `accumulated_ek.value`는 모두 최소화된 누적 오차 계수를 나타낸다.

## 8. Optimal coefficients / algorithm parameters

단일 SDP의 해를 최종 해로 사용한다. 원본과 같이 `pplot=1`인 실행에서 아래 계수 추출을 수행한다. `pplot=0`이면 배열 추출과 9단계의 그림·파일 출력을 생략한다. $a_k,b_k,c_k$는 `Q[k+1]`의 원소에서, $a_k'$와 $e_k$는 각각 `ap[k+1]`, `e[k+1]`에서 읽는다. `d[k+1]`에는 recurrence로 미리 계산한 상수가 들어 있다. free variable은 `.value`를 읽고, 고정 계수는 상수 자체를 읽는다.

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

`apk[k]`는 $x_k$의 거리항 계수 $a_k'$이다. `ek[0]=0`은 그래프 표시용이고, 각 단계의 오차는 `ek[k+1]=e_k`로 저장한다. 따라서 `ekc[k]=\sum_{t=0}^{k-1}e_t`는 상태 $k$에 도달할 때까지의 누적합이다.

코드는 대상이 `cp.Expression`이면 `.value`를, 상수이면 그 값 자체를 읽는다. 예를 들어 `Q[2]`와 `ap[2]`에서 $a_1,b_1,c_1,a_1'$를 읽는 동안, `d[2]`에서는 recurrence로 계산한 $d_1=1$을 읽는다. 기본 설정 `alpha=0, L=1`에서는 모든 단계에서 $\delta_k=1/L=1$이고 $d_k=k$이다. 따라서 기존 평균 $y_k$와 SGD iterate $x_k$의 가중치는 각각 $k/(k+1)$과 $1/(k+1)$이다.

## 9. Plotting, numerical analysis

### 출력 설정과 실행

출력 관련 변수는 `pplot`, `ssave`, `folder`, `nname`이며 기본값은 각각 `1`, `1`, `"SaveData/"`, `"SGD_Primal_Averaging.dat"`이다. `pplot`과 `ssave`에 따른 실행 범위는 다음과 같다.

| `pplot` | `ssave` | 8–9단계의 실행 |
| --- | --- | --- |
| 0 | 0 또는 1 | 계수 배열 추출, plotting, 저장을 모두 생략 |
| 1 | 0 | 계수 배열 추출과 plotting 수행, 파일 저장 생략 |
| 1 | 1 | 계수 배열 추출, plotting, 파일 저장 수행 |

각 `relax` 값에 대해 SDP를 따로 푼다. 결과를 함께 보관하려면 `folder` 또는 `nname`을 다르게 설정한다. `relax` 값을 바꾸면 초기·최종 계수 조건과 decision variable도 다시 구성한다.

### 계수 출력과 저장

코드는 $a_k,b_k,c_k,d_k,e_k$와 오차 계수의 누적합, $a_k'$를 일곱 패널로 그린다. `pplot=1`이면 이 배열들을 만들고, 여기에 `ssave=1`도 설정되어 있으면 [SaveData/SGD_Primal_Averaging.dat](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/SaveData/SGD_Primal_Averaging.dat)에 `k, ak, bk, ck, dk, ek, ekc, apk`를 저장한다. 저장되는 인덱스 1,…,101은 논문의 상태 인덱스 0,…,100에 대응한다.

논문에는 이 파일의 두 `relax` 설정을 직접 비교하는 별도 그림이 없다. 수치 결과를 해석할 때는 출력된 계수를 Theorem 7의 potential 및 오차 계수와 비교할 수 있다.

### Theorem 7과 수치 결과의 대응

Theorem 7에서는 $0\le\delta_k\le1/L$일 때 다음 계수들이 감소 조건을 만족함을 보인다.

$$
Q_k=0,\qquad a_k'=L/2,\qquad
d_{k+1}=d_k+\delta_kL,\qquad e_k=\frac{L\delta_k^2}{2}.
$$

기본 설정인 $L=1,\delta_k=1$에서는 $d_k=k$, $e_k=1/2$이므로 100단계의 오차 계수 합은 50이다. 이는 `relax=1`에서 사용할 수 있는 해석적 계수 선택이다. 최종 거리항을 버리면 `relax=0`에서도 유효한 상한을 얻으므로, 50은 수치 최적값과 비교할 기준이 된다.

원본 저장 파일의 마지막 행에는 `dk=100`, `ekc=49.5`, `apk=0`이 들어 있다. 이 수치 결과는 최종 거리항을 제거한 기본 설정에서 얻은 것이며, 모든 단계에서 $a_k'=L/2$를 유지하는 정리의 계수 선택과는 최종 거리항 계수값이 다르다. 여기서 `ek`는 마지막 단계의 오차이고 `ekc`는 누적합이다.

### Plotting 배열과 저장 열의 대응

| 출력 배열 | 대응하는 수식 | 배열에 담기는 순서 |
| --- | --- | --- |
| `ak,bk,ck,dk` | $y_k$ potential의 $a_k,b_k,c_k,d_k$ | 상태 0,…,100 |
| `apk` | $x_k$ 거리항의 $a_k'$ | 상태 0,…,100 |
| `ek` | $e_k$ | 표시용 0 뒤에 100개 단계의 값 |
| `ekc` | $\sum_{t=0}^{k-1}e_t$ | 상태 $k$까지의 누적합 |

```python
if pplot:
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
```

마지막 행의 `apk`가 0이면 최종 potential에 거리항이 없고, $L/2$이면 거리항이 남는다. 따라서 Theorem 7과 계수 수열을 비교할 때는 이 최종 거리항 계수값도 함께 고려해야 한다.
