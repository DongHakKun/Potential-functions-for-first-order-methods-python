# 실험 2. Bounded Variance SGD — §3.2.2, Figure 2

이 문서는 논문 §3.2.2의 식 (4), Figure 2, Appendix E.7(pp.48–49)에 대응하는 [A_StochasticGradientDescent.py](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/A_StochasticGradientDescent.py)의 수치 실험을 설명한다. `relax=0,1,2,3`에 따른 potential과 SDP를 살펴보고, 코드에서 계수와 수치 결과를 얻는 과정을 따라간다.

수식에서 상태의 인덱스는 $k=0,\ldots,N$, 한 단계 전이의 인덱스는 $k=0,\ldots,N-1$로 둔다. Python 딕셔너리는 1부터 시작하므로 $Q_k,d_k$는 `Q[k+1]`, `d[k+1]`에, $e_k,\lambda$는 `e[k+1]`, `lam[k+1]`에 대응한다. 코드 블록의 루프 변수 `k`는 원본대로 `1,...,N`을 사용한다.

최적점과 최적 함수값은 $x_\star,f_\star$, gradient는 $f'(x)$로 쓴다. Gram 표현에는 Appendix E.7의 $\mathbf x_0,\mathbf x_i,\mathbf g_0,\mathbf g_i,\mathbf f_0,\mathbf f_i$를 그대로 사용한다. 여기서 $0,i$는 한 전이 안의 점 번호이며, 알고리즘의 단계 $k$와 구분한다.

1–5단계에서는 potential과 한 단계 LMI를 설명한다. 6단계에서 이를 CVXPY로 구현하고, 7–9단계에서 최적화 결과를 읽어 그래프와 저장 데이터로 확인한다.

## 1. Parameterized Potential

$f\in\mathcal F_L$로 두고, 균등한 $i\in\{1,\ldots,n\}$에 대한 stochastic oracle에 다음 조건을 적용한다.

$$
\mathbb E_iG(x;i)=f'(x),\qquad
\mathbb E_i\|G(x;i)-f'(x)\|^2\le\sigma^2.
$$

Potential family는 다음과 같다.

$$
\begin{aligned}
\phi_k^f&=
\begin{pmatrix}x_k-x_\star\\f'(x_k)\end{pmatrix}^{\!\top}
(Q_k\otimes I_d)
\begin{pmatrix}x_k-x_\star\\f'(x_k)\end{pmatrix}
+d_k(f(x_k)-f_\star),\\
Q_k&=\begin{pmatrix}a_k&c_k\\c_k&b_k\end{pmatrix}.
\end{aligned}
$$

| 수식의 기호 | Python 표현 | 역할 |
| --- | --- | --- |
| $L,\mu,n,N$ | `L=1`, `m=0`, `n=2`, `N=100` | Figure 2의 문제와 실험 규모 |
| $a_k,b_k,c_k$ | `Q[k+1][0,0]`, `[1,1]`, `[1,0]` | potential의 이차항 계수 |
| $d_k$ | `d[k+1]` | 함수값 차이 계수 |
| $e_k$ | `e[k+1]` | 한 단계에서 variance에 곱해지는 오차 계수 |
| $d_N$ | `tau`, `dN` | 1차 SDP의 최대화 대상 |
| $\sum_{k=0}^{N-1}e_k$ | `accumulated_ek` | `reoptimize=1`일 때 2차 SDP의 최소화 대상 |

코드의 `tau`는 이 실험의 $d_N$을 나타낸다. 뒤의 parameter-selection 실험에서 알고리즘을 정하는 $\tau_k$와 구분한다.

`relax`는 한 번의 실행에서 `0,1,2,3` 중 하나를 선택한다. 기본값은 `0`이다. 중간 상태 $k=1,\ldots,N-1$에는 선택한 행의 조건만 적용한다.

| `relax` | 고정하는 중간 계수 | SDP가 정하는 중간 계수 |
| --- | --- | --- |
| 0 | 없음 | $a_k,b_k,c_k,d_k$ |
| 1 | $c_k=0$ | $a_k,b_k,d_k$ |
| 2 | $b_k=c_k=0$ | $a_k,d_k$ |
| 3 | $a_k=L/2,\ b_k=c_k=0$ | $d_k$ |

`relax` 값과 관계없이 $e_k$와 interpolation multiplier는 결정변수이다. $Q_k$는 대칭 행렬이며 별도의 PSD 제약은 없다. 초기·최종 계수는 2단계에서 지정한다.

Figure 2의 기본 설정은 다음과 같다.

```python
L = 1
m = 0
n = 2
N = 100
alpha = 0
delta = lambda k: 1 / L / (k + 1) ** alpha
```

`alpha=0`에서는 $\delta_k=1/L=1$이다. `alpha`는 step-size의 decay exponent이며, parameter-selection 실험의 $\alpha_k$와는 별개이다. 이 실험에서는 step-size를 SDP가 정하는 변수로 두지 않는다.

수식의 $\delta_k$는 원본 코드의 `delta(k+1)`에 대응한다. 따라서 `alpha`를 바꾸면 첫 호출 `delta(1)`은 $1/(L2^\alpha)$가 되어, 논문의 $1/[L(k+1)^\alpha]$보다 한 단계 뒤의 값에서 시작한다. 원본 코드를 그대로 실행하는 경우에는 이 인덱스 대응을 유지한다.

## 2. Potential Inequality

각 전이 $k=0,\ldots,N-1$에서 다음 iterate와 기대값 부등식을 사용한다.

$$
x_{k+1}^{(i)}=x_k-\delta_kG(x_k;i),\qquad
\boxed{\mathbb E_i[\phi_{k+1}^f(x_{k+1}^{(i)})]
\le\phi_k^f(x_k)+e_k\sigma^2}.
$$

오른쪽의 $e_k\sigma^2$는 oracle의 variance에 따른 오차이다. 초기 potential은

$$
\phi_0^f=\frac L2\|x_0-x_\star\|^2,
\qquad a_0=L/2,\quad b_0=c_0=d_0=0
$$

으로 정한다. 최종 potential은 `relax` 값에 따라 다음과 같다.

| `relax` | 최종 계수 | 최종 potential |
| --- | --- | --- |
| 0, 1, 2 | $a_N=b_N=c_N=0,\ d_N=\texttt{tau}$ | $\phi_N^f=d_N(f(x_N)-f_\star)$ |
| 3 | $a_N=L/2,\ b_N=c_N=0,\ d_N=\texttt{tau}$ | $\phi_N^f=(L/2)\|x_N-x_\star\|^2+d_N(f(x_N)-f_\star)$ |

최종 거리항은 `relax=3`에서만 남는다. $Q_0,Q_N,d_0,d_N$에는 위 표의 값을 사용한다.

원본의 `a0,b0,c0,d0,aN,bN,cN,dN`이 이 계수에 대응하며, `tau=cp.Variable(name="tau")`가 최종 계수 $d_N$이다. CVXPY에서 초기·최종 계수를 지정하는 전체 코드는 6단계에 둔다. `Q[1]`과 `Q[N+1]`은 각각 $Q_0,Q_N$에 대응한다.

각 단계의 부등식에 기대값을 취하고 더하면 중간 potential이 소거된다. `relax=0,1,2`에서는

$$
d_N\mathbb E[f(x_N)-f_\star]
\le\frac L2\|x_0-x_\star\|^2+
\sigma^2\sum_{k=0}^{N-1}e_k,
$$

`relax=3`에서는

$$
\frac L2\mathbb E\|x_N-x_\star\|^2
+d_N\mathbb E[f(x_N)-f_\star]
\le\frac L2\|x_0-x_\star\|^2+
\sigma^2\sum_{k=0}^{N-1}e_k
$$

를 얻는다. 후자의 비음수 거리항을 버리면 같은 함수값 상한이 성립한다. 따라서 $d_N>0$인 결과에서 모든 설정의 함수값 상한은

$$
\mathbb E[f(x_N)-f_\star]
\le\frac{L\|x_0-x_\star\|^2}{2d_N}
+\frac{\sum_{k=0}^{N-1}e_k}{d_N}\sigma^2
$$

로 읽는다. 6단계는 $d_N$을 최대화하는 1차 SDP와, `reoptimize=1`인 경우 그 objective value를 거의 유지하며 누적 오차를 최소화하는 2차 SDP를 구성한다.

## 3. Finite-dimensional Gram representation

Appendix E.7, pp.48–49에서는 일반성을 잃지 않고 $x_\star=f_\star=0$으로 두고 다음 행렬을 사용한다. 최적점에서는 $f'(x_\star)=0$이다.

$$
\begin{aligned}
P&=[x_k\ G(x_k;1)\ \cdots\ G(x_k;n)\ f'(x_{k+1}^{(1)})\ \cdots\ f'(x_{k+1}^{(n)})],\\
F&=[f(x_k)\ f(x_{k+1}^{(1)})\ \cdots\ f(x_{k+1}^{(n)})],\\
G&=P^\top P\in\mathbb S_+^{1+2n}.
\end{aligned}
$$

$n=2$이면 벡터 5개, 함수값 3개이며, interpolation에는 최적점을 포함한 네 점을 사용한다.

```python
dimG = 1 + 2 * n
dimF = n + 1
nbPts = 2 + n
```

기본 설정 $n=2$에서 각 열을 고르는 계수벡터를 성분으로 쓰면 다음과 같다.

$$
\begin{aligned}
\mathbf x_0&=\begin{pmatrix}1\\0\\0\\0\\0\end{pmatrix},&
\mathbf G_1&=\begin{pmatrix}0\\1\\0\\0\\0\end{pmatrix},&
\mathbf G_2&=\begin{pmatrix}0\\0\\1\\0\\0\end{pmatrix},\\
\mathbf g_0&=\begin{pmatrix}0\\1/2\\1/2\\0\\0\end{pmatrix},&
\mathbf g_1&=\begin{pmatrix}0\\0\\0\\1\\0\end{pmatrix},&
\mathbf g_2&=\begin{pmatrix}0\\0\\0\\0\\1\end{pmatrix},\\
\mathbf f_0&=\begin{pmatrix}1\\0\\0\end{pmatrix},&
\mathbf f_1&=\begin{pmatrix}0\\1\\0\end{pmatrix},&
\mathbf f_2&=\begin{pmatrix}0\\0\\1\end{pmatrix}.
\end{aligned}
$$

일반적인 $n$에서는 $\mathbf x_0$의 첫 번째 성분, $\mathbf G_i$의 $1+i$번째 성분, $\mathbf g_i$의 $1+n+i$번째 성분을 각각 1로 두고 나머지는 0으로 둔다. 함수값의 계수벡터 $\mathbf f_0,\mathbf f_i$도 각각 첫 번째와 $1+i$번째 성분만 1이다. 현재 gradient와 다음 점의 계수벡터는

$$
\mathbf g_0=\frac1n\sum_{i=1}^n\mathbf G_i,\qquad
\mathbf x_i=\mathbf x_0-\delta_k\mathbf G_i
$$

로 얻는다. $n=2$의 다음 점을 성분으로 쓰면

$$
\begin{aligned}
\mathbf x_1&=\begin{pmatrix}1\\-\delta_k\\0\\0\\0\end{pmatrix},&
\mathbf x_2&=\begin{pmatrix}1\\0\\-\delta_k\\0\\0\end{pmatrix}
\end{aligned}
$$

이다. 점·gradient의 계수벡터는 $\mathbb R^{1+2n}$에, 함수값의 계수벡터는 $\mathbb R^{1+n}$에 속한다. 최적점에 해당하는 벡터는 모두 0이다.

| Appendix E.7의 기호 | Python 표현 | 나타내는 값 |
| --- | --- | --- |
| $\mathbf x_0$ | `xk` | $P\mathbf x_0=x_k$ |
| $\mathbf G_i$ | `Gxk[i-1]` | $P\mathbf G_i=G(x_k;i)$ |
| $\mathbf g_0$ | `gxk` | $P\mathbf g_0=f'(x_k)$ |
| $\mathbf x_i$ | `xk1(k+1)[i-1]` | $P\mathbf x_i=x_{k+1}^{(i)}$ |
| $\mathbf g_i$ | `gxk1[i-1]` | $P\mathbf g_i=f'(x_{k+1}^{(i)})$ |
| $\mathbf f_0$ | `fxk` | $F\mathbf f_0=f(x_k)$ |
| $\mathbf f_i$ | `fxk1[i-1]` | $F\mathbf f_i=f(x_{k+1}^{(i)})$ |

```python
xk = np.zeros(dimG)
xk[0] = 1  # x_k.
xs = np.zeros(dimG)  # x* = 0.
Gxk = np.zeros((n, dimG))
Gxk[:, 1:n+1] = np.eye(n)  # Gxk[i] is G(xk; i).
gxk = np.sum(Gxk, axis=0) / n  # f'(x_k).
gxk1 = np.zeros((n, dimG))
gxk1[:, 1+n:1+2*n] = np.eye(n)  # f'(x_{k+1}^{(i)}).
gxs = np.zeros(dimG)  # f'(x*) = 0.
```

논문의 $V_k,V_{k+1}^{(i)}$를 직접 쓰면 다음과 같다.

$$
\begin{aligned}
V_k&=[\mathbf x_0-\mathbf x_\star\ \mathbf g_0]
Q_k[\mathbf x_0-\mathbf x_\star\ \mathbf g_0]^\top,\\
V_{k+1}^{(i)}&=[\mathbf x_i-\mathbf x_\star\ \mathbf g_i]
Q_{k+1}[\mathbf x_i-\mathbf x_\star\ \mathbf g_i]^\top.
\end{aligned}
$$

코드의 `statesKG`와 `statesK1G`는 위 행렬곱에서 $Q_k,Q_{k+1}$ 왼쪽의 두 열을 만든다. 다음 potential의 기대 이차항은 $n^{-1}\sum_{i=1}^n V_{k+1}^{(i)}$로 직접 쓴다. 코드에서는 `statesK1G @ Q[k+1] @ statesK1G.T / n`을 outcome마다 더한다.

```python
fxk = np.zeros(dimF)
fxk[0] = 1  # f(x_k).
fxk1 = np.zeros((n, dimF))
fxk1[:, 1:n+1] = np.eye(n)  # f(x_{k+1}^{(i)}).
fxs = np.zeros(dimF)  # f(x*) = 0.

xk1 = lambda k: np.ones((n, 1)) * xk - delta(k) * Gxk
```

Gram matrix 자체를 결정변수로 선언하지 않고, 논문의 dualization으로 얻은 LMI 조건에서 potential 계수와 multiplier를 구한다. 코드의 `G`, `F`는 interpolation용 딕셔너리이며, 위 수식의 Gram matrix와 함수값 행렬과 구분한다.

## 4. Interpolation matrices + multipliers

Appendix E.7의 점 번호 집합은 $I=\{\star,0,1,\ldots,n\}$이다. 각 $p,q\in I$에 대해 Appendix D.4의 interpolation 조건을 사용한다.

$$
\begin{aligned}
M_{p,q}&=[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]
M[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]^\top,\\
F(\mathbf f_p-\mathbf f_q)+\operatorname{Tr}(GM_{p,q})&\ge0.
\end{aligned}
$$

논문 식 (19)의 $M$은 $\mu=0$인 smooth convex 경우이다. 원본 Python 코드는 strong convexity까지 지원하는 다음 행렬을 사용하며, 기본 설정 `m=0`에서 논문의 행렬과 일치한다.

$$
M=\frac1{2(L-\mu)}
\begin{pmatrix}
-L\mu&L\mu&\mu&-L\\
L\mu&-L\mu&-\mu&L\\
\mu&-\mu&-1&1\\
-L&L&1&-1
\end{pmatrix}.
$$

코드의 `TT = np.vstack((xi, xj, gi, gj))`는 $[\mathbf x_p\ \mathbf x_q\ \mathbf g_p\ \mathbf g_q]^\top$에 해당한다. 같은 `lam[k][i-1,j-1]`을 `TT.T @ M @ TT`와 `fi-fj`에 곱해 각각 이차항과 함수값 계수에 더한다. `lam[k] >= 0`은 원소별 부호 조건이다.

코드 딕셔너리의 키 `1`은 최적점, `2`는 현재 점, `2+i`는 outcome $i$의 다음 점이다. $n=2$에서는 네 점의 서로 다른 순서쌍 12개를 사용한다. `lam[k]`는 4×4로 선언하되 대각 원소는 합에서 제외한다.

Variance에는 Appendix E.7에서 정의한 $A_{\mathrm{var}}$를 사용한다.

$$
A_{\mathrm{var}}=\frac1n\sum_{i=1}^n
(\mathbf G_i-\mathbf g_0)(\mathbf G_i-\mathbf g_0)^\top,
\qquad \operatorname{Tr}(GA_{\mathrm{var}})\le\sigma^2.
$$

$n=2$에서는 다음과 같다.

$$
A_{\mathrm{var}}=\frac14
\begin{pmatrix}0\\1\\-1\\0\\0\end{pmatrix}
\begin{pmatrix}0\\1\\-1\\0\\0\end{pmatrix}^{\!\top}
\in\mathbb S_+^5.
$$

코드의 `Avar`가 이 행렬이며, `e[k] >= 0`인 계수를 곱해 제약에 반영한다.

## 5. Linear equality + semidefinite inequality

Appendix E.7, p.49의 조건은 다음과 같다. $\lambda_{p,q}$는 한 전이에서 사용하는 multiplier이다.

$$
\begin{aligned}
\frac1n\sum_{i=1}^nV_{k+1}^{(i)}-V_k-e_kA_{\mathrm{var}}
+\sum_{p,q\in I}\lambda_{p,q}M_{p,q}&\preceq0,\\
\frac{d_{k+1}}n\sum_{i=1}^n(\mathbf f_i-\mathbf f_\star)
-d_k(\mathbf f_0-\mathbf f_\star)
+\sum_{p,q\in I}\lambda_{p,q}(\mathbf f_p-\mathbf f_q)&=0,\\
\lambda_{p,q}&\ge0,\qquad e_k\ge0.
\end{aligned}
$$

첫 번째 좌변은 5×5 대칭행렬이고, 두 번째 좌변은 세 함수값에 곱해지는 계수를 모은 벡터이다. 코드에서는 각각 `cons_SDP[k]`, `cons_LIN[k]`에 저장한다. `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`, `e[k] >= 0`을 적용한다.

함수값 계수를 소거하면 남은 이차항이 모든 $G\succeq0$에 대해 0 이하가 된다. 또한 $e_k\ge0$와 variance 상한에 의해 $-e_k\sigma^2\le-e_k\operatorname{Tr}(GA_{\mathrm{var}})$이므로, 2단계의 potential inequality가 성립한다.

$n=2$에서 함수값 선형 등식을 각 성분으로 풀어 쓰면 다음과 같다.

$$
\begin{aligned}
-d_k+\sum_{q\in I}(\lambda_{0,q}-\lambda_{q,0})&=0,\\
\frac{d_{k+1}}2+\sum_{q\in I}(\lambda_{1,q}-\lambda_{q,1})&=0,\\
\frac{d_{k+1}}2+\sum_{q\in I}(\lambda_{2,q}-\lambda_{q,2})&=0.
\end{aligned}
$$

각각 $f(x_k)$, $f(x_{k+1}^{(1)})$, $f(x_{k+1}^{(2)})$의 계수이다. 대각 multiplier의 기여는 서로 상쇄되므로 코드의 합에서 제외해도 같다.

## 6. SDP formulation in CVXPY

`reoptimize`는 `relax`와 독립적으로 선택한다. 기본값은 `reoptimize=1`, `tol_reopt=1e-5`이다.

| `reoptimize` | 풀 문제 | 후처리에 사용할 계수 |
| --- | --- | --- |
| 0 | $d_N$을 최대화하는 1차 SDP | 1차 SDP의 계수 |
| 1 | 1차 SDP 후, $d_N$ 하한을 추가하여 $\sum e_k$를 최소화하는 2차 SDP | 2차 SDP의 계수 |

논문 식 (4)의 두 번째 최적화는 첫 번째 최적화에서 얻은 최종 함수값 계수를 고정하고 누적 variance 계수를 최소화한다. 원본 코드는 첫 번째 문제의 목적식 `obj=-tau`에서 얻은 `-float(obj.value)`를 사용해 `tau >= -float(obj.value) - tol_reopt`를 추가한다. 이 허용오차 조건과 기존 제약 아래에서 `accumulated_ek`를 최소화한다.

### 초기·최종 계수 조건과 공통 행렬 구성

초기·최종 계수와 `relax` 분기에 해당하는 코드는 다음과 같다.

```python
a0 = L / 2
b0 = 0
c0 = 0
d0 = 0
tau = cp.Variable(name="tau")
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

Q0 = np.array([[a0, c0], [c0, b0]])
q0 = d0
QN = np.array([[aN, cN], [cN, bN]])
qN = dN
statesKG = np.column_stack((xk - xs, gxk))
statesKF = fxk

M = np.array([
    [-L*m, L*m, m, -L],
    [L*m, -L*m, -m, L],
    [m, -m, -1, 1],
    [-L, L, 1, -1],
]) / (2 * (L - m))
Avar = np.zeros((dimG, dimG))
for i in range(1, n + 1):
    Avar = Avar + np.outer(Gxk[i-1] - gxk, Gxk[i-1] - gxk) / n
```

`Q0,QN`은 2단계의 $Q_0,Q_N$이고, `q0,qN`은 $d_0,d_N$이다. `M,Avar`는 4단계의 interpolation·variance 행렬에 해당한다.

### 변수 생성 후 전이별 LMI 조립

먼저 초기·최종 계수값과 빈 딕셔너리를 준비한다. 각 전이에서는 필요한 중간 potential 변수와 `lam[k],e[k]`를 생성한 다음, 현재 항·다음 항·interpolation 항을 순서대로 조립한다. 모든 항을 더한 뒤에 해당 전이의 제약을 `cons`에 넣는다.

```python
Q = {1: Q0, N+1: QN}
d = {1: q0, N+1: qN}
lam = {}
e = {}
cons_SDP = {}
cons_LIN = {}
cons = [tau >= 0]

for k in range(1, N + 1):
    if k < N:
        if relax == 0:
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.Variable((2, 2), symmetric=True, name=f"Q_{k+1}")
        elif relax == 1:
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.bmat([
                [cp.Variable(name=f"a_{k+1}"), 0],
                [0, cp.Variable(name=f"b_{k+1}")]
            ])
        elif relax == 2:
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = cp.bmat([[cp.Variable(name=f"a_{k+1}"), 0], [0, 0]])
        elif relax == 3:
            d[k+1] = cp.Variable(name=f"d_{k+1}")
            Q[k+1] = np.array([[L/2, 0], [0, 0]])

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

    for i in range(1, n + 1):
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
```

루프의 `k=1`은 수식의 상태 0에서 상태 1로 넘어가는 전이이며, 생성되는 `Q[2]`가 $Q_1$이다. 마지막 `k=N`에서는 중간 변수 생성을 건너뛰고 이미 지정한 `Q[N+1],d[N+1]`을 사용한다. `relax=3`에서는 상수 NumPy 행렬을, 다른 설정에서는 해당 자유도에 맞는 CVXPY 표현식을 사용한다.

### 전체 SDP 규모와 제약

선택한 `relax`에서 자유로운 potential 계수, 각 전이의 interpolation multiplier와 variance 계수, 최종 $d_N$을 최적화한다. 모든 전이에 5단계의 NSD·선형 등식·부호 조건과 1·2단계의 계수 조건을 적용하고 $d_N\ge0$을 둔다. $Q_k\succeq0$ 제약은 추가하지 않는다.

$n=2,N=100$에서는 5×5 NSD LMI가 100개, 함수값 선형 등식이 300개이다. 원본처럼 `lam[k]`의 대각 원소도 선언할 때 독립 scalar 변수 수는 다음과 같다.

| `relax` | 중간 $Q_k$ 독립 원소 | 중간 $d_k$ | $\lambda$ | $e_k$ | $d_N$ | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 297 | 99 | 1600 | 100 | 1 | 2097 |
| 1 | 198 | 99 | 1600 | 100 | 1 | 1998 |
| 2 | 99 | 99 | 1600 | 100 | 1 | 1899 |
| 3 | 0 | 99 | 1600 | 100 | 1 | 1800 |

기본 `relax=0`에서는 $2097=297+99+1600+100+1$개의 독립 scalar 변수를 사용한다. 두 번째 SDP는 같은 결정변수와 제약을 사용하고 objective와 $d_N$의 하한만 바꾼다.

## 7. MOSEK numerical optimization

기본 설정은 `solver="MOSEK"`, `verbose=1`, `tolerance=1e-8`이다. 첫 번째 SDP를 풀 때 이 값을 `MSK_DPAR_INTPNT_CO_TOL_PFEAS` 옵션으로 전달해 MOSEK의 primal feasibility tolerance를 지정한다.

아래의 `solve`는 원본 Python 파일과 같은 디렉터리의 `saveData.py`에서 가져온 함수이다. MOSEK를 선택하면 `_MatlabOrderMOSEK` adapter가 변수 순서와 PSD 계수 전달 방식을 MATLAB/YALMIP에 맞춘다. 이때 canonicalization과 해 복원은 CVXPY가 그대로 담당하며, 다른 solver를 선택하면 일반 CVXPY 경로로 계산한다.

계산은 다음 순서로 진행한다.

1. CVXPY가 affine objective와 선형·PSD 제약을 cone problem으로 변환한다.
2. MOSEK가 이 유한 차원 SDP의 수치해를 계산한다.
3. CVXPY가 계산된 해를 원래 `cp.Variable`의 `.value`에 저장한다.

이 계산에서 구하는 것은 potential의 계수와 제약식을 구성하는 multiplier이다. 특정 데이터셋에 SGD를 실행해 iterate의 궤적을 얻는 과정은 포함하지 않는다.

모든 설정에서 먼저 `obj=-tau`를 최소화하여 $d_N$을 최대화한다. `reoptimize=1`일 때만 두 번째 문제의 objective를 `accumulated_ek`로 바꿔 누적 오차를 최소화한다. 원본과 같은 solver 호출 순서를 사용한다.

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

첫 번째 `solve` 직후의 `-float(obj.value)`가 첫 번째 SDP에서 얻은 최종 계수이다. 두 번째 계산은 같은 변수의 `.value`를 갱신하므로, 두 결과를 비교하려면 첫 번째 계산 직후의 값을 따로 보관한다.

두 번째 호출에서는 `solver_opt`를 새로 만들면서 MOSEK의 `mosek_params`는 다시 넣지 않는다. 따라서 해당 tolerance는 MOSEK의 기본값을 사용한다. SDPA의 옵션은 다시 추가한다.

`reoptimize=0`이면 첫 번째 SDP만 푼다. 이때의 $e_k$도 제약식을 만족하는 값이지만, 누적합을 최소화한 결과는 아니다.

## 8. Optimal coefficients / algorithm parameters

`reoptimize=0`에서는 1차 SDP의 값을, `reoptimize=1`에서는 두 번째 solve가 갱신한 값을 사용한다. `Q,d,e,lam`에서 해당 수치 계수를 읽는다. 기본 설정 `alpha=0, L=1`에서는 모든 단계에서 step-size $\delta_k=1/L=1$을 사용한다.

원본은 표시용 계수 배열의 생성과 추출을 `if pplot:` 안에서 수행한다. `pplot=1`인 경우 아래 배열을 만들고 추출하며, `pplot=0`에서는 이 표시용 경로를 건너뛰고 풀이 결과는 `Q,d,e,lam`에 남는다. 다음 코드는 원본의 `if pplot:` 내부에 대응한다.

```python
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
```

Potential 계수가 CVXPY 변수·표현식이면 `.value`를 읽고, `relax`로 고정한 상수이면 NumPy 원소나 scalar를 그대로 읽는다. 따라서 `relax=3`의 상수 $Q_k$에도 같은 추출 경로를 적용한다. `e`와 `lam`은 모든 설정에서 CVXPY 변수이다.

배열 `ak,bk,ck,dk`의 인덱스 $k$에는 상태 $k$의 계수가 들어간다. `ek[0]=0`은 그래프 표시용이고, 논문의 전이 계수 $e_k$는 `ek[k+1]`에 저장한다. 따라서

$$
\texttt{ekc[k]}=\sum_{t=0}^{k-1}e_t,
\qquad\texttt{ekc[N]}=\texttt{accumulated\_ek.value}.
$$

저장될 같은 행에서도 `dk`는 해당 상태의 함수값 계수이고, `ek`는 직전 상태에서 넘어온 전이의 오차 계수이다. 9단계는 이 차이를 유지하여 그래프와 저장 열을 해석한다.

## 9. Plotting, numerical analysis

### Figure 2와 결과 저장

논문 Figure 2는 $a_k,b_k,d_k,e_k$를 비교한다. 원본 Python은 $c_k$와 누적 오차 `ekc`를 포함하여 여섯 패널을 그린다. Figure 2의 곡선과 `relax` 값의 대응은 다음과 같다.

| `relax` | Figure 2의 곡선 |
| --- | --- |
| 0 | 갈색 실선 |
| 1 | 빨간 점선 |
| 2 | 파란 점선 |
| 3 | 보라 점선 |

한 번 실행하면 선택한 `relax` 값의 결과를 그린다. 네 설정을 비교하려면 같은 실험 조건에서 `relax`를 바꾸어 각각 실행하고, 결과와 `relax` 값을 함께 보관해 같은 축에 그린다.

출력 관련 기본값은 `pplot=1`, `ssave=1`, `folder="SaveData/"`, `nname="SGD.dat"`이다. `pplot=1`이면 8단계의 계수를 추출하여 그래프를 그리며, 그 안에서 `ssave=1`이면 `SaveData/SGD.dat`에 저장한다. `pplot=0`이면 저장도 생략한다. 여러 `relax` 값을 비교할 때는 이전 실행의 결과를 별도 파일로 보관한다.

원본 그래프의 가로축은 `np.arange(1, N+2)`이다. 따라서 파일의 `k` 열과 그래프의 표시 인덱스는 $1,\ldots,N+1$이며, 논문의 상태 인덱스보다 1 크다.

### Plotting 배열과 저장 열의 대응

| 출력 배열 | 논문에서 나타내는 양 | 저장되는 위치 |
| --- | --- | --- |
| `ak,bk,ck,dk` | $a_k,b_k,c_k,d_k$ | 상태 $k=0,\ldots,N$ |
| `ek` | $e_k$ | 표시용 0 뒤에 전이 $k=0,\ldots,N-1$의 계수 |
| `ekc` | $\sum_{t=0}^{k-1}e_t$ | 상태 $k$까지의 누적합 |

아래 저장 코드는 원본의 `if pplot:` 내부에 두며, 8단계에서 생성한 배열을 사용한다.

```python
if ssave:
    labels = ['k', 'ak', 'bk', 'ck', 'dk', 'ek', 'ekc']
    data = np.column_stack((np.arange(1, N + 2), ak, bk, ck, dk, ek, ekc))
    saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
```

### 논문 표의 $e_N$와 누적 오차

논문 p.8의 기본 수치 실험 보고값은 $d_N^{\mathrm{opt}}\approx201$, $e_N^{\mathrm{opt}}\approx100.5$이다. [원본 SGD.dat](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/2_Stochastic_BoundedVariance/SaveData/SGD.dat)의 마지막 행에는 `dk=201`, `ek=100.5`, `ekc=2057.5`가 저장되어 있다. 여기서 `ek`는 마지막 전이의 오차 계수이고, `ekc`는 전체 누적합 $\sum e_k$이다. $100.5$를 누적 오차로 해석하지 않는다.

논문 표의 $e_N$를 코드의 마지막 개별 오차 계수와 대응시킬 때, 이 문서의 0 기반 전이 인덱스에서는 $e_{N-1}$이다. 앞의 수치는 논문과 저장 데이터에 보고된 값이며, 실행 결과는 `relax` 값과 `reoptimize` 설정에 따라 달라진다. 특히 `reoptimize=0`의 오차 계수는 누적합을 최소화한 결과가 아니다.

`relax=3`의 potential은 Theorem 5의

$$
\phi_k^f=d_k(f(x_k)-f_\star)+\frac L2\|x_k-x_\star\|^2
$$

와 같은 형태이다. 정리의 계수 선택은 다음과 같다.

$$
d_{k+1}=d_k+\delta_kL,\qquad
e_k=\frac{\delta_k^2L}{2}(1+d_{k+1}),\qquad
\delta_kd_{k+1}\ge e_k.
$$

$\delta_k=1/L$, $d_0=0$을 대입하면 $d_k=k$, $e_k=(k+2)/(2L)$이고,

$$
\sum_{k=0}^{N-1}e_k=\frac{N(N+3)}{4L}
$$

이다. 이 계수는 정리에서 제시한 해석적 선택이다. `relax=3`은 potential의 형태만 고정하므로, SDP에서 얻은 계수는 이 수열과 다를 수 있다.

각 `relax` 값의 함수값 bound를 비교할 때는 $d_N$, 개별 $e_k$, 누적합 $\sum e_k$와 비율 $(\sum e_k)/d_N$을 함께 사용한다. 2단계의 bound에서 초기 거리항의 계수는 $L/(2d_N)$, variance의 계수는 $(\sum e_k)/d_N$이다.
