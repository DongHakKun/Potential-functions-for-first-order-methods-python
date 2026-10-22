# 실험 5. Overparameterized Models — §4, Theorem 8, Figure 5

이 문서는 [A_ParameterSelection.py](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/3_Stochastic_Overparametrized/A_ParameterSelection.py)의 수치 실험을 설명한다. 논문 §4, Appendix F.2의 식 (22)와 Figure 5, Appendix C와 D.5의 line-search 및 $S_k$를 이용한 거리항 표현 구성을 다룬다. 논문의 대응 위치는 pp.11–12, 37–39, 51–52이다.

`relax=0,1,2,3`에 따른 중간 계수와 초기·최종 계수 조건을 1–2단계에서 설명한다. 3–5단계에서는 Gram 표현과 행렬 조건을 살펴보고, 6단계에서 이를 CVXPY로 구현한다. 7–9단계에서는 최적화 결과에서 potential 계수와 알고리즘 파라미터를 구하고 그래프로 확인한다.

수식의 상태 인덱스는 $k=0,\ldots,N$이고 전이 인덱스는 $k=0,\ldots,N-1$이다. Python 딕셔너리는 1부터 시작하므로 $q_{r,k},a_k,d_k$는 `Q[k+1][r-1]`, `a[k+1]`, `d[k+1]`에 대응한다. 전이 $k$의 multiplier와 PSD 변수는 `lam[k+1]`, `mu_Y[k+1]`, `mu_X[k+1]`, `S[k+1]`에 저장한다. 아래 코드 블록의 루프 변수 `k`는 원본과 같이 1부터 시작한다.

수식의 $i$는 stochastic outcome, $\ell$은 component를 나타낸다. 논문 Appendix F.2의 potential과 알고리즘 기호를 사용하며, 3–6단계의 행렬 계산은 원본 Python 코드의 배열 이름으로 설명한다.

## 1. Parameterized Potential

이 실험에서는 다음 조건을 만족하는 overparameterized finite-sum 문제를 다룬다.

$$
f(x)=\frac1n\sum_{\ell=1}^nf_\ell(x),\qquad
f_\ell\in\mathcal F_L,\qquad f_\ell'(x_\star)=0\quad(\ell=1,\ldots,n)
$$

앞의 bounded variance setting과 달리, 모든 component의 gradient가 공통 최적점에서 0이라고 가정한다. 이 가정은 component별 interpolation에 반영하며, 별도의 variance 항 $e_k\sigma^2$나 행렬 $A_{\mathrm{var}}$는 사용하지 않는다.

Appendix F.2에서는 다음 potential의 형태를 정하고 그 계수를 찾는다. 전체 gradient뿐 아니라 component gradient의 norm 제곱도 포함한다는 점이 앞의 실험과 다르다.

$$
\begin{aligned}
\phi_k^f={}&q_{1,k}\|x_k-x_\star\|^2
+q_{2,k}\|f'(x_k)\|^2
+q_{3,k}\frac1n\sum_\ell\|f_\ell'(x_k)\|^2\\
&+q_{4,k}\langle f'(x_k),x_k-x_\star\rangle
+d_k(f(x_k)-f_\star)+a_k\|z_k-x_\star\|^2.
\end{aligned}
$$

| 수식의 기호 | Python 표현 | 의미 |
| --- | --- | --- |
| $L,\mu,n,N$ | `L=1`, `m=0`, `n=2`, `N=100` | 기본 설정 |
| $(q_{1,k},\ldots,q_{4,k})$ | `Q[k+1]` | 네 potential 계수를 담은 벡터 |
| $a_k$ | `a[k+1]` | $\|z_k-x_\star\|^2$의 계수 |
| $d_k$ | `d[k+1]` | 함수값 차이 계수 |
| $d_N$ | `tau` | 최대화할 최종 계수 |
| $\tau_k,\alpha_k,\delta_k,\gamma_k$ | `tauk, alphak, deltak, gammak` | 해에서 복원할 알고리즘 파라미터 |

앞의 네 실험에서 `Q[k]`는 2×2 행렬이었지만, 여기서는 네 potential 계수를 담은 벡터이다. `tau`는 최종 함수값 계수 $d_N$을 나타내며, 알고리즘 파라미터 $\tau_k$를 저장하는 배열 `tauk`와는 역할이 다르다.

파일 첫머리 주석에는 세 번째 항의 계수가 `2 q3k`로 적혀 있지만, 실제 연산은 `Q[k][2] * term3_k`이다. 위 식은 논문 F.2와 실제 연산에 맞춰 $q_{3,k}$를 사용했다. 교차항의 계수도 $q_{4,k}$이며, 앞의 실험에서 사용한 $2c_k$와는 표기 방식이 다르다.

### `relax` 값에 따른 중간 계수

`relax`는 0, 1, 2, 3 중 하나이며 기본값은 0이다. 기본 실험 설정은 $L=1,\mu=0,n=2,N=100$이다. `relax` 값에 따른 계수 조건은 다음과 같다.

| `relax` | 고정하는 중간 계수 ($1\le k\le N-1$) | 자유로운 중간 계수 | 알고리즘 관련 추가 조건 |
| --- | --- | --- | --- |
| 0 | 없음 | $q_{1,k},\ldots,q_{4,k},a_k,d_k$ | 없음 |
| 1 | $a_k=L/2$ | $q_{1,k},\ldots,q_{4,k},d_k$ | 없음 |
| 2 | $a_k=L/2$ | $q_{1,k},\ldots,q_{4,k},d_k$ | `mu_X[k+1][1] == 0` |
| 3 | $a_k=L/2,\ q_{1,k}=\cdots=q_{4,k}=0$ | $d_k$ | `mu_X[k+1][1] == 0`을 추가하지 않음 |

`relax=2`에서는 전이마다 `mu_X[k+1][1] == 0`을 추가한다. `mu_X[k+1][0]`이 0이 아니면 8단계의 비율은 $\alpha_k=0$이다. 분모가 0인 경우는 8단계에서 설명한다. `relax=3`의 주석에도 $\alpha_k=0$이라고 적혀 있으나, 코드에서 이 등식을 추가하는 분기는 `relax == 2`뿐이다. Figure 5에서는 `relax=0,1,2`의 결과를 비교한다.

표의 “자유”는 SDP의 결정변수라는 뜻이다. 이 계수들도 5단계의 LMI와 6단계의 전체 제약을 만족해야 한다. 예를 들어 $a_k$는 $S_{k-1}\succeq0$, $(S_{k-1})_{11}=a_k$의 영향을 받는다. 표는 중간 상태의 계수를 다루며, $k=0,N$의 초기·최종 계수 조건은 2단계에서 설명한다.

## 2. Potential Inequality

앞의 실험에서는 알고리즘을 고정한 채 potential의 계수를 찾았다. 여기서는 다음 template의 $\tau_k,\alpha_k,\delta_k,\gamma_k$도 함께 정한다.

$$
\begin{aligned}
y_{k+1}&=(1-\tau_k)x_k+\tau_kz_k,\\
x_{k+1}^{(i)}&=y_{k+1}-\alpha_kf_i'(y_{k+1}),\\
z_{k+1}^{(i)}&=(1-\delta_k)y_{k+1}+\delta_kz_k-\gamma_kf_i'(y_{k+1}).
\end{aligned}
$$

이때 각 단계에서 만족해야 할 potential inequality는 다음과 같다.

$$
\boxed{\mathbb E_i[\phi_{k+1}^f(x_{k+1}^{(i)},z_{k+1}^{(i)})]
\le\phi_k^f(x_k,z_k)}.
$$

네 파라미터를 `cp.Variable`로 선언하고 갱신식을 potential에 직접 대입하면, 파라미터와 potential 계수 사이에 곱이 생긴다. 따라서 그대로는 SDP가 되지 않는다. 논문은 이를 피하기 위해 먼저 다음 두 line-search를 고려한다.

$$
\begin{aligned}
y_{k+1}&=\underset{x\in x_k+\operatorname{span}\{z_k-x_k\}}{\operatorname{argmin}}f(x),\\
x_{k+1}^{(i)}&=\underset{x\in y_{k+1}+\operatorname{span}\{f_i'(y_{k+1})\}}{\operatorname{argmin}}f(x)
\end{aligned}
$$

이 line-search들의 optimality condition을 multiplier로 결합하고, $z_{k+1}^{(i)}$의 거리항은 $S_k$를 이용한 거리항 표현으로 표현한다. 그렇게 얻은 SDP를 푼 뒤 multiplier와 `S[k]`에서 원래 template의 네 파라미터를 복원한다. 각각의 과정은 4·5절과 8절에서 연결된다.

### 초기·최종 potential

공통 초기·최종 계수 조건은 다음과 같다.

$$
q_{1,0}=\cdots=q_{4,0}=q_{1,N}=\cdots=q_{4,N}=0,\qquad
d_0=0,\qquad a_0=L/2,\qquad d_N=\texttt{tau}.
$$

최종 거리항의 계수는 `relax` 값에 따라 다음과 같이 달라진다.

$$
\boxed{
a_N=\begin{cases}
0,&\texttt{relax}=0,\\
L/2,&\texttt{relax}\in\{1,2,3\}.
\end{cases}}
$$

따라서 초기·최종 potential은 다음과 같다.

$$
\phi_0^f=\frac L2\|z_0-x_\star\|^2,
\qquad
\phi_N^f=\begin{cases}
d_N(f(x_N)-f_\star),&\texttt{relax}=0,\\
d_N(f(x_N)-f_\star)+\dfrac L2\|z_N-x_\star\|^2,
&\texttt{relax}\in\{1,2,3\}.
\end{cases}
$$

초기 상태는 $z_0=x_0$로 둔다. 코드의 `tau`는 최종 함수값 계수 $d_N$을 나타내며, 알고리즘 파라미터 $\tau_k$와 구분한다. `a[N+1]`에는 위 식의 $a_N$을 사용한다.

각 단계의 부등식에 기대값을 취하고 더하면 초기 potential이 최종 potential의 기대값을 제어한다. `relax=1,2,3`의 최종 거리항도 비음수이므로, $d_N>0$인 해에서는 같은 형태의 함수값 오차 bound를 얻는다. SDP는 각 설정의 초기·최종 계수 조건 아래에서 $d_N$을 최대화한다.

## 3. Finite-dimensional Gram representation

일반성을 잃지 않고 $x_\star=0$, $f_\ell(x_\star)=0$으로 둔다. 공통 최적점 가정에 따라 $f_\ell'(x_\star)=0$이다. 아래 $P,F$의 열 배치는 원본 Python 코드의 주석을 따른다.

$$
\begin{aligned}
P=[&x_k\ z_k\ y_{k+1}\mid
x_{k+1}^{(1)}\ \cdots\ x_{k+1}^{(n)}\mid
f_1'(x_k)\ \cdots\ f_n'(x_k)\mid\\
&f_1'(y_{k+1})\ \cdots\ f_n'(y_{k+1})\mid
\{f_\ell'(x_{k+1}^{(i)})\}_{i=1,\ldots,n;\,\ell=1,\ldots,n}],\\
F=[&\{f_\ell(x_k)\}_\ell\mid\{f_\ell(y_{k+1})\}_\ell\mid
\{f_\ell(x_{k+1}^{(i)})\}_{i,\ell}],\\
G={}&P^\top P\in\mathbb S_+^{3+3n+n^2}.
\end{aligned}
$$

$n=2$이면 $P$의 벡터는 13개, $F$의 함수값은 8개이며 interpolation point는 최적점을 포함해 5개이다.

```python
dimG = 3 + 3*n + n**2
dimF = 2*n + n**2
nbPts = n + 3
```

| Python 표현 | 나타내는 값 | $n=2$에서 고르는 열 |
| --- | --- | --- |
| `xk`, `zk`, `yk1` | $x_k,z_k,y_{k+1}$ | $P$의 1, 2, 3번째 열 |
| `xk1[i-1]` | $x_{k+1}^{(i)}$ | $P$의 4, 5번째 열 |
| `gxk[l-1]` | $f_\ell'(x_k)$ | $P$의 6, 7번째 열 |
| `gyk1[l-1]` | $f_\ell'(y_{k+1})$ | $P$의 8, 9번째 열 |
| `gxk1[l-1,:,i-1]` | $f_\ell'(x_{k+1}^{(i)})$ | $P$의 10–13번째 열 |

`yk1`와 `xk1`는 고정된 step-size로 계산한 점이 아니라 독립된 열을 고르는 배열이다. 이 점들의 관계는 line-search 조건으로 반영한다.

```python
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
GXS = np.sum(gxs, axis=0)/n
```

`gxk1`의 첫 축은 component, 마지막 축은 outcome이다. $n=2$에서 shape은 `(2,13,2)`이다. `GXK`, `GYK1`, `GXK1`은 각 점에서 component gradient를 평균 내어 전체 gradient를 나타낸다.

```python
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
```

`fxk`, `fyk1`, `fxk1`도 component와 outcome을 같은 순서로 배치한다. Component 평균은 `FXK`, `FYK1`, `FXK1`에 저장한다. 코드의 `G`, `F` 딕셔너리는 이 배열들을 interpolation에 연결하며, 수식의 Gram matrix와 함수값 행렬과 구분한다.

### Potential의 각 항 계산

원본 코드의 `term1_k`–`term4_k`는 차례로 거리 제곱, 전체 gradient의 norm 제곱, component gradient의 norm 제곱 평균, 교차항을 Gram 표현으로 계산한다. `term1_k1`–`term4_k1`은 다음 점에서 계산한 같은 항의 outcome 평균이다.

```python
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
    term3_k += np.outer(gxk[i-1, :], gxk[i-1, :])/n
term3_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    for j in range(1, n+1):
        term3_k1 += np.outer(gxk1[i-1, :, j-1], gxk1[i-1, :, j-1])/n**2
term4_k = np.outer(xk-xs, GXK)
term4_k = (term4_k + term4_k.T)/2
term4_k1 = np.zeros((dimG, dimG))
for i in range(1, n+1):
    temp = np.outer(xk1[i-1, :]-xs, GXK1[i-1, :])
    temp = (temp + temp.T)/2
    term4_k1 += temp/n
```

`term2_k1`은 각 점에서 component gradient를 먼저 평균 낸 뒤 outer product를 구하고, 이를 outcome에 대해 평균 낸다. `term3_k1`은 component마다 outer product를 만든 뒤 component와 outcome 모두에 대해 평균 낸다. 그래서 후자의 계수는 $1/n^2$이다. `term4_k`와 `term4_k1`은 `(temp + temp.T)/2` 방식으로 대칭화한다.

## 4. Interpolation matrices + multipliers

각 component $f_\ell$에 Appendix D.4의 smooth convex interpolation 조건을 적용한다.

$$
f_\ell(x_p)-f_\ell(x_q)-\langle f_\ell'(x_q),x_p-x_q\rangle
-\frac1{2L}\|f_\ell'(x_p)-f_\ell'(x_q)\|^2\ge0.
$$

각 component에 대해 $x_\star,x_k,y_{k+1},x_{k+1}^{(1)},x_{k+1}^{(2)}$의 다섯 점을 사용한다. `X`, `G`, `F`는 위치·해당 component의 gradient·함수값 배열을 같은 키에 저장한다. Component 두 개에서 서로 다른 순서쌍 20개씩, 총 40개의 interpolation inequality를 사용한다.

행렬 `M`은 다음과 같다. 기본 설정 `m=0`에서는 논문 식 (19)의 행렬과 일치한다.

```python
M = np.array([[-L*m, L*m, m, -L],
              [L*m, -L*m, -m, L],
              [m, -m, -1, 1],
              [-L, L, 1, -1]])/(2*(L-m))
```

`TT = np.vstack((xi, xj, gi, gj))`로 네 벡터를 쌓고 `TT.T @ M @ TT`와 `fi-fj`를 구한다. 같은 `lam[k][i-1,j-1,l-1]`을 두 항에 곱해 `cons_SDP[k]`, `cons_LIN[k]`에 더한다. `lam[k] >= 0`은 원소별 부호 조건이고, 대각 원소는 합에서 제외한다.

### Line-search 등식

Appendix F.2의 line-search에서 얻는 직교 조건을 직접 쓰면 다음과 같다.

$$
\begin{aligned}
\langle f'(y_{k+1}),x_k-y_{k+1}\rangle&=0,\\
\langle f'(y_{k+1}),z_k-x_k\rangle&=0,\\
\langle f'(x_{k+1}^{(i)}),y_{k+1}-x_{k+1}^{(i)}\rangle&=0,\\
\langle f'(x_{k+1}^{(i)}),f_i'(y_{k+1})\rangle&=0.
\end{aligned}
$$

원본 코드의 `A[1],A[2]`와 `B[1,i],B[2,i]`가 이 네 내적의 대칭행렬 표현이다. 두 번째 line-search의 탐색 방향은 component gradient이고, 최소화하는 함수는 전체 함수 $f$이므로 마지막 내적에는 두 gradient가 함께 들어간다.

```python
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
```

등식에 곱하는 `mu_Y[k]`, `mu_X[k]`에는 부호 제약을 두지 않는다. `A`는 outcome과 무관하며, `B`는 outcome별로 계산한 뒤 $1/n$씩 더한다. 구현에서는 outcome 간 symmetry에 따라 `mu_X[k]`의 두 계수를 공통으로 사용한다. 6단계의 `l`은 component, interpolation 안의 `i,j`는 점 번호를 가리킨다.

## 5. Linear equality + semidefinite inequality

### $S_k$를 이용한 다음 거리항의 표현

Appendix D.5, p.39는 거리항에 생기는 파라미터 곱을 3×3 PSD 변수 $S_k$로 옮긴다. 원본 Python 코드는 이 구성을 Appendix F.2의 component gradient와 outcome 평균에 적용한다.

다음 거리항의 파라미터 부분은

$$
a_{k+1}
\begin{pmatrix}1\\\delta_k\\-\gamma_k\end{pmatrix}
\begin{pmatrix}1&\delta_k&-\gamma_k\end{pmatrix}
$$

이다. 여기에 대응하는 원본 변수 `S[k]`를 대칭 3×3 결정변수로 선언하고 `S[k] >> 0`, `S[k][0,0] == a[k+1]`을 적용한다. 수식의 단계 $k$로 쓰면 $S_k\succeq0$, $(S_k)_{11}=a_{k+1}$이다.

각 outcome에서는 `TT = np.column_stack((yk1, zk-yk1, gyk1[i-1,:]))`를 만들고 `TT @ S[k] @ TT.T`를 계산한다. 이를 $1/n$씩 더한 `dsstep_avg`가 다음 거리항의 이차항 행렬이다. `TT`는 고정된 배열이므로 이 식은 `S[k]`에 대해 affine이다. 별도의 rank-one 제약 없이 해에서 파라미터를 복원하는 이유는 Appendix D.5와 8단계에 설명되어 있다.

### 전체 제약

`cons_SDP[k]`에는 현재 거리항을 빼고 `dsstep_avg`를 더한다. 이어서 3단계의 `term1_k`–`term4_k`와 `term1_k1`–`term4_k1`에 각각 현재·다음 potential 계수를 곱해 차이를 더한다. 4단계의 line-search 등식과 component별 interpolation 항도 같은 행렬에 더한다.

`cons_LIN[k]`에는 현재 함수값 계수 `-d[k]*statesKF`와 다음 함수값 계수 `d[k+1]*statesK1F`를 더한 뒤, 각 interpolation의 `lam[k][i-1,j-1,l-1]*(fi-fj)`를 더한다.

$n=2$에서 이차항은 13×13이고 함수값 계수는 길이 8이다. Potential에서 온 함수값 계수를 $F$의 열 순서대로 쓰면 다음과 같다.

$$
\left(-\frac{d_k}2,-\frac{d_k}2,0,0,
\frac{d_{k+1}}4,\frac{d_{k+1}}4,\frac{d_{k+1}}4,\frac{d_{k+1}}4\right).
$$

현재 함수값에는 component 평균의 $1/2$가 붙고, 다음 함수값에는 component와 outcome 평균의 $1/4$가 붙는다. Potential에 없는 $f_\ell(y_{k+1})$도 interpolation에는 사용하므로 해당 계수까지 함께 소거한다.

모든 항을 더한 뒤 `cons_SDP[k] << 0`, `cons_LIN[k] == 0`, `lam[k] >= 0`과 앞의 `S[k]` 조건을 적용한다. 함수값 계수를 소거하고 남은 이차항을 NSD로 제한하여 potential inequality를 보장한다. 계산식과 제약 등록은 6단계의 원본 코드에 나와 있다.

## 6. SDP formulation in CVXPY

논문 식 (22)에 따라 최종 함수값 계수 $d_N$을 최대화한다. 3–4단계의 Gram 표현과 상수행렬로 5단계의 제약식을 CVXPY 제약으로 옮기고, 초기·최종 계수 조건과 `relax` 값에 따른 계수 조건도 함께 적용한다. 모든 행렬 원소와 선형 등식은 결정변수에 대해 affine이다.

### CVXPY 변수와 제약의 조립 순서

코드에서는 먼저 `tau`와 초기·최종 계수를 정하고 딕셔너리를 채운다. 이후 전이마다 중간 계수·multiplier·$S_k$를 이용한 거리항 표현 변수를 선언하고, $S_k$를 이용한 거리항 표현 항, potential 차이, line-search 항, interpolation 항을 차례로 더한다. 마지막으로 PSD·NSD 조건, 선형 등식과 `relax` 값에 따른 조건을 추가한다.

아래 코드는 원본의 선언과 행렬 연산을 이 순서로 모은 것이다. 3–4단계에서 정의한 벡터와 행렬 `M`, `A`, `B`, `term1_k`–`term4_k1`을 사용한다.

```python
a0 = L/2
q10 = 0
q20 = 0
q30 = 0
q40 = 0
d0 = 0
tau = cp.Variable(name='tau')  # variable to be maximized (problem (3))
aN = 0
q1N = 0
q2N = 0
q3N = 0
q4N = 0
dN = tau
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

Q, d, a = {}, {}, {}
lam, mu_Y, mu_X, S = {}, {}, {}, {}
cons_SDP, cons_LIN = {}, {}
Q[1] = np.array([q10, q20, q30, q40])
d[1] = d0
a[1] = a0
Q[N+1] = np.array([q1N, q2N, q3N, q4N])
d[N+1] = dN
a[N+1] = aN
statesKF = FXK-FXS
statesK1F = np.sum(FXK1, axis=0)/n-FXS
cons = [tau >= 0]

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

    S[k] = cp.Variable((3, 3), symmetric=True, name=f'S_{k}')
    dsstep_avg = np.zeros((dimG, dimG))
    for i in range(1, n+1):
        # E_i||z_{k+1}^(i)-x*||^2, with the original full 3x3 S_k.
        TT = np.column_stack((yk1, zk-yk1, gyk1[i-1, :]))
        dsstep = TT @ S[k] @ TT.T
        dsstep_avg = dsstep_avg + dsstep/n

    cons_SDP[k] = (
        -a[k]*np.outer(zk-xs, zk-xs) + dsstep_avg
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

    if relax == 2:
        cons.append(mu_X[k][1] == 0)
    cons.append(S[k] >> 0)
    cons.append(S[k][0, 0] == a[k+1])
    cons.append(cons_SDP[k] << 0)
    cons.append(cons_LIN[k] == 0)
    cons.append(lam[k] >= 0)
obj = tau
```

마지막 `k=N`에서는 `if k < N`을 건너뛰고 2단계의 최종 potential 계수를 사용한다. 코드의 `Q[k]`와 `Q[k+1]`은 각각 수식의 $q_{r,k-1}$과 $q_{r,k}$를 담는다. `relax=2`의 `mu_X[k][1] == 0`은 모든 전이에 추가하고, `relax=3`에는 추가하지 않는다.

Interpolation 루프의 `l`은 component를, `i,j`는 interpolation point를 나타낸다. `X`, `G`, `F`는 각각 점의 위치, 해당 component의 gradient, 함수값을 고르는 배열의 딕셔너리이다. 각 순서쌍의 동일한 multiplier를 이차항 `TT.T @ M @ TT`와 함수값 계수 `fi-fj`에 곱해 더한다.

### 전체 SDP와 결정변수 규모

목적함수는 $\max d_N$이다. 모든 전이에 5단계의 NSD·선형 등식·multiplier 부호 조건과 $S_k\succeq0$, $(S_k)_{11}=a_{k+1}$을 적용한다. `relax=2`에서는 `mu_X[k][1] == 0`도 추가한다.

이 SDP의 결정변수는 중간 potential 계수 $q_{1,k},\ldots,q_{4,k},a_k,d_k$, 최종 $d_N$, 단계별 `lam`, `mu_Y`, `mu_X`, `S`이다. 알고리즘의 네 파라미터는 이 해에서 복원하므로, SDP 안에서는 파라미터와 potential 계수를 직접 곱하지 않는다.

$n=2,N=100$의 원래 SDP 규모는 다음과 같다.

| 항목 | 규모 |
| --- | --- |
| NSD LMI `cons_SDP[k] << 0` | 13×13, 100개 |
| PSD 변수 $S_k\succeq0$ | 3×3, 100개 |
| 함수값 scalar linear equality | 800개 |
| $(S_k)_{11}=a_{k+1}$ | 100개 |
| 선언된 interpolation multiplier | 5000개, 그중 4000개를 interpolation에 사용 |
| 부호 제한이 없는 line-search multiplier | 400개 |
| 대칭 PSD 변수의 독립 원소 | 600개 |

기본 설정의 독립 scalar 변수는 총 $5000+400+600+396+99+99+1=6595$개이다. 앞의 세 항은 interpolation multiplier, line-search multiplier, PSD 변수에서 나오고, 뒤의 네 항은 중간 $q,a,d$와 최종 $d_N$에서 나온다.

| `relax` | 중간 $q$ | 중간 $a$ | 중간 $d$ | $\lambda$ | `mu_Y, mu_X` | $S$ | $d_N$ | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 99 | 99 | 5000 | 400 | 600 | 1 | 6595 |
| 1 | 396 | 0 | 99 | 5000 | 400 | 600 | 1 | 6496 |
| 2 | 396 | 0 | 99 | 5000 | 400 | 600 | 1 | 6496 |
| 3 | 0 | 0 | 99 | 5000 | 400 | 600 | 1 | 6100 |

표는 코드에서 선언한 변수 수를 센다. 대칭행렬 $S$는 단계마다 독립 원소 6개로 계산했다. `relax=2`에서는 선언 후 `mu_X[k][1] == 0`을 단계마다 추가하므로, 등식은 100개 늘어나지만 표의 변수 수는 `relax=1`과 같다.

## 7. MOSEK numerical optimization

기본 설정은 `solver="MOSEK"`, `tolerance=1e-8`이다. 이 값을 `MSK_DPAR_INTPNT_CO_TOL_PFEAS` 옵션으로 전달해 MOSEK의 primal feasibility tolerance를 지정한다.

아래의 `solve`는 원본 Python 파일과 같은 디렉터리의 `saveData.py`에서 가져온 함수이다. MOSEK를 선택하면 `_MatlabOrderMOSEK` adapter가 변수 순서와 PSD 계수 전달 방식을 MATLAB/YALMIP에 맞춘다. 이때 canonicalization과 해 복원은 CVXPY가 그대로 담당하며, 다른 solver를 선택하면 일반 CVXPY 경로로 계산한다.

계산은 다음 순서로 진행된다.

1. CVXPY가 affine objective와 선형·PSD 제약을 cone problem으로 변환한다.
2. MOSEK가 이 유한 차원 SDP의 수치해를 계산한다.
3. CVXPY가 계산된 해를 원래 `cp.Variable`의 `.value`에 저장한다.

이 계산에서 구하는 것은 potential의 계수와 제약식을 구성하는 multiplier, PSD 변수이다. 특정 데이터셋에 알고리즘을 실행해 iterate의 궤적을 얻는 과정은 포함되지 않는다.

최적화 직전에는 `solver_cons=list(cons)`로 제약 목록을 복사한다. MOSEK는 앞서 구성한 13×13 NSD와 3×3 PSD 제약을 그대로 사용한다.

CLARABEL·SDPA 분기는 `m==0`과 초기·최종 계수 조건을 확인한 뒤, 제약에 의해 0이 되는 행·열을 별도의 등식으로 분리한다. 문제를 바꾸는 것이 아니라 같은 제약을 다른 형태로 전달하는 과정이다.

```python
solverDetails = cp.Problem(cp.Minimize(-obj), solver_cons)
solve(solverDetails, **solver_opt)
print('Elapsed time:', time.perf_counter()-tic)
print('Solver status:', solverDetails.status, 'objective:', solverDetails.value)
```

objective를 `cp.Minimize(-obj)`로 정의했으므로 `solverDetails.value`에는 $-d_N$이, `tau.value`에는 최대화한 $d_N$이 들어 있다. 이 실험은 한 번의 최적화로 끝난다. bounded variance 실험처럼 $\sum e_k$를 최소화하는 두 번째 문제는 없다.

## 8. Optimal coefficients / algorithm parameters

### Potential 계수 추출

원본의 출력 조건 `pplot=1`에서 이 단계의 배열 생성·계수 추출·비율 계산을 수행한다. `pplot=0`이면 원본 코드와 같이 이 후처리와 9단계의 plotting·저장을 실행하지 않는다. Potential 계수와 파라미터 비율을 저장할 배열은 길이 $N+1$로 초기화한다.

```python
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
for i in range(1, N+2):
    apk[i-1] = a[i].value if isinstance(a[i], cp.Expression) else a[i]
    Q1[i-1] = Q[i][0].value if isinstance(Q[i], cp.Expression) else Q[i][0]
    Q2[i-1] = Q[i][1].value if isinstance(Q[i], cp.Expression) else Q[i][1]
    Q3[i-1] = Q[i][2].value if isinstance(Q[i], cp.Expression) else Q[i][2]
    Q4[i-1] = Q[i][3].value if isinstance(Q[i], cp.Expression) else Q[i][3]
    dk[i-1] = d[i].value if isinstance(d[i], cp.Expression) else d[i]
```

`Q[i][0]`–`Q[i][3]`에서 $q_{1,i-1}$–$q_{4,i-1}$를 읽고, `a[i]`에서 읽은 거리항 계수 $a_{i-1}$은 `apk[i-1]`에 저장한다. CVXPY 변수이면 `.value`를 사용하고, `relax=3`의 `Q`나 `relax=1,2,3`의 `a`처럼 고정한 항이면 상수 자체를 읽는다.

### 알고리즘 파라미터 복원

SDP의 해에서 파라미터를 복원하는 원본 코드는 다음과 같다. `mu_Y`, `mu_X`의 비율로 $\tau_k,\alpha_k$를 구하고, `S`의 첫 행의 비율로 $\delta_k,\gamma_k$를 구한다. 각 비율은 분모가 0이 아닐 때 파라미터를 정한다.

```python
for i in range(1, N+1):
    tauk[i-1] = np.divide(mu_Y[i].value[1], mu_Y[i].value[0])
    alphak[i-1] = -np.divide(mu_X[i].value[1], mu_X[i].value[0])
    deltak[i-1] = np.divide(S[i].value[0, 1], S[i].value[0, 0])
    gammak[i-1] = -np.divide(S[i].value[0, 2], S[i].value[0, 0])
```

앞의 두 비율은 line-search multiplier에서, 뒤의 두 비율은 $S_k$의 첫 행에서 나온다. 부호는 4단계의 $x_k-y_{k+1}$, $y_{k+1}-x_{k+1}^{(i)}$ 방향에 맞춰 정해진다.

$(S_k)_{11}=a_{k+1}>0$이면 첫 행에서 복원한 $\delta_k,\gamma_k$로 원래 형태의 rank-one 행렬을 만들 수 있다. $S_k$에서 이 행렬을 뺀 차이가 PSD이므로, 복원한 거리항은 $S_k$로 나타낸 거리항보다 커지지 않는다. 따라서 앞에서 얻은 제약식을 유지할 수 있다. 이것이 Appendix D.5에서 rank-one 제약을 직접 두지 않는 이유이다.

### 분모가 0인 비율의 출력과 해석

원본 코드는 모든 전이에서 위 `np.divide`를 계산한다. 분모가 0이면 이 비율만으로 파라미터를 정할 수 없다. 이때 배열에 나타나는 `NaN`, 무한대 또는 수치 오차로 인한 큰 값은 유효한 알고리즘 파라미터가 아니다.

특히 `relax=0`의 마지막 전이에서는 2단계의 $a_N=0$과 5단계의 $S_k$의 조건에 의해

$$
(S_{N-1})_{11}=0.
$$

정확한 PSD 행렬에서 이 대각원소가 0이면 대응하는 첫 행·열도 0이다. 따라서 $(S_{N-1})_{12}=(S_{N-1})_{13}=0$이고, 마지막 $\delta_{N-1},\gamma_{N-1}$의 식은 $0/0$이다. 최종 potential에 $z_N$의 거리항이 없으므로 이 두 파라미터는 해당 제약식의 비율에서 결정되지 않는다. 코드가 마지막 전이에도 비율을 계산하므로, 그래프를 볼 때 이 값을 중간 전이에서 얻은 파라미터와 구분해야 한다.

`mu_Y` 또는 `mu_X`의 첫 번째 성분이 0인 경우에도 그 비율만으로 유한한 파라미터를 정할 수 없다. `relax=2`는 `mu_X[k][1] == 0`을 추가하므로 첫 번째 성분이 0이 아닌 전이에서는 $\alpha_k=0$이다. 두 성분이 모두 0이면 `np.divide`의 출력과 $\alpha_k=0$이라는 설정을 구분해야 한다.

### 복원식의 분자·분모와 부호

$y_{k+1}=(1-\tau_k)x_k+\tau_kz_k$를 첫 두 직교 조건의 가중합에 대입하면, gradient와 내적을 이루는 방향은 $z_k-x_k$의 배수가 된다. 그 계수가 0이 되도록 정한 값이 코드의 `mu_Y` 비율이다.

$x_{k+1}^{(i)}=y_{k+1}-\alpha_kf_i'(y_{k+1})$를 나머지 두 조건에 대입하면, 탐색 방향에 곱해지는 계수는 `mu_X`의 첫 성분에 $\alpha_k$를 곱한 값과 두 번째 성분의 합이다. 이 합을 0으로 두므로 $\alpha_k$의 복원식에는 음의 부호가 붙는다.

$S_k$에 대해서도 5절에서 출발한 rank-one 행렬을 다시 보면 두 비율을 확인할 수 있다.

$$
a_{k+1}
\begin{pmatrix}1\\\delta_k\\-\gamma_k\end{pmatrix}
\begin{pmatrix}1&\delta_k&-\gamma_k\end{pmatrix}.
$$

첫 행은 $(a_{k+1},a_{k+1}\delta_k,-a_{k+1}\gamma_k)$이다. 따라서 `S[k][0,1] / S[k][0,0]`으로 $\delta_k$를, `-S[k][0,2] / S[k][0,0]`으로 $\gamma_k$를 얻는다. 앞의 두 multiplier 비율과 함께, 각 분모가 0이 아닌 전이에서는 line-search 없이 원래 template을 실행할 네 파라미터가 정해진다. 분모가 0인 출력은 앞 절의 해석을 적용한다.

## 9. Plotting, numerical analysis

### Figure 5와 코드 출력

논문 Figure 5의 갈색 실선, 빨간 점선, 파란 점선은 각각 `relax=0,1,2`의 결과이다. 논문에서는 $q_{1,k}$가 약 $10^{-7}$로 작아 그래프에서 생략했다고 설명한다. Python 코드는 이 계수도 포함해 여섯 potential 계수와 네 알고리즘 파라미터의 비율 출력을 제시한다. 한 번 실행할 때는 선택한 설정 하나의 결과만 파란 선으로 그린다.

기본 설정은 $a_N=0$이므로 마지막 $S_k$의 $(1,1)$ 원소도 0으로 고정된다. 이때는 최종 potential에 $z_N$의 거리항이 없어 마지막 $\delta,\gamma$를 앞의 비율로 정할 수 없다.

수치해에서는 이 분모가 아주 작은 값으로 남아 큰 비율이 나올 수 있다. 논문 Figure 5도 최종 potential에서 $z_N$을 제어하지 않는다는 이유로 마지막 값의 표시 범위를 제한한다. 따라서 마지막 비율은 중간 단계의 파라미터와 구분해 읽는다.

### Theorem 8과 상태 기호의 대응

Figure 5의 parameter selection을 통해 얻은 단순한 구조가 Theorem 8의 primal averaging과 연결된다. 다만 Theorem 8과 Appendix F.2는 같은 상태를 서로 다른 기호로 나타내므로, 아래 대응을 적용한 뒤 파라미터를 비교한다.

| Theorem 8 원문의 상태 | 이 문서와 F.2의 상태 | 역할 |
| --- | --- | --- |
| $y_k$ | $x_k$ | averaged state |
| $x_k$ | $z_k$ | gradient-step state |

이 대응에 따라 Theorem 8의 기본 선택을 이 문서의 파라미터로 쓰면 다음과 같다.

$$
\tau_k=\frac1{k+1},\qquad\alpha_k=0,\qquad
\delta_k=1,\qquad\gamma_k=1/L,\qquad d_k=k
$$

이 값들을 2절의 algorithm template에 대입하면 다음 갱신식을 얻는다.

$$
\begin{aligned}
y_{k+1}&=\frac{k}{k+1}x_k+\frac1{k+1}z_k,\\
x_{k+1}^{(i)}&=y_{k+1},\\
z_{k+1}^{(i)}&=z_k-\frac1L f_i'(y_{k+1}).
\end{aligned}
$$

이 갱신식의 $x_k,z_k,y_{k+1}$는 F.2와 이 문서의 표기를 따른다. 위 해석적 선택에서는 $d_{100}=100$이므로 수치 결과와 비교할 기준이 된다. SDP는 이 값을 고정하지 않고, 선택한 `relax` 조건 아래에서 $d_{100}$을 최대화한다.

### 결과 저장과 인덱스

`pplot=1`이면 potential 계수를 추출하고 8단계의 파라미터 비율을 계산한 뒤 그래프를 그린다. 이때 `ssave=1`도 설정되어 있으면 [SaveData/ParameterSelection.dat](https://github.com/DongHakKun/Potential-functions-for-first-order-methods-python/blob/main/3_Stochastic_Overparametrized/SaveData/ParameterSelection.dat)에 다음 열을 저장한다.

```text
k, Q1, Q2, Q3, Q4, apk, dk, alphak, tauk, deltak, gammak
```

`Q1`–`Q4`에는 $q_{1,k}$–$q_{4,k}$가, `apk`에는 이 실험의 $a_k$가 들어간다. 앞의 averaging 파일에서 같은 이름의 `apk`가 $a_k'$를 저장했던 것과는 의미가 다르다.

Potential 계수는 상태 0,…,100에 대해 101개이고, 파라미터 비율을 계산하는 위치는 전이 0,…,99에 대해 100개이다. 코드는 함께 저장할 수 있도록 파라미터 배열도 길이 101로 초기화한다. 따라서 마지막 원소의 0은 비율 계산 결과가 아닌 초기화 값이다. 이 저장용 마지막 0과, `relax=0`의 마지막 전이에서 $\delta,\gamma$가 식별되지 않는 문제는 서로 구분한다.

### Plotting 배열과 저장 열의 대응

| 저장 열 | 논문에서의 의미 | 실제 값의 개수 |
| --- | --- | --- |
| `Q1,Q2,Q3,Q4` | $q_{1,k},\ldots,q_{4,k}$ | 상태 계수 101개씩 |
| `apk,dk` | $a_k,d_k$ | 상태 계수 101개씩 |
| `alphak,tauk,deltak,gammak` | 전이 $k$의 네 파라미터 비율 출력 | 비율 계산 위치 100개씩, 마지막은 초기화한 0; 분모가 0인 결과는 8단계에 따라 해석 |

```python
if ssave:
    labels = ['k', 'Q1', 'Q2', 'Q3', 'Q4', 'apk', 'dk',
              'alphak', 'tauk', 'deltak', 'gammak']
    data = np.column_stack((np.arange(1, N+2), Q1, Q2, Q3, Q4, apk, dk,
                            alphak, tauk, deltak, gammak))
    saveData(Path(__file__).resolve().parent / folder / nname, data, labels)
```

potential 계수의 그래프에는 `np.arange(1,N+2)`를 사용한다. 반면 알고리즘 파라미터의 그래프에는 `np.arange(1,N+1)`과 배열의 `[:-1]`를 사용해 마지막 초기화 값을 제외한다. 이렇게 상태 계수 101개와 전이 파라미터 100개의 서로 다른 인덱스 범위를 반영한다.
