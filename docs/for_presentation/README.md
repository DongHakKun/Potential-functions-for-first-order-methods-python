# Potential function 논문 학습 노트 — 수정본

## 읽는 순서

1. [3_1_1_Figure_1_Theorem_3.md](3_1_1_Figure_1_Theorem_3.md): GD, Figure 1, Theorem 3
2. [3_2_2_Figure_2_Theorem_5.md](3_2_2_Figure_2_Theorem_5.md): SGD, Figure 2, Theorem 5
3. [Averaging_expansion_Theorem_6-7_Table_2.md](Averaging_expansion_Theorem_6-7_Table_2.md): Averaging, Theorem 6·7, Table 2

원래 파일명과 설명 흐름을 유지하면서 문서별로 생략된 정의와 유도를 보강했어.

수식은 해당 절의 논문 표기를 사용하고, 구현을 설명할 때는 `1_Deterministic_SmoothConvex`부터 `5_Stochastic_VarianceAtOptimum`까지의 원본 `.py` 변수명을 사용해. 논문과 원본 코드의 계수나 인덱스가 다르면 그 위치에서 직접 대응시켜. 노트북의 실행 보조 변수는 논문·원본 구현의 수식 표기로 사용하지 않아.

## 주요 수정

| 문서 | 반영 내용 |
| --- | --- |
| GD | Gram 행렬·계수벡터 정의, interpolation 행렬, Lagrangian에서 LMI까지의 연결, Python 코드 대응, Figure 1의 경계조건 보강 |
| SGD | 일반 $n$의 계수벡터와 potential 행렬, variance 제약의 부호, Figure 2의 최종 거리항, 표기 충돌 및 비교표 수정 |
| Averaging | Theorem 6의 완전제곱 전개, 두 방법의 실제 LMI와 잔차 행렬, 가중평균과 코드의 산술평균 구분, 감소 step-size의 오차항 분석 및 최종 계수표 교정 |
| 공통 | 수식 구분자·표·제목 단계 정리, 그림 연결 복구 |

## 그림

- `assets/figure_1.png`, `assets/figure_2.png`: 제공된 논문 PDF의 Figure 1·2에서 가져온 그림이야.
- `assets/variance_coefficients.png`: Theorem 5·6·7의 정확한 오차 계수식을 사용해 $L=1$, $\delta_k=1$에서 그린 비교 그래프야.

압축을 푼 뒤 MD 파일과 `assets` 폴더를 같은 위치에 두면 그림이 연결돼. 수식을 지원하는 Markdown 뷰어에서 열면 본문의 수식도 함께 볼 수 있어.
