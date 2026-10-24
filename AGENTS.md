# Presentation Markdown

The Markdown files in `docs/` are standalone sources for presentation slides/PPTX and will be exported separately from the notebooks.

- Use the paper's mathematical notation throughout `docs/` whenever a corresponding notation exists, including iterates, sampled oracle indices, gradients, potential coefficients, selectors, and SDP expressions.
- Retain implementation names such as `relax` and `tau` when explaining experiment settings or code, and state their correspondence to the mathematical notation when needed.
- For quantities not explicitly indexed in the paper, use the existing algorithm variables directly and define their selectors locally. Use auxiliary numeric point indices only where the paper defines them.
- Keep experiment settings, optimization procedures, and explanations of tables self-contained in the Markdown.
- Do not add notebook links, notebook attachment references, or directions that require access to `ipynb/` to understand the presentation.
- Do not add bibliography/source sections or external paper/implementation reference links to the presentation Markdown unless requested.
- In Korean presentation prose, describe solving optimization problems consistently as `MOSEK으로 최적화한다`, using the appropriate conjugation.
- Name sequential optimization problems explicitly as `첫 번째 SDP` and `두 번째 SDP` in prose, headings, and tables, rather than using bare first/second or optimization-stage labels.
- Use initial/intermediate/last wording naturally with the relevant indexed quantity, such as `초기 potential $\phi_0^f$`, `마지막 점 $x_N$`, and `마지막 오차 계수 $e_{N-1}$`. Do not append redundant `(k=0)` or `(k=N)` labels. State necessary index ranges separately: potential coefficients run over `0<=k<=N`, intermediate coefficients over `0<k<N`, and error coefficients `e_k` over `0<=k<N`, ending at `e_{N-1}`. In comparison tables, identify the actual compared term, such as `a_N` or `e_{N-1}`.
- At use sites, briefly identify the source experiment and selected SDP. Refer directly to its results; do not repeat the optimization procedure, regroup experiments, or create a result inventory. Do not imply that one fixed-parameter solution supplies results for other parameter values.
- When describing growth or decay, explicitly name the varying parameter, such as `k`, `N`, or `alpha`.
- Present shorthand notation as a self-contained helper box with a bold title, using a Markdown blockquote rather than a section heading. Include the selector definitions and update identities, then resume the main text before the potential coefficient matrices `V_k` are defined.
