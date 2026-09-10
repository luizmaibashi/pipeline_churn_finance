---
tipo: grilling
status: resolvido
criado: 2026-09-10
resolvido: 2026-09-10
---

# Ticket 0003: Escopo da página — abas, profundidade, identidade visual

## Bloqueio

O `app.py` tem 4 abas com necessidades de runtime diferentes. Precisa de decisão
do Luiz sobre o que vai pra página e como ela se parece.

| Aba | Precisa do modelo vivo? | Dado a servir |
|---|---|---|
| 🎯 Predição Individual | **Sim** — `predict_proba` + gauge contínuo + `risk_factors()` | nenhum (input do usuário) |
| 📈 Análise da Carteira | Não — snapshot | `output/data/base_clientes_v2.csv`, `thresholds_v2.csv` |
| 🔬 Performance do Modelo | Não — snapshot | `comparacao_v1_v2.csv`, `feature_importance_v2.csv`, `confusion_matrix.csv` |
| 🧭 Carteira Exposta por Assessor | Não — snapshot | `carteira_exposta_por_assessor.csv` |
| (SHAP) | Não — snapshot | `output/shap/v2/client_explanations.csv` |

## Perguntas para o Luiz

1. **As 4 abas, ou um subconjunto curado?** A Predição Individual é a única que
   exige o port do modelo (custo de 0001/0002). As outras 3 são "só" HTML+dataviz
   sobre CSV. Vale o port pela aba 1, ou a página vira snapshot puro (Balde 1) e
   a predição interativa fica de fora?
2. **Identidade visual.** O `app.py` atual é dark (`#0e1117`, texto `#e8ecf4`,
   acentos neon). Memória da base: "não repetir a paleta teal-green por preguiça,
   rodar `ui-ux-pro-max --design-system` calibrado por domínio". Wealth /
   early-warning é o domínio. Recalibrar do zero ou manter o dark atual?
3. **Gauge contínuo.** Se a Opção C (grade pré-computada) entrar, o gauge perde
   resolução. Isso é aceitável, ou o gauge contínuo é requisito da aba 1?
4. **SHAP.** A explicação por cliente entra como aba própria, tooltip na predição,
   ou fica fora?

## Resultado (Luiz, 2026-09-10)

1. **As 4 abas entram.** Predição Individual (modelo vivo), Análise da Carteira,
   Performance do Modelo, Carteira Exposta por Assessor.
2. **Tab 1 com sliders livres + gauge contínuo** — híbrido, port JS (0005).
3. **Identidade visual: recalibrar por domínio.** Rodar
   `ui-ux-pro-max --design-system` para wealth / early-warning comportamental.
   Descartar o dark `#0e1117` + neon do `app.py` (tell de IA). Não cair na paleta
   teal-green (`#166e5a`) dos outros projetos por default — calibrar do zero para
   este domínio. Sequência anti-cara-de-IA do playbook:
   `artifact-design → dataviz → ui-ux-pro-max → escrita-organica`.
4. **SHAP:** `client_explanations.csv` (40 KB) entra como camada da Tab 1 —
   fatores de contribuição por cliente ao lado do gauge. `shap_values.csv`
   (372 KB) **não** vai pro bundle.
5. Gauge contínuo é requisito da Tab 1 (fecha a porta pra Opção C).
