const explanations = await fetch("./data/client_explanations.json").then((response) => response.json());
const example = explanations.sort((a, b) => b.churn_prob - a.churn_prob)[0];
const panel = document.querySelector("#predict");
const section = document.createElement("article");
section.className = "card";
section.innerHTML = `<h2>Explicabilidade: caso do snapshot</h2>
  <p class="metric">Exemplo pré-computado para ${example.cliente_id} (${example.segmento}), com probabilidade de churn de ${(example.churn_prob * 100).toFixed(1)}%.</p>
  <p>${example.explicacao.replaceAll("\n", "<br>")}</p>
  <p class="metric">Este texto explica uma relação do snapshot. O perfil livre acima é pontuado no navegador e não recebe SHAP aproximado.</p>`;
panel.append(section);
