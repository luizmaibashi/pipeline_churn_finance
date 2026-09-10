export function roundHalfEven(value, decimals) {
  const factor = 10 ** decimals;
  const scaled = value * factor;
  const lower = Math.floor(scaled);
  const fraction = scaled - lower;
  const rounded = Math.abs(fraction - 0.5) < Number.EPSILON * Math.max(1, Math.abs(scaled))
    ? (lower % 2 === 0 ? lower : lower + 1)
    : Math.round(scaled);
  return rounded / factor;
}

export function predict(input, model) {
  const row = { ...input };
  for (const column of model.imputer_cols) {
    if (row[column] == null || Number.isNaN(row[column])) row[column] = model.imputer_medians[column];
  }
  const fe = model.fe;
  row.engajamento_score = roundHalfEven((row.freq_contato_mes / fe.freq_max) * (row.qtd_produtos / fe.qtd_max), 4);
  row.retorno_relativo = roundHalfEven(row.retorno_12m_pct - fe.media_retorno, 2);
  row.flag_risco = row.retorno_relativo < 0 && row.freq_contato_mes === 0 && row.qtd_produtos === 1 ? 1 : 0;
  row.intensidade_rel = roundHalfEven(Math.log1p(row.meses_cliente) * Math.log1p(row.freq_contato_mes), 4);
  const segmentIndex = model.encoder.categories[0].indexOf(row.segmento);
  if (segmentIndex < 0) throw new Error(`Segmento inválido: ${row.segmento}`);
  const x = model.feature_order.map((name) => name.startsWith("ordinals__") ? segmentIndex : Number(row[name.split("__", 2)[1]]));
  let raw = model.gb.init_raw;
  for (const tree of model.trees) {
    let node = 0;
    while (tree.cl[node] !== -1) node = Math.fround(x[tree.f[node]]) <= tree.th[node] ? tree.cl[node] : tree.cr[node];
    raw += model.gb.lr * tree.val[node];
  }
  return 1 / (1 + Math.exp(-raw));
}
