# Thresholds v2

Seleção por custo `10 × FN + FP`, com recall mínimo de 0,75. Segmentos com menos de 5 positivos na validação usam o threshold global.

| segmento      |   threshold |   recall |   precision |   fn |   fp |   n_valid | origem_threshold   |
|:--------------|------------:|---------:|------------:|-----:|-----:|----------:|:-------------------|
| Alta Renda    |        0.1  | 0.5      |        0.25 |   10 |   30 |       136 | segmento           |
| Family Office |        0.1  | 0        |        0    |    1 |    0 |        13 | global_fallback    |
| Private       |        0.07 | 0.818182 |        0.25 |    2 |   27 |       111 | segmento           |
| Wealth        |        0.1  | 0.666667 |        0.4  |    1 |    3 |        40 | global_fallback    |
