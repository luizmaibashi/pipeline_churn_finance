import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import FeatureEngineer, StructuralNullImputer

FEATURES_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
]

FEATURES_AFTER_FE = [
    "meses_cliente", "qtd_produtos", "retorno_12m_pct",
    "freq_contato_mes", "saldo_bi", "engajamento_score",
    "retorno_relativo", "flag_risco", "intensidade_rel"
]

# Direção A+B (ADR-0001): early-warning comportamental + risco de assessor.
# Colunas com nulo estrutural (retorno_12m_pct, dias_desde_ultimo_contato,
# tempo_resposta_medio_horas) exigem SimpleImputer no pipeline — fit só no
# treino, igual ao FeatureEngineer, para não vazar (ver _build_preprocessing_v2).
FEATURES_V2_NUMERICAS_COM_NULO = [
    "retorno_12m_pct", "dias_desde_ultimo_contato", "tempo_resposta_medio_horas"
]
FEATURES_V2_EXTRA = [
    "dias_desde_ultimo_contato", "variacao_freq_contato_3m",
    "tempo_resposta_medio_horas", "auc_exposto",
    "sem_historico_12m", "cliente_novo_sem_contato_hist"
]
FEATURES_V2_BASE = [
    "segmento", "meses_cliente", "qtd_produtos",
    "retorno_12m_pct", "freq_contato_mes", "saldo_bi"
] + FEATURES_V2_EXTRA

def _build_preprocessing():
    """Constrói o ColumnTransformer com o OrdinalEncoder para o pipeline."""
    encoder = OrdinalEncoder(categories=[["Varejo", "Alta Renda", "Wealth", "Corporate"]])
    return ColumnTransformer(
        transformers=[
            ("ordinals", encoder, ["segmento"]),
            ("pass", "passthrough", FEATURES_AFTER_FE)
        ]
    )

def _create_pipeline(classifier, scale=False):
    """Cria um scikit-learn Pipeline com o Custom Transformer e pré-processamento."""
    steps = [
        ("fe", FeatureEngineer()),
        ("prep", _build_preprocessing())
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def benchmark_algorithms(train_df: pd.DataFrame, test_df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Executa o benchmark de múltiplos classificadores nos dados estratificados."""
    X_train, y_train = train_df[FEATURES_BASE], train_df["churn"]
    X_test, y_test = test_df[FEATURES_BASE], test_df["churn"]
    random_state = parameters.get("random_state", 42)

    modelos_bench = {
        "Dummy (baseline)"   : _create_pipeline(DummyClassifier(strategy="stratified", random_state=random_state)),
        "Logistic Regression": _create_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state), scale=True),
        "Decision Tree"      : _create_pipeline(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=random_state)),
        "Random Forest"      : _create_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state)),
        "Gradient Boosting"  : _create_pipeline(GradientBoostingClassifier(
                                    n_estimators=parameters.get("n_estimators", 300),
                                    learning_rate=parameters.get("learning_rate", 0.03),
                                    max_depth=parameters.get("max_depth", 4),
                                    random_state=random_state
                                ))
    }

    benchmark_results = []
    for nome, pipeline in modelos_bench.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        f1_mac  = f1_score(y_test, y_pred, average="macro")
        f1_chur = f1_score(y_test, y_pred, pos_label=1, average="binary")
        roc     = roc_auc_score(y_test, y_prob)
        acc     = (y_pred == y_test).mean()

        benchmark_results.append({
            "modelo": nome, 
            "acuracia": acc,
            "f1_macro": f1_mac, 
            "f1_churn": f1_chur, 
            "roc_auc": roc
        })
        
    return pd.DataFrame(benchmark_results)


def train_final_model(train_df: pd.DataFrame, parameters: dict) -> Pipeline:
    """Treina o classificador final de Gradient Boosting no conjunto de treinamento."""
    X_train, y_train = train_df[FEATURES_BASE], train_df["churn"]
    
    gb_final = _create_pipeline(GradientBoostingClassifier(
        n_estimators=parameters.get("n_estimators", 300),
        learning_rate=parameters.get("learning_rate", 0.03),
        max_depth=parameters.get("max_depth", 4),
        random_state=parameters.get("random_state", 42)
    ))
    gb_final.fit(X_train, y_train)
    return gb_final


def evaluate_final_model(model: Pipeline, test_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Avalia o pipeline final treinado.
    Retorna métricas consolidadas e matriz de confusão.
    """
    X_test, y_test = test_df[FEATURES_BASE], test_df["churn"]
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_chur = f1_score(y_test, y_pred, pos_label=1, average="binary")
    roc     = roc_auc_score(y_test, y_prob)
    acc     = (y_pred == y_test).mean()

    metrics = {
        "f1_macro": f1_mac,
        "f1_churn": f1_chur,
        "roc_auc": roc,
        "acuracia": acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    df_cm = pd.DataFrame({
        "tn": [tn], "fp": [fp], "fn": [fn], "tp": [tp]
    })
    
    return metrics, df_cm


def cross_validate(model: Pipeline, df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Executa a validação cruzada estratificada de 5 folds."""
    X, y = df[FEATURES_BASE], df["churn"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=parameters.get("random_state", 42))
    
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    
    df_cv = pd.DataFrame({
        "fold"    : [f"Fold {i+1}" for i in range(5)],
        "f1_macro": cv_scores.round(4)
    })
    return df_cv, float(cv_scores.mean()), float(cv_scores.std())


def _build_preprocessing_v2():
    """ColumnTransformer v2: OrdinalEncoder no segmento, passthrough no resto
    (FEATURES_AFTER_FE_V2, gerado após StructuralNullImputer + FeatureEngineer)."""
    encoder = OrdinalEncoder(categories=[["Varejo", "Alta Renda", "Wealth", "Corporate"]])
    features_pass = [
        "meses_cliente", "qtd_produtos", "retorno_12m_pct",
        "freq_contato_mes", "saldo_bi", "engajamento_score",
        "retorno_relativo", "flag_risco", "intensidade_rel",
    ] + FEATURES_V2_EXTRA
    return ColumnTransformer(transformers=[
        ("ordinals", encoder, ["segmento"]),
        ("pass", "passthrough", features_pass),
    ])


def _create_pipeline_v2(classifier, scale=False):
    steps = [
        ("null_imputer", StructuralNullImputer(FEATURES_V2_NUMERICAS_COM_NULO)),
        ("fe", FeatureEngineer()),
        ("prep", _build_preprocessing_v2()),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def train_and_compare_v1_v2(
    df_v1: pd.DataFrame, df_v2_clean: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Testa o critério de sucesso do ADR-0001 §5: modelo v2 (early-warning +
    AuC exposto) supera a baseline v1 (reativa, só saldo/segmento) em
    recall/ROC-AUC, medido sobre split estratificado idêntico (mesma
    população, mesmo grão — GATE ML "comparação modelo-vs-baseline").

    IMPORTANTE: split feito UMA VEZ sobre o df_v2_clean (que tem os mesmos
    cliente_id do df_v1, só com colunas extras) e aplicado aos dois
    conjuntos de features, para não comparar populações diferentes.
    """
    random_state = parameters.get("random_state", 42)
    test_size = parameters.get("test_size", 0.20)

    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        df_v2_clean.index, test_size=test_size, random_state=random_state,
        stratify=df_v2_clean["churn"]
    )

    df_v1_indexed = df_v1.set_index("cliente_id")
    df_v2_indexed = df_v2_clean.set_index("cliente_id")
    ids_train = df_v2_clean.loc[train_idx, "cliente_id"]
    ids_test = df_v2_clean.loc[test_idx, "cliente_id"]

    v1_train = df_v1_indexed.loc[ids_train].reset_index()
    v1_test = df_v1_indexed.loc[ids_test].reset_index()
    v2_train = df_v2_indexed.loc[ids_train].reset_index()
    v2_test = df_v2_indexed.loc[ids_test].reset_index()

    resultados = []
    for nome, X_cols, train_df, test_df, pipeline_fn in [
        ("v1_baseline_reativa", FEATURES_BASE, v1_train, v1_test, _create_pipeline),
        ("v2_early_warning_advisor", FEATURES_V2_BASE, v2_train, v2_test, _create_pipeline_v2),
    ]:
        pipe = pipeline_fn(GradientBoostingClassifier(
            n_estimators=parameters.get("n_estimators", 300),
            learning_rate=parameters.get("learning_rate", 0.03),
            max_depth=parameters.get("max_depth", 4),
            random_state=random_state
        ))
        pipe.fit(train_df[X_cols], train_df["churn"])
        y_pred = pipe.predict(test_df[X_cols])
        y_prob = pipe.predict_proba(test_df[X_cols])[:, 1]

        resultados.append({
            "modelo": nome,
            "recall_churn": recall_score(test_df["churn"], y_pred, pos_label=1),
            "f1_churn": f1_score(test_df["churn"], y_pred, pos_label=1),
            "roc_auc": roc_auc_score(test_df["churn"], y_prob),
            "n_teste": len(test_df),
        })

    df_resultado = pd.DataFrame(resultados)
    return df_resultado


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Extrai e ordena as importâncias das features calculadas pelo Gradient Boosting."""
    gb_clf = model.named_steps["clf"]
    importances = pd.DataFrame({
        "feature"   : ["segmento_enc"] + FEATURES_AFTER_FE,
        "importance": gb_clf.feature_importances_
    }).sort_values("importance", ascending=False)
    return importances
