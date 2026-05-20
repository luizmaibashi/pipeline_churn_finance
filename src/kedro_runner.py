import os
import yaml
import pandas as pd
import joblib
import json

class DataCatalog:
    """
    Abstração simplificada do Data Catalog do Kedro.
    Gerencia o carregamento e salvamento de datasets sem acoplamento de I/O nos nodes.
    """
    def __init__(self, catalog_path="conf/base/catalog.yml"):
        self.catalog_path = catalog_path
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Arquivo de catálogo não encontrado: {catalog_path}")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def load(self, name):
        if name not in self.config:
            raise ValueError(f"Dataset '{name}' não está cadastrado no catalog.yml")
        
        info = self.config[name]
        ds_type = info.get("type")
        filepath = info.get("filepath")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo do dataset '{name}' não encontrado no caminho: {filepath}")

        if ds_type == "pandas.CSVDataSet":
            return pd.read_csv(filepath)
        elif ds_type == "json.JSONDataSet":
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        elif ds_type == "pickle.PickleDataSet":
            return joblib.load(filepath)
        else:
            raise ValueError(f"Tipo de dataset não suportado: {ds_type}")

    def save(self, name, data):
        if name not in self.config:
            raise ValueError(f"Dataset '{name}' não está cadastrado no catalog.yml")
        
        info = self.config[name]
        ds_type = info.get("type")
        filepath = info.get("filepath")

        # Cria diretórios pai se não existirem
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if ds_type == "pandas.CSVDataSet":
            data.to_csv(filepath, index=False)
        elif ds_type == "json.JSONDataSet":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        elif ds_type == "pickle.PickleDataSet":
            joblib.dump(data, filepath)
        else:
            raise ValueError(f"Tipo de dataset não suportado para escrita: {ds_type}")
        
        print(f"  [Catalog] Dataset salvo com sucesso em: {filepath}")


def load_parameters(parameters_path="conf/base/parameters.yml"):
    """Carrega parâmetros globais e hiperparâmetros de configuração."""
    if not os.path.exists(parameters_path):
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {parameters_path}")
    with open(parameters_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
