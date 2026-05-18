import os
import subprocess
import glob
import json
from agent import run_agent

# Configurações
MONITOR_DIR = os.path.join("output", "monitor")

def orchestrate_drift_and_summary():
    print("=== [Stanford Workflow] Executando Orquestrador Agêntico de Drift ===")
    
    # 1. Executar Monitor de Drift (Pipeline Clássico)
    print("\n1. Iniciando monitor.py...")
    try:
        subprocess.run(["python", "monitor.py", "--alert-only"], check=True)
        print("-> Drift não atingiu nível crítico para acionar retreino obrigatório (Exit 0).")
    except subprocess.CalledProcessError:
        print("-> [ALERTA] Monitor detectou drift crítico (Exit 1).")
    
    # 2. Ler o último relatório JSON
    list_of_files = glob.glob(f"{MONITOR_DIR}/*.json")
    if not list_of_files:
        print("Nenhum relatório de drift encontrado.")
        return
        
    latest_report_path = max(list_of_files, key=os.path.getctime)
    
    with open(latest_report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
        
    status = report.get('summary', {}).get('status', 'OK')
    
    if status in ['ATENCAO', 'CRITICO']:
        print(f"\n2. Drift detectado ({status}). Acionando IA Agêntica para Executive Summary...")
        
        # 3. Engenharia de Prompt para o Agente analisar o contexto
        prompt = (
            f"Fui acionado pelo monitor de MLOps. O status atual do modelo é {status}. "
            f"Temos {report['summary']['n_alertas']} alertas disparados nas distribuições de dados.\n"
            f"Por favor, verifique os alertas_drift na sua base e me gere um resumo executivo "
            f"explicando para a diretoria qual é o impacto comercial desse envelhecimento do modelo "
            f"e o que devemos fazer."
        )
        
        # O agente fará a "reflexão" baseada no problema
        resposta, tool_calls = run_agent(prompt)
        
        print("\n=== RESUMO EXECUTIVO DO AGENTE ===")
        print(resposta)
        print("==================================")
        
        # Salvar o resumo
        summary_path = latest_report_path.replace(".json", "_executive_summary.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Executive Summary - Agente Autônomo\n\n")
            f.write(resposta)
            
        print(f"\nResumo salvo em: {summary_path}")
        
    else:
        print(f"\n2. Status do modelo é {status}. Nenhuma ação de IA necessária. Economizando inferência.")

if __name__ == "__main__":
    orchestrate_drift_and_summary()
