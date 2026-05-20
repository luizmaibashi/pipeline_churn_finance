import os
import sys
import time
import json
import datetime

# Garante importações corretas a partir da raiz
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.job_queue import JobQueue
import api
from api import ClienteInput, _load_model, _load_shap_explanations, _predict_one

def main():
    print("=" * 60)
    print("CHURN FINANCE — BACKGROUND INFERENCE WORKER")
    print("=" * 60)
    
    # Inicializa o estado global da API no contexto do worker
    print("[Worker] Carregando modelo e explicadores SHAP...")
    try:
        model, version, meta = _load_model()
        api._state["model"] = model
        api._state["version"] = version
        api._state["meta"] = meta
        api._state["shap_df"] = _load_shap_explanations()
        print(f"[Worker] Modelo v{version} carregado com sucesso.")
    except Exception as e:
        print(f"[FATAL] Não foi possível carregar o modelo: {e}")
        sys.exit(1)
        
    queue = JobQueue()
    use_redis = queue.redis_client is not None
    
    if use_redis:
        print("[Worker] Modo: Conectado ao Redis. Aguardando mensagens na fila 'churn_jobs_queue'...")
    else:
        print("[Worker] Modo: SQLite local. Monitorando a tabela 'jobs' a cada 1 segundo...")
        
    try:
        while True:
            job_id = None
            if use_redis:
                # BLPOP bloqueia até ter um job na fila (timeout 2 segundos)
                pop_res = queue.redis_client.blpop("churn_jobs_queue", timeout=2)
                if pop_res:
                    _, job_id_bytes = pop_res
                    job_id = job_id_bytes.decode('utf-8')
            else:
                # Consulta SQLite por jobs pendentes
                import sqlite3
                conn = sqlite3.connect(queue.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM jobs WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT 1")
                row = cursor.fetchone()
                conn.close()
                if row:
                    job_id = row[0]
                else:
                    time.sleep(1) # Dorme 1 segundo antes de tentar novamente
                    
            if job_id:
                print(f"[Worker] Processando Job {job_id}...")
                
                # Atualiza status para PROCESSING
                queue.update_job(job_id, "PROCESSING")
                
                try:
                    # Busca payload do job
                    job_data = queue.get_status(job_id)
                    if not job_data:
                        raise ValueError(f"Dados do Job {job_id} não encontrados no banco/cache.")
                        
                    # Recarrega o payload original que contém os clientes
                    if use_redis:
                        # Em Redis o payload é pego na consulta inicial do hash
                        job_hash = queue.redis_client.hgetall(f"job:{job_id}")
                        payload_str = job_hash[b"payload"].decode('utf-8')
                    else:
                        conn = sqlite3.connect(queue.db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,))
                        payload_str = cursor.fetchone()[0]
                        conn.close()
                        
                    payload_dict = json.loads(payload_str)
                    clientes_list = payload_dict.get("clientes", [])
                    
                    results = []
                    for c_dict in clientes_list:
                        # Converte dicionário para modelo ClienteInput
                        cliente_obj = ClienteInput(**c_dict)
                        # Roda predição individual
                        pred_res = _predict_one(cliente_obj)
                        results.append(pred_res.dict())
                        
                    # Gera os agregados do lote
                    alto   = [r for r in results if r["risk_level"] == "ALTO"]
                    medio  = [r for r in results if r["risk_level"] == "MEDIO"]
                    baixo  = [r for r in results if r["risk_level"] == "BAIXO"]
                    humano = [r for r in results if "REVISAO" in r["flow"]]
                    
                    total_auc_risk = round(sum(r["auc_at_risk_MM"] for r in alto), 2)
                    
                    summary = {
                        "total_clientes"         : len(results),
                        "alto_risco"             : len(alto),
                        "medio_risco"            : len(medio),
                        "baixo_risco"            : len(baixo),
                        "revisao_humana"         : len(humano),
                        "auc_total_em_risco_MM"  : total_auc_risk,
                        "pct_alto_risco"         : f"{len(alto)/len(results)*100:.1f}%",
                    }
                    
                    batch_result = {
                        "total": len(results),
                        "scored_at": datetime.datetime.now().isoformat(),
                        "model_version": version,
                        "results": results,
                        "summary": summary
                    }
                    
                    # Atualiza o Job para COMPLETED com os resultados
                    queue.update_job(job_id, "COMPLETED", result=batch_result)
                    print(f"[Worker] Job {job_id} concluído com sucesso (Total: {len(results)} clientes).")
                    
                except Exception as ex:
                    print(f"[Worker] Erro ao processar Job {job_id}: {ex}")
                    queue.update_job(job_id, "FAILED", error=str(ex))
                    
    except KeyboardInterrupt:
        print("\n[Worker] Finalizando worker por interrupção do usuário.")

if __name__ == "__main__":
    main()
