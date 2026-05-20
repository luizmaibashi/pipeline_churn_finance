import os
import uuid
import json
import sqlite3
import datetime

try:
    import redis
except ImportError:
    redis = None

REDIS_URL = os.environ.get("REDIS_URL", None)

class JobQueue:
    """
    Fila de execução assíncrona com suporte dual:
    Usa Redis se configurado em ambiente de produção;
    Recai para SQLite para execuções locais sem necessidade de infraestrutura adicional.
    """
    def __init__(self, db_path="output/jobs.db"):
        self.db_path = db_path
        self.redis_client = None
        
        if REDIS_URL and redis is not None:
            try:
                # Remove prefixo se necessário ou usa direto
                self.redis_client = redis.from_url(REDIS_URL)
                # Testa conexão
                self.redis_client.ping()
                print(f"[JobQueue] Conectado ao Redis em: {REDIS_URL}")
            except Exception as e:
                print(f"[JobQueue] Falha de ping no Redis ({e}). Usando SQLite fallback.")
                self.redis_client = None
        else:
            if REDIS_URL and redis is None:
                print("[JobQueue] REDIS_URL está configurada, mas o pacote 'redis' não está instalado. Usando SQLite fallback.")
                
        if not self.redis_client:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._init_sqlite()

    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                payload TEXT,
                result TEXT,
                error TEXT,
                created_at TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def enqueue(self, payload_dict) -> str:
        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        
        if self.redis_client:
            # Envia via Redis
            self.redis_client.hset(f"job:{job_id}", mapping={
                "status": "PENDING",
                "payload": json.dumps(payload_dict),
                "created_at": created_at,
                "result": "",
                "error": ""
            })
            self.redis_client.rpush("churn_jobs_queue", job_id)
        else:
            # Envia via SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO jobs (id, status, payload, created_at) VALUES (?, ?, ?, ?)",
                (job_id, "PENDING", json.dumps(payload_dict), created_at)
            )
            conn.commit()
            conn.close()
            
        return job_id

    def get_status(self, job_id) -> dict:
        if self.redis_client:
            job_data = self.redis_client.hgetall(f"job:{job_id}")
            if not job_data:
                return None
            
            # Converte bytes do Redis para string
            data = {k.decode('utf-8'): v.decode('utf-8') for k, v in job_data.items()}
            return {
                "id": job_id,
                "status": data.get("status"),
                "result": json.loads(data.get("result")) if data.get("result") else None,
                "error": data.get("error") if data.get("error") else None,
                "created_at": data.get("created_at"),
                "completed_at": data.get("completed_at") if data.get("completed_at") != "" else None
            }
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT status, result, error, created_at, completed_at FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            return {
                "id": job_id,
                "status": row[0],
                "result": json.loads(row[1]) if row[1] else None,
                "error": row[2] if row[2] else None,
                "created_at": row[3],
                "completed_at": row[4]
            }

    def update_job(self, job_id, status, result=None, error=None):
        completed_at = datetime.datetime.now().isoformat() if status in ["COMPLETED", "FAILED"] else ""
        
        if self.redis_client:
            mapping = {"status": status}
            if result:
                mapping["result"] = json.dumps(result)
            if error:
                mapping["error"] = error
            if completed_at:
                mapping["completed_at"] = completed_at
            self.redis_client.hset(f"job:{job_id}", mapping=mapping)
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if status in ["COMPLETED", "FAILED"]:
                cursor.execute(
                    "UPDATE jobs SET status = ?, result = ?, error = ?, completed_at = ? WHERE id = ?",
                    (status, json.dumps(result) if result else None, error, completed_at if completed_at != "" else None, job_id)
                )
            else:
                cursor.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()
            conn.close()
