FROM python:3.10-slim

WORKDIR /app

# Instala dependências de compilação necessárias para pacotes Python, se houver
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir redis

# Copia todo o código fonte do projeto
COPY . .

# Expõe a porta interna da API do FastAPI
EXPOSE 8000

# Execução padrão (será substituída para o worker no docker-compose)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
