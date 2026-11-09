# Sentinel AI — Guardrailed Document Intelligence & RAG Pipeline

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://sentinelai44.vercel.app/)
[![API Status](https://img.shields.io/badge/Backend_API-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://sentinelai-5n2h.onrender.com/)
[![Database](https://img.shields.io/badge/Vector_DB-Neon_pgvector-00E599?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech/)
[![Model](https://img.shields.io/badge/NVIDIA_NIM-Nemotron--3.5-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com/)

Sentinel AI is a production-grade, secure Document Intelligence QA system powered by NVIDIA NIM microservices, PostgreSQL `pgvector`, NeMo Guardrails, and RAGAS automated benchmarking.

---

## 🌐 Live Production Links

| Component | Platform | URL |
| :--- | :--- | :--- |
| **Frontend Web App** | **Vercel** | [https://sentinelai44.vercel.app/](https://sentinelai44.vercel.app/) |
| **Backend REST API** | **Render** | [https://sentinelai-5n2h.onrender.com/](https://sentinelai-5n2h.onrender.com/) |
| **API Interactive Docs** | **Swagger / OpenAPI** | [https://sentinelai-5n2h.onrender.com/docs](https://sentinelai-5n2h.onrender.com/docs) |
| **Cloud Vector Store** | **Neon PostgreSQL** | Serverless `pgvector` (2048-dim embeddings) |

---

## 🌟 Key Capabilities
- **Document Ingestion & Multi-Format Parsing**: Upload PDF, DOCX, or TXT documents. Documents are chunked and embedded into PostgreSQL pgvector using `nvidia/nemotron-3.5-embed-1b`.
- **Dynamic Question Generation**: Automatically analyzes document excerpts to generate tailored questions in the top ribbon menu for one-click querying.
- **ChatGPT-Style Conversational QA**: Instant question rendering, animated thinking/searching state with pulsing Sentinel AI avatar, verifiable citations, reasoning traces, and excerpt inspection.
- **Session Privacy & Zero Retention**: Dedicated "Clear Session" and "Wipe PDF & Data" actions purge vector embeddings, uploaded files, and chat history so no data is exposed to subsequent users.
- **Built-in RAGAS Evaluation**: Persistent benchmark scorecard evaluating Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.12+
- PostgreSQL 15+ with the `pgvector` extension enabled
- NVIDIA NIM API Key ([build.nvidia.com](https://build.nvidia.com/))

### 2. Setup Environment
```powershell
# Clone the repository
git clone <your-repo-url>
cd Sentinel_AI

# Create and activate virtual environment
python -m venv RAG
.\RAG\Scripts\activate  # On Linux/macOS: source RAG/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure `.env`
Create a `.env` file in the root directory:
```ini
NVIDIA_API_KEY=nvapi-your-key-here
DB_PASSWORD=your_postgres_password
DB_USER=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentinelrag
```

### 4. Run the Server
```powershell
python backend/main.py
```
Open **http://127.0.0.1:8000** in your browser.

---

## 🐳 Deployment Options

### Method 1: Docker Compose (Recommended for Local / VPS)
The included `docker-compose.yml` launches both the PostgreSQL pgvector database and the Sentinel AI application with a single command.

1. Ensure your `.env` contains your `NVIDIA_API_KEY` and preferred `DB_PASSWORD`.
2. Run:
```bash
docker compose up -d --build
```
3. Access the web interface at **http://localhost:8000**.
4. To stop:
```bash
docker compose down
```

---

### Method 2: Cloud PaaS (Render / Railway / Fly.io)

You can deploy Sentinel AI to managed cloud platforms with zero server management:

#### 1. Setup a Managed pgvector Database
Use **Neon**, **Supabase**, or **Railway PostgreSQL**:
- Ensure the `vector` extension is active:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- Copy the provided connection string (e.g., `postgresql://user:password@ep-xyz.neon.tech/sentinelrag?sslmode=require`).

#### 2. Deploy Web Service (e.g., Render / Railway)
- Connect your GitHub repository.
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `NVIDIA_API_KEY`: Your NVIDIA NIM key (`nvapi-...`).
  - `DATABASE_URL`: Your cloud database connection URL.
  - `PYTHON_VERSION`: `3.12.0`

---

### Method 3: Linux VPS (Ubuntu / Debian + Nginx + Systemd)

#### 1. Install PostgreSQL with pgvector
```bash
sudo apt update && sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all
# Install pgvector
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```
Enable the extension in your database:
```sql
CREATE DATABASE sentinelrag;
\c sentinelrag
CREATE EXTENSION vector;
```

#### 2. Configure Systemd Service (`/etc/systemd/system/sentinel.service`)
```ini
[Unit]
Description=Sentinel AI FastAPI Service
After=network.target postgresql.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Sentinel_AI
Environment="PATH=/home/ubuntu/Sentinel_AI/venv/bin"
EnvironmentFile=/home/ubuntu/Sentinel_AI/.env
ExecStart=/home/ubuntu/Sentinel_AI/venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```
Start and enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sentinel
```

#### 3. Configure Nginx Reverse Proxy (`/etc/nginx/sites-available/sentinel`)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable and reload:
```bash
sudo ln -s /etc/nginx/sites-available/sentinel /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```
*(Optional: Add SSL certificate using `sudo certbot --nginx -d your-domain.com`)*
