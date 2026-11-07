# quant-research-agent

A LangGraph-based agent that pulls Chinese market announcements and news, builds text factors, and runs a cross-sectional signal through a backtest. A RAG layer lets the agent retrieve relevant announcements when writing a research summary.

## Pipeline

The graph is a state machine with a retry loop around the factor-building step. If the rank IC comes back below a threshold, a DeepSeek model diagnoses the failure and the graph rebuilds factors up to a fixed retry count before moving on.

Stages:
1. Data ingestion. Choice (EMQuantAPI) is the primary source for CSI 300 constituents, announcements, company news, and daily prices. A file-based provider with sample data is used when Choice isn't installed, so the graph still runs without any credentials.
2. Factor construction. Text is cleaned, deduplicated, and converted into sentiment and event-intensity factors.
3. Signal. A cross-sectional model produces daily rank IC, ICIR, and AUC.
4. Backtest. Grouped returns and a neutralized excess return with transaction costs.
5. RAG retrieval and report. Relevant announcements are pulled from Postgres and passed to DeepSeek with the metrics.

## Storage and retrieval

Postgres with pgvector stores document metadata and 512-dim embeddings from a BAAI/bge-small-zh model (fastembed ONNX, no PyTorch needed). The embedding index uses HNSW with cosine distance.

Retrieval runs two paths: a dense vector search through pgvector and a sparse keyword search through a Postgres tsvector with jieba tokenization. Results are merged with reciprocal rank fusion (k=60). A simple query classifier routes direct questions straight to the LLM, single-entity questions through one retrieval pass, and multi-entity or comparison questions through query expansion and a larger candidate pool.

## Running it

```
git clone https://github.com/Tom-Teixeira/quant-research-agent.git
cd quant-research-agent
pip install -r requirements.txt
cp .env.example .env
docker compose -f docker-compose.db.yml up -d
python main_v2.py --stage check
python main_v2.py --stage ingest --start 2024-01-01 --end 2024-01-31
python main_v2.py --stage research --start 2024-01-01 --end 2024-01-10
```

The default mode is mock, which uses sample data and no external API calls. Set `QUANT_AGENT_MODE=live` and fill in DeepSeek and Choice credentials to run against real data.

## Layout

- `graph_v2.py` and `state.py` define the LangGraph topology and shared state
- `ingest/` has the Choice/file providers and the chunking and cleaning pipeline
- `db/` has the schema, migrations, and the pgvector client
- `rag/` has the embedder, jieba segmenter, and adaptive retrieval logic
- `tools/` has the LangChain tools used by graph nodes
- `scripts/` has verification scripts and sample data generation

## Contact

21735749a@gmail.com
