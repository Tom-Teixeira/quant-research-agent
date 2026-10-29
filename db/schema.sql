-- 量化数据仓库 schema（PostgreSQL + pgvector）
-- 目标：10万+ 公告/新闻 超大规模存储 + 向量检索
-- 初始化脚本由 docker-entrypoint-initdb.d 自动执行

CREATE EXTENSION IF NOT EXISTS vector;

-- ============ 1. 公告/新闻 元数据表（结构化存储） ============
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    doc_type      TEXT NOT NULL,                -- announcement / news / research_report
    ts_code       TEXT,                         -- 股票代码（如 600000.SH）
    name          TEXT,                         -- 公司/来源名称
    title         TEXT NOT NULL,                -- 标题
    content       TEXT,                         -- 正文（PDF 解析或新闻正文）
    ann_date      DATE NOT NULL,                -- 发布日期
    source_url    TEXT,                         -- PDF/原文链接
    extra         JSONB DEFAULT '{}',           -- 扩展字段（事件类型、情感词典分等）
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_date ON documents (ann_date);
CREATE INDEX IF NOT EXISTS idx_documents_code ON documents (ts_code);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (doc_type);
-- 幂等入库去重键（doc_type + ts_code + title + ann_date）
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_key
    ON documents (doc_type, ts_code, title, ann_date);

-- ============ 2. 向量表（pgvector，RAG 检索） ============
-- embedding 维度：bge-small-zh-v1.5 = 512 维（中文金融场景）
CREATE TABLE IF NOT EXISTS doc_embeddings (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,                 -- 文本分块序号
    chunk_text    TEXT NOT NULL,                -- 分块原文
    embedding     vector(512) NOT NULL,         -- 向量（bge-small-zh 512 维）
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- HNSW 索引：10万+ 规模推荐（百万级内查询毫秒级）
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON doc_embeddings USING hnsw (embedding vector_cosine_ops);

-- ============ 3. 因子面板表（复用一期因子结果落库） ============
CREATE TABLE IF NOT EXISTS factor_panels (
    date        DATE NOT NULL,
    code        TEXT NOT NULL,
    sentiment_mean FLOAT, sentiment_std FLOAT,
    event_mean  FLOAT, event_max FLOAT,
    n_announcements INT, tone_vol FLOAT,
    ret_next    FLOAT,
    PRIMARY KEY (date, code)
);

-- ============ 4. 检索辅助视图：文档+向量 联查 ============
CREATE OR REPLACE VIEW v_docs AS
SELECT d.id AS doc_id, d.doc_type, d.ts_code, d.name, d.title,
       d.ann_date, d.source_url, e.chunk_index, e.chunk_text, e.embedding
FROM documents d
JOIN doc_embeddings e ON e.doc_id = d.id;
