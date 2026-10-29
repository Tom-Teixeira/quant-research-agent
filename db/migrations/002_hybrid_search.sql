-- 002 迁移：混合检索（HNSW 稠密 + tsvector 稀疏 → RRF 融合）
-- 适用：既有容器（quant-pgvector）已初始化过 schema.sql 的场景
-- 中文 BM25 稀疏检索依赖 jieba 分词（应用层入库时写入 content_seg，
-- 分词文本经 to_tsvector('simple', ...) 建索引，'simple' 配置按空白切词）
-- 执行：psql -h localhost -U quant -d quantdb -f db/migrations/002_hybrid_search.sql
-- 或：python scripts/backfill_tsv.py（推荐：自动完成 ALTER + 回填 + 索引）

BEGIN;

-- 1. 分词文本列（jieba 空格分词，供 tsvector 稀疏检索）
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_seg TEXT;

-- 2. tsvector 列 + GIN 索引（title 权重 A / 正文权重 B）
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_tsv tsvector;
CREATE INDEX IF NOT EXISTS idx_documents_content_tsv
    ON documents USING GIN (content_tsv);

-- 3. 增量维护触发器（新写入/更新文档自动重算 content_tsv）
CREATE OR REPLACE FUNCTION documents_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv :=
        setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(NEW.content_seg, '')), 'B');
    RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_tsv ON documents;
CREATE TRIGGER trg_documents_tsv
    BEFORE INSERT OR UPDATE OF title, content_seg ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_tsv_trigger();

COMMIT;
