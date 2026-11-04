-- 저장 계층: SQLite. 모든 테이블 생성은 여기서만 한다 (IF NOT EXISTS -> 재실행 안전).
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    company_id INTEGER PRIMARY KEY,
    corp_code  TEXT NOT NULL UNIQUE,
    name       TEXT NOT NULL UNIQUE,
    stock_code TEXT,
    industry   TEXT NOT NULL
);

-- 사업연도별 사업보고서 1건 (수집 실패도 status/fail_reason으로 기록). 원본 XML은 raw_xml, ZIP은 zip_path 파일로 보관.
CREATE TABLE IF NOT EXISTS reports (
    report_id     INTEGER PRIMARY KEY,
    company_id    INTEGER NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    business_year INTEGER NOT NULL,                 -- 사업연도 (제출연도 아님)
    rcept_no      TEXT UNIQUE,                      -- 실패 시 NULL
    rcept_dt      TEXT,
    report_nm     TEXT,
    is_correction INTEGER,
    status        TEXT NOT NULL CHECK (status IN ('OK','FAIL')),
    fail_reason   TEXT,
    zip_path      TEXT,
    raw_xml       BLOB,
    collected_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (company_id, business_year)
);

CREATE TABLE IF NOT EXISTS sections (
    section_id    INTEGER PRIMARY KEY,
    report_id     INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    section_no    INTEGER NOT NULL,                 -- II장 내 순서 (1..)
    title         TEXT NOT NULL,                    -- 예: '3. 원재료 및 생산설비'
    analysis_area TEXT NOT NULL,                    -- BUSINESS/PRODUCT/.../UNMAPPED
    xml_line      INTEGER,
    UNIQUE (report_id, section_no)
);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id      INTEGER PRIMARY KEY,
    report_id        INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    section_id       INTEGER NOT NULL REFERENCES sections(section_id) ON DELETE CASCADE,
    analysis_area    TEXT NOT NULL,                 -- 조회 효율을 위해 sections에서 복사
    subsection_title TEXT,
    paragraph_key    TEXT NOT NULL,                 -- S3-P012 / S3-T004
    sentence_key     TEXT NOT NULL,                 -- S3-P012-01
    text             TEXT NOT NULL,
    source_type      TEXT NOT NULL CHECK (source_type IN ('paragraph','table')),
    table_key        TEXT,
    xml_line         INTEGER,
    is_duplicate     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (report_id, sentence_key)
);

-- 정렬 결과. 방법(tfidf/embedding)별 점수는 별도 행. 미매칭은 한쪽 id가 NULL.
CREATE TABLE IF NOT EXISTS sentence_matches (
    match_id        INTEGER PRIMARY KEY,
    old_sentence_id INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    new_sentence_id INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    method          TEXT NOT NULL CHECK (method IN ('tfidf','embedding')),
    kind            TEXT NOT NULL CHECK (kind IN ('PAIR','UNMATCHED_OLD','UNMATCHED_NEW')),
    rank            INTEGER,
    similarity      REAL,                           -- 정렬 점수 (변화 판정 점수와 별개)
    is_match        INTEGER NOT NULL DEFAULT 0,
    CHECK ((kind = 'PAIR' AND old_sentence_id IS NOT NULL AND new_sentence_id IS NOT NULL)
        OR (kind = 'UNMATCHED_OLD' AND old_sentence_id IS NOT NULL AND new_sentence_id IS NULL)
        OR (kind = 'UNMATCHED_NEW' AND new_sentence_id IS NOT NULL AND old_sentence_id IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_matches ON sentence_matches (method, kind, COALESCE(old_sentence_id, 0), COALESCE(new_sentence_id, 0));

CREATE TABLE IF NOT EXISTS detected_changes (
    change_id        INTEGER PRIMARY KEY,
    old_sentence_id  INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    new_sentence_id  INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    change_type      TEXT NOT NULL CHECK (change_type IN ('NEW','REMOVED','EXPANDED','REDUCED','MODIFIED','UNCHANGED','UNCERTAIN')),
    change_topic     TEXT,
    change_summary   TEXT,                          -- 근거 문장으로 뒷받침되는 사실만
    detection_method TEXT NOT NULL,                 -- tfidf / embedding / rule 등
    model_info       TEXT,                          -- 모델명·임계값
    tfidf_similarity REAL,
    embedding_similarity REAL,
    confidence       REAL,
    review_status    TEXT DEFAULT 'auto',
    CHECK (old_sentence_id IS NOT NULL OR new_sentence_id IS NOT NULL)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_changes ON detected_changes (detection_method, COALESCE(old_sentence_id, 0), COALESCE(new_sentence_id, 0));

CREATE TABLE IF NOT EXISTS industry_summary (
    summary_id             INTEGER PRIMARY KEY,
    industry               TEXT NOT NULL,
    year_from              INTEGER NOT NULL,
    year_to                INTEGER NOT NULL,
    analysis_area          TEXT NOT NULL,
    change_type            TEXT NOT NULL,
    affected_company_count INTEGER NOT NULL,
    analyzed_company_count INTEGER NOT NULL,
    affected_company_ratio REAL NOT NULL,
    UNIQUE (industry, year_from, year_to, analysis_area, change_type)
);

-- 정답 라벨 (평가 데이터). 문장 id가 없는 쪽은 NULL(대응 없음).
CREATE TABLE IF NOT EXISTS gold_labels (
    gold_id           INTEGER PRIMARY KEY,
    old_sentence_id   INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    new_sentence_id   INTEGER REFERENCES sentences(sentence_id) ON DELETE CASCADE,
    correct_alignment INTEGER,
    actual_change     INTEGER,
    change_type       TEXT,
    split             TEXT NOT NULL DEFAULT 'test' CHECK (split IN ('tune','test')),  -- 임계값 조정용/최종 평가용 분리
    annotator         TEXT,
    review_status     TEXT,
    CHECK (old_sentence_id IS NOT NULL OR new_sentence_id IS NOT NULL)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_gold ON gold_labels (COALESCE(old_sentence_id, 0), COALESCE(new_sentence_id, 0), annotator);

CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id    INTEGER PRIMARY KEY,
    method     TEXT NOT NULL,
    task       TEXT NOT NULL,                       -- alignment / change_detection / change_type
    metric     TEXT NOT NULL,                       -- precision_at_k, f1, macro_f1 ...
    value      REAL,                                -- 라벨 없으면 NULL ('not evaluated')
    note       TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (method, task, metric)
);

-- 동일 기업·연도·분석 영역 조회용
CREATE INDEX IF NOT EXISTS ix_reports_company_year ON reports (company_id, business_year);
CREATE INDEX IF NOT EXISTS ix_sentences_report_area ON sentences (report_id, analysis_area);
CREATE INDEX IF NOT EXISTS ix_sections_report_area ON sections (report_id, analysis_area);
CREATE INDEX IF NOT EXISTS ix_matches_old ON sentence_matches (old_sentence_id);
CREATE INDEX IF NOT EXISTS ix_matches_new ON sentence_matches (new_sentence_id);
CREATE INDEX IF NOT EXISTS ix_changes_type ON detected_changes (change_type);
