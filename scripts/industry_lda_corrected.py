"""Industry LDA on TF-IDF changed sentences (corrected version).

Differences from industry_lda_compare.py:
  * Documents = only the changed sentences of a paragraph (unchanged context removed),
    split into chunks of at most CHUNK consecutive changed sentences.
  * Sections '5. 위험관리 및 파생거래' excluded; footnote / source-citation sentences removed.
  * Paragraphs with >100 sentences (parsing failures that merged financial notes) excluded.
  * One shared stopword list; company names and company-specific terms
    (>= SPECIFIC_SHARE of normalized document frequency in one company) removed.
  * LDA trained on a company-balanced sample (each company capped at BALANCE x smallest),
    then applied to all documents.
  * k chosen over K_RANGE x SEEDS by NPMI + top-word diversity + cross-seed stability.

Usage: python scripts/industry_lda_corrected.py semiconductor|battery|biopharma
"""
import csv, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from kiwipiepy import Kiwi
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

root = Path(__file__).resolve().parents[1]
industry = sys.argv[1]
INDUSTRIES = {'semiconductor': ['SK하이닉스', '삼성전자', '한미반도체'],
              'battery': ['LG에너지솔루션', '삼성SDI', 'SK이노베이션'],
              'biopharma': ['삼성바이오로직스', '셀트리온', '유한양행']}
companies = INDUSTRIES[industry]
CHUNK, MAX_PARA, BALANCE, SPECIFIC_SHARE = 8, 100, 3.0, 0.9
K_RANGE, SEEDS = range(4, 11), [0, 1, 2, 3, 4]
CHANGE_TYPES = {'NEW', 'MODIFIED', 'EXPANDED', 'REDUCED'}
EXCLUDED_SECTION = re.compile(r'위험관리 및 파생거래')
out = root / 'data/results_industry_lda_corrected' / industry
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- stopwords
STOP = set('''
경쟁력 당사 회사 기업 사업 관련 경우 영향 통해 기준 분야 전망 예상 지속 증가 감소 확대 축소 성장 수치 비중 기록 발생
적용 제공 진행 운영 수행 당기 전기 금액 해당 전년 올해 지난해 다음 전체 부분 상황 정도 시기 수준 현재 미래 측면
내용 위한 대응 확보 추진 강화 개선 위하여 최근 중심 가운데 연도 이후 또한 당시 내외 국내 국외 각종 이번 과거 처음
시간 당사자 우리 모든 각각 상기 아래 관계 전반 발표 대한 대하여 이후 이전 기간 시점 예정 계획 노력 다양 지속적
시장 수요 제품 기술 산업 글로벌 세계 주요 대비 담당 출처 자료 참조 가트너 gartner idc omdia trendforce
억원 백만원 천원 달러 비율 규모 실적 사항 현황 부문 사업부 사업부문 연결 종속 자회사 법인 지배
공시 작성 기준일 서류 포함 이상 이하 가능 변화 활용 기반 바탕 보유 특성 방법 기대 사용 필요 일부 다수 해외 신규
레미케이드 아바스틴 허셉틴 휴미라 엔브렐 루센티스 솔리리스 악템라 스텔라라 프롤리아 아일리아 졸레어 리툭산 키트루다
램시마 트룩시마 허쥬마 유플라이마 베그젤마 짐펜트라 스테키마 렉라자 옴리클로 앱토즈마 에피스클리 오퓨비즈 하드리마
'''.split())

STOP_EXTRA = set('''
등 및 또는 등등 기타 기타사항 참고 비고 별도
동일 상이 차이 여부 관련하여 관련된 관한
따른 따라 따라서는 의한 인한 비롯 비롯한
통한 중심으로 중심적 종합 종합적
일반 일반적 기본 기본적 구체적
적극 적극적 효율적 효과적
목적 목표 방침 방향 관점
상세 세부 별첨 첨부 기재 명시
단위 합계 총계 총액
첫째 둘째 셋째 채용 인재 인력 직원 임직원
'''.split())

STOP = STOP | STOP_EXTRA


COMPANY_NAMES = set('''
삼성 삼성전자 하이닉스 sk 한미 한미반도체 엘지 lg 에너지솔루션 sdi 이노베이션 셀트리온 유한 유한양행 바이오로직스
에피스 하만 harman 갤럭시 galaxy 에스케이 이노 sk온 skon 온 엔무브 지오센트릭 에너지 lges samsung hynix celltrion
'''.split()) - {'에너지'}
STOP |= COMPANY_NAMES
UNIT_CODES = set('gb tb mb gbps tc ct dx ds co ltd inc corp mm nm um kg km kwh mwh gwh twh mw gw kw ppm usd krw eur'.split())

kiwi = Kiwi()
tag_counts = defaultdict(Counter)   # used to restrict company-specific removal to proper nouns / Latin tokens
def tokenize(text):
    toks = []
    for z in kiwi.tokenize(text):
        w = z.form.lower()
        if z.tag not in ('NNG', 'NNP', 'SL') or len(w) < 2 or w in STOP:
            continue
        if re.fullmatch(r'[ivx]+|p\d+', w) or w in UNIT_CODES:   # roman numerals, units, product-code prefixes
            continue
        tag_counts[w][z.tag] += 1
        toks.append(w)
    return toks

def is_name_like(w):
    """Proper noun or Latin-script token (brands, product names, subsidiaries), not a common noun."""
    tag = tag_counts[w].most_common(1)[0][0]
    return tag in ('NNP', 'SL')

def bad_sentence(text):
    if len(text) < 25 or len(re.findall(r'[가-힣]', text)) < 15:
        return True
    if re.match(r'^\s*(※|\*|주\s*\d*\s*[\)）:：]|\(주\d*\)|\(?\s*출처|\(?\s*자료\s*[:：])', text):
        return True
    return bool(re.search(r'참조하시기 바랍니다|공시서류의|자세한 사항은|해당사항 없', text))

# ---------------------------------------------------------------- business scope (multi-business companies)
# Only the industry-relevant business of a conglomerate is kept. A sentence is judged by its own keywords;
# if it has none (or a tie), the keywords of its whole paragraph (title path + all sentences) decide.
SCOPE = {
    '삼성전자': dict(
        keep=re.compile(r'반도체|메모리|파운드리|웨이퍼|낸드|디램|시스템\s*LSI|System\s*LSI|이미지\s*센서|'
                        r'(?<![A-Za-z])(?:DS|DRAM|NAND|HBM|SSD|LSI|SoC|AP|CIS|Foundry|DDR\d?|LPDDR\d?|GAA|EUV|nm)(?![A-Za-z])', re.I),
        drop=re.compile(r'TV|텔레비전|모니터|가전|냉장고|세탁기|에어컨|오디오|헤드폰|스피커|Harman|하만|전장|콕핏|'
                        r'텔레매틱스|디스플레이|패널|OLED|QLED|LCD|SDC|Galaxy|갤럭시|스마트폰|HHP|폴더블|태블릿|'
                        r'웨어러블|네트워크\s*시스템|Set\s*제품|(?<![A-Za-z])(?:DX|MX|VD|DA)(?![A-Za-z])', re.I),
        neutralize=re.compile(r'(?!x)x')),   # nothing to neutralize
    'SK이노베이션': dict(
        keep=re.compile(r'배터리|전지|전기차|SK\s*On|SK\s*온|분리막|아이이테크놀로지|양극|음극|전해|리튬|니켈|코발트|'
                        r'소재|리사이클|(?<![A-Za-z])(?:ESS|BaaS|x?EV|LiBS|FCW|SKIET|BMS)(?![A-Za-z])', re.I),
        drop=re.compile(r'석유|원유|정유|윤활|기유|납사|나프타|아로마틱|파라자일렌|올레핀|폴리머|화학제품|지오센트릭|엔무브|'
                        r'루브리컨츠|트레이딩|LNG|도시가스|천연가스|가스공사|집단에너지|전력|발전소|발전사업|송배전|'
                        r'E&S|이엔에스|수소|충전소|충전\s*사업|충전기|광구|탐사|아스팔트|항공유|벙커|'
                        r'플라스틱|촉매|Nexlene|에너지솔루션\s*사업|재생\s*에너지|신재생', re.I),
        neutralize=re.compile(r'전기차\s*충전')),   # EV charging belongs to the E&S business, not batteries
}
def in_scope(company, sid, text, pid):
    rule = SCOPE.get(company)
    if rule is None:
        return True
    def score(t):
        return len(rule['keep'].findall(rule['neutralize'].sub(' ', t))), len(rule['drop'].findall(t))
    k, d = score(text)
    if k != d:
        return k > d
    p = paras[pid]
    k, d = score(' '.join(p['title_path']) + ' ' + ' '.join(s['text'] for s in p['sentences']))
    return k > d

# ---------------------------------------------------------------- paragraphs
paras = {}
for line in (root / f'data/processed/preprocessing_v2/{industry}/paragraphs.jsonl').open(encoding='utf8'):
    p = json.loads(line)
    if p.get('company') in companies and p.get('source_type') == 'paragraph':
        paras[p['paragraph_id']] = p
sent_para, sent_order = {}, {}
for pid, p in paras.items():
    for i, s in enumerate(p['sentences']):
        sent_para[s['sentence_id']] = pid
        sent_order[s['sentence_id']] = i

def para_ok(pid):
    p = paras[pid]
    return len(p['sentences']) <= MAX_PARA and not EXCLUDED_SECTION.search(' '.join(p['title_path']))

# ---------------------------------------------------------------- changed sentences
tfidf_rows = list(csv.DictReader((root / 'data/results_v2/tfidf_masked_th0.6.csv').open(encoding='utf-8-sig')))
changed, dropped, out_of_scope = {}, Counter(), {}
for r in tfidf_rows:
    if r['company'] not in companies or r['change_type'] not in CHANGE_TYPES:
        continue
    sid, text = r['new_id'], r['new_text'].strip()
    pid = sent_para.get(sid)
    if pid is None or paras[pid]['company'] != r['company']:
        dropped['not in paragraph source'] += 1; continue
    if len(paras[pid]['sentences']) > MAX_PARA:
        dropped['oversized paragraph (parsing error)'] += 1; continue
    if EXCLUDED_SECTION.search(' '.join(paras[pid]['title_path'])):
        dropped['risk/derivatives section'] += 1; continue
    if bad_sentence(text):
        dropped['footnote/short/reference'] += 1; continue
    if not in_scope(r['company'], sid, text, pid):
        if sid not in out_of_scope:
            dropped['out of industry scope'] += 1
            out_of_scope[sid] = (r['company'], r['year_new'], r['change_type'], ' > '.join(paras[pid]['title_path'][1:]), text)
        continue
    c = changed.setdefault(sid, dict(sentence_id=sid, company=r['company'], year=r['year_new'],
                                     paragraph_id=pid, text=text, change_types=set()))
    c['change_types'].add(r['change_type'])
for c in changed.values():
    c['change_types'] = '|'.join(sorted(c['change_types']))
if out_of_scope:
    with (out / 'excluded_out_of_scope.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['sentence_id', 'company', 'year', 'change_type', 'section', 'text'])
        for sid, row in sorted(out_of_scope.items()):
            w.writerow([sid, *row])

# ---------------------------------------------------------------- documents: chunks of changed sentences
by_para = defaultdict(list)
for sid, c in changed.items():
    by_para[c['paragraph_id']].append(sid)
docs = []  # dict(doc_id, company, year, sids, text)
for pid, sids in sorted(by_para.items()):
    sids.sort(key=sent_order.get)
    for n in range(0, len(sids), CHUNK):
        part = sids[n:n + CHUNK]
        docs.append(dict(doc_id=f'{pid}-C{n // CHUNK + 1}', company=paras[pid]['company'],
                         year=changed[part[0]]['year'], sids=part,
                         text=' '.join(changed[s]['text'] for s in part)))
for d in docs:
    d['tokens'] = tokenize(d['text'])
docs = [d for d in docs if len(d['tokens']) >= 3]
kept_sids = {s for d in docs for s in d['sids']}
dropped['fewer than 3 tokens'] += len(changed) - len(kept_sids)
changed = {s: c for s, c in changed.items() if s in kept_sids}

# ---------------------------------------------------------------- company-specific terms
n_company = Counter(d['company'] for d in docs)
df_company = defaultdict(Counter)
for d in docs:
    for w in set(d['tokens']):
        df_company[w][d['company']] += 1
specific = {}
for w, cnt in df_company.items():
    if sum(cnt.values()) < 3 or not is_name_like(w):
        continue
    rel = {c: cnt[c] / n_company[c] for c in companies if n_company[c]}
    top_c = max(rel, key=rel.get)
    share = rel[top_c] / sum(rel.values())
    if share >= SPECIFIC_SHARE:
        specific[w] = (top_c, share, sum(cnt.values()))
for d in docs:
    d['tokens'] = [w for w in d['tokens'] if w not in specific]
with (out / 'removed_company_specific_terms.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['term', 'company', 'normalized_share', 'doc_freq'])
    for t, (c, s, n) in sorted(specific.items(), key=lambda x: (-x[1][2], x[0])):  # 동률은 용어 순(실행마다 같은 순서)
        w.writerow([t, c, round(s, 3), n])

# ---------------------------------------------------------------- vectorize
vec = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=3, max_df=.5)
X = vec.fit_transform([' '.join(d['tokens']) for d in docs])
terms = vec.get_feature_names_out()
nonempty = np.asarray(X.sum(axis=1)).ravel() > 0
binary = (X > 0).astype(np.float64).tocsc()
N = X.shape[0]
p_w = np.asarray(binary.sum(axis=0)).ravel() / N

# company-balanced training sample (stratified by year inside company)
rng = np.random.default_rng(42)
cap = int(BALANCE * min(n_company.values()))
train_idx = []
for c in companies:
    idx = [i for i, d in enumerate(docs) if d['company'] == c and nonempty[i]]
    if len(idx) > cap:
        by_year = defaultdict(list)
        for i in idx:
            by_year[docs[i]['year']].append(i)
        take = []
        for y, ii in by_year.items():
            m = round(cap * len(ii) / len(idx))
            take += list(rng.choice(ii, size=min(m, len(ii)), replace=False))
        idx = take
    train_idx += idx
train_idx = np.array(sorted(train_idx))
Xtrain = X[train_idx]

def npmi(top):
    vals = []
    for a in range(len(top)):
        for b in range(a):
            pij = binary[:, top[a]].multiply(binary[:, top[b]]).sum() / N
            if pij == 0:
                vals.append(-1.0); continue
            vals.append(np.log(pij / (p_w[top[a]] * p_w[top[b]])) / -np.log(pij))
    return float(np.mean(vals))

def tops(model, n=10):
    return [list(c.argsort()[-n:][::-1]) for c in model.components_]

def match(ma, mb):
    ta, tb = [set(t) for t in tops(ma)], [set(t) for t in tops(mb)]
    J = np.array([[len(x & y) / len(x | y) for y in tb] for x in ta])
    r, c = linear_sum_assignment(-J)
    return J[r, c].mean()

def stability(models):
    return float(np.mean([match(models[a], models[b]) for a in range(len(models)) for b in range(a)]))

def central(runs):
    """Run whose topics agree most with the other seeds (medoid), not the luckiest seed."""
    ms = [r[0] for r in runs]
    return max(range(len(ms)), key=lambda a: np.mean([match(ms[a], ms[b]) for b in range(len(ms)) if b != a]))

# ---------------------------------------------------------------- model selection
grid = {}
sel_rows = []
for k in K_RANGE:
    runs = []
    for seed in SEEDS:
        m = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=100,
                                      learning_method='batch', doc_topic_prior=0.1, topic_word_prior=0.05)
        m.fit(Xtrain)
        t = tops(m)
        runs.append((m, float(np.mean([npmi(x) for x in t])), len({w for x in t for w in x}) / (10 * k)))
    stab = stability([r[0] for r in runs])
    grid[k] = (runs, stab)
    row = dict(k=k, npmi=np.mean([r[1] for r in runs]), npmi_sd=np.std([r[1] for r in runs]),
               diversity=np.mean([r[2] for r in runs]), stability=stab)
    sel_rows.append(row)
    print(f"k={k} npmi={row['npmi']:.4f}±{row['npmi_sd']:.4f} div={row['diversity']:.3f} stab={stab:.3f}", flush=True)

def mm(key):
    v = np.array([r[key] for r in sel_rows]); return (v - v.min()) / (v.max() - v.min() + 1e-12)
composite = (mm('npmi') + mm('diversity') + mm('stability')) / 3
for r, s in zip(sel_rows, composite):
    r['composite'] = float(s)
# Fixed after manual review of audit_4..7 (interpretability + cross-seed stability); see model_selection.csv
K_OVERRIDE = {'semiconductor': 5, 'battery': 6, 'biopharma': 5}
k = K_OVERRIDE.get(industry, sel_rows[int(np.argmax(composite))]['k'])
runs, _ = grid[k]
model = runs[central(runs)][0]
with (out / 'model_selection.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(sel_rows[0])); w.writeheader(); w.writerows(sel_rows)

theta = model.transform(X)
theta[~nonempty] = 1.0 / k

# ---------------------------------------------------------------- audits for every k (best seed)
for kk, (rs, stab) in grid.items():
    m = rs[central(rs)][0]
    th = m.transform(X)
    lines = [f'변화 문장 LDA: 토픽 {kk}개, 문서(변화 문장 묶음) {N}개, 학습 문서 {len(train_idx)}개, 단어 {len(terms)}개',
             f"NPMI={np.mean([npmi(x) for x in tops(m)]):.4f}, 다양성={len({w for x in tops(m) for w in x})/(10*kk):.3f}, 시드 안정성={stab:.3f}"]
    for i, top in enumerate(tops(m)):
        comp_share = Counter()
        for j, d in enumerate(docs):
            comp_share[d['company']] += th[j, i]
        tot = sum(comp_share.values())
        share_txt = ', '.join(f'{c} {comp_share[c]/tot:.0%}' for c in companies)
        lines.append(f'\n토픽 {i+1}: ' + ', '.join(terms[top]) + f'   [{share_txt}]')
        seen = set()
        for j in np.argsort(-th[:, i]):
            if len(seen) >= 5: break
            key = docs[j]['text'][:35]
            if key in seen: continue
            seen.add(key)
            lines.append(f"  [{docs[j]['company']} {docs[j]['year']}, 확률 {th[j, i]:.2f}] " + docs[j]['text'][:200])
    (out / f'audit_{kk}.txt').write_text('\n'.join(lines), encoding='utf8')

# ---------------------------------------------------------------- outputs
topics = [[terms[j] for j in t] for t in tops(model)]
labels_file = out / 'topic_labels.json'
custom = json.loads(labels_file.read_text(encoding='utf8')) if labels_file.exists() else {}
labels = [f'T{i+1}: ' + custom.get(str(i + 1), ', '.join(w[:4])) for i, w in enumerate(topics)]
(out / 'topics.txt').write_text('\n'.join(f'T{i+1}: ' + ', '.join(w) for i, w in enumerate(topics)), encoding='utf8')
summary = [f'선택된 토픽 수: {k} (NPMI·다양성·시드 안정성 종합 점수 최대, 시드 중 대표(medoid) 모델{", 수동 지정" if industry in K_OVERRIDE else ""})',
           f'학습 문서: 변화 문장만 문단 단위로 묶고 최대 {CHUNK}문장씩 분할 → {N}개 (회사 균형 학습 샘플 {len(train_idx)}개, 상한 {cap})',
           '회사별 문서 수: ' + ', '.join(f'{c} {n_company[c]}' for c in companies),
           f'회사 특이 단어 제거: {len(specific)}개 (removed_company_specific_terms.csv)',
           '제외된 변화 문장: ' + ', '.join(f'{a} {b}' for a, b in dropped.items())]
(out / 'selected_model.txt').write_text('\n'.join(summary) + '\n', encoding='utf8')
print('\n'.join(summary))

doc_of = {s: j for j, d in enumerate(docs) for s in d['sids']}
with (out / 'sentence_topic_probabilities.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['company', 'year', 'sentence_id', 'paragraph_id', 'doc_id', 'change_types', 'text'] + [f'topic_{i+1}' for i in range(k)])
    for sid, c in sorted(changed.items()):
        j = doc_of[sid]
        w.writerow([c['company'], c['year'], sid, c['paragraph_id'], docs[j]['doc_id'], c['change_types'], c['text'], *theta[j]])

# matched old sentences (MODIFIED/EXPANDED/REDUCED) -> same document construction on the old side
pairs, old_by_para = {}, defaultdict(set)
for r in tfidf_rows:
    if r['company'] not in companies or r['change_type'] not in ('MODIFIED', 'EXPANDED', 'REDUCED'):
        continue
    if r['new_id'] not in changed or not r['old_id'] or r['old_id'] not in sent_para:
        continue
    pid = sent_para[r['old_id']]
    if not para_ok(pid):
        continue
    pairs[(r['old_id'], r['new_id'])] = r
    old_by_para[pid].add(r['old_id'])
old_text = {}
for pid, p in paras.items():
    for s in p['sentences']:
        old_text[s['sentence_id']] = s['text']

def old_side_theta(groups, tag):
    """{paragraph_id: {old sentence ids}} -> {sid: (doc_id, theta)}. 새 쪽과 같은 문서 구성(문단 내 최대 CHUNK문장)에
    기존 vectorizer/model의 transform()만 적용한다(재학습 없음). 토큰이 없는 묶음은 확률 없음."""
    res = {}
    for pid, sids in groups.items():
        sids = sorted(sids, key=sent_order.get)
        for n in range(0, len(sids), CHUNK):
            part = sids[n:n + CHUNK]
            x = vec.transform([' '.join(w for w in tokenize(' '.join(old_text[s] for s in part)) if w not in specific)])
            if x.nnz:
                t = model.transform(x)[0]
                for s in part:
                    res[s] = (f'{pid}-{tag}{n // CHUNK + 1}', t)
    return res

old_matched = old_side_theta(old_by_para, 'OC')
oldtheta = {s: t for s, (_, t) in old_matched.items()}

# REMOVED old sentences: same filters as the new side (paragraph size/section, footnotes, industry scope), then
# transform() with the existing model so they land in the same topic space. Kept separate from the matched groups
# so the matched-side probabilities (and the sankey) stay exactly as before.
removed_by_para, removed_dropped = defaultdict(set), Counter()
for r in tfidf_rows:
    if r['company'] not in companies or r['change_type'] != 'REMOVED':
        continue
    sid, text = r['old_id'], r['old_text'].strip()
    pid = sent_para.get(sid)
    if pid is None or paras[pid]['company'] != r['company']:
        removed_dropped['not in paragraph source'] += 1; continue
    if not para_ok(pid):
        removed_dropped['oversized paragraph or risk/derivatives section'] += 1; continue
    if bad_sentence(text):
        removed_dropped['footnote/short/reference'] += 1; continue
    if not in_scope(r['company'], sid, text, pid):
        removed_dropped['out of industry scope'] += 1; continue
    removed_by_para[pid].add(sid)
old_removed = old_side_theta(removed_by_para, 'RC')
removed_dropped['no vocabulary tokens'] = sum(len(v) for v in removed_by_para.values()) - len(old_removed)

old_types = defaultdict(set)
for r in tfidf_rows:
    if r['company'] in companies and r['old_id'] and (r['old_id'] in old_matched or r['old_id'] in old_removed):
        if r['change_type'] in ('MODIFIED', 'EXPANDED', 'REDUCED', 'REMOVED'):
            old_types[r['old_id']].add(r['change_type'])
with (out / 'sentence_topic_probabilities_old.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['company', 'year', 'sentence_id', 'paragraph_id', 'doc_id', 'old_side', 'change_types', 'text']
               + [f'topic_{i+1}' for i in range(k)])
    for side, d in (('matched', old_matched), ('removed', old_removed)):
        for sid, (doc_id, t) in sorted(d.items()):
            p = paras[sent_para[sid]]
            w.writerow([p['company'], p['business_year'], sid, sent_para[sid], doc_id, side,
                        '|'.join(sorted(old_types[sid])), old_text[sid], *t])

# one row per change pair of the source CSV (5 types) with a stable pair id and the topic side it uses
import hashlib
def pair_id(r):
    key = '|'.join([r['company'], r['year_old'], r['year_new'], r['area'], r['change_type'], r.get('old_id') or '', r.get('new_id') or ''])
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]
new_theta = {sid: theta[doc_of[sid]] for sid in changed}
old_theta_all = {**{s: t for s, (_, t) in old_matched.items()}, **{s: t for s, (_, t) in old_removed.items()}}
with (out / 'sentence_topic_probabilities_pairs.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pair_id', 'source_change_row', 'company', 'year_old', 'year_new', 'area', 'change_type',
                'old_sentence_id', 'new_sentence_id', 'topic_side', 'join_status'] + [f'topic_{i+1}' for i in range(k)])
    for i, r in enumerate(tfidf_rows, start=1):
        if r['company'] not in companies or r['change_type'] not in CHANGE_TYPES | {'REMOVED'}:
            continue
        side, sid = ('old', r['old_id']) if r['change_type'] == 'REMOVED' else ('new', r['new_id'])
        t = (old_theta_all if side == 'old' else new_theta).get(sid)
        w.writerow([pair_id(r), i, r['company'], r['year_old'], r['year_new'], r['area'], r['change_type'],
                    r.get('old_id') or '', r.get('new_id') or '', side, 'ok' if t is not None else 'no_topic',
                    *(t if t is not None else [''] * k)])

# fitted model + everything needed to transform new text into the same topic space
import joblib, sklearn, kiwipiepy
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
src_csv = root / 'data/results_v2/tfidf_masked_th0.6.csv'
src_jsonl = root / f'data/processed/preprocessing_v2/{industry}/paragraphs.jsonl'
meta = dict(industry=industry, companies=companies, n_topics=k, model_seed=SEEDS[central(runs)],
            seeds=SEEDS, k_range=list(K_RANGE), k_override=K_OVERRIDE.get(industry), train_sample_rng_seed=42,
            chunk=CHUNK, max_paragraph_sentences=MAX_PARA, balance=BALANCE, specific_share=SPECIFIC_SHARE,
            input_change_csv=str(src_csv.relative_to(root)), input_change_csv_sha256=sha256_of(src_csv),
            input_paragraphs_jsonl=str(src_jsonl.relative_to(root)), input_paragraphs_jsonl_sha256=sha256_of(src_jsonl),
            versions=dict(sklearn=sklearn.__version__, kiwipiepy=kiwipiepy.__version__, numpy=np.__version__, joblib=joblib.__version__),
            removed_dropped=dict(removed_dropped))
meta['artifact_version'] = f"{industry}-k{k}-{meta['input_change_csv_sha256'][:12]}"
joblib.dump(dict(vectorizer=vec, model=model, specific_terms=sorted(specific), stopwords=sorted(STOP),
                 unit_codes=sorted(UNIT_CODES), metadata=meta), out / 'lda_artifacts.joblib')
(out / 'lda_artifacts_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
print('REMOVED old-side:', len(old_removed), 'sentences, dropped', dict(removed_dropped))

# ---------------------------------------------------------------- figures (same design as industry_lda_compare.py)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from scipy.interpolate import PchipInterpolator
colors = ['#527B92', '#D28A59', '#8E7AA1', '#8EA865', '#B56B75', '#C1A05D', '#518E81', '#7E94B8', '#A67D63', '#84955C', '#926C85']
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'font.size': 10})
years = [2022, 2023, 2024, 2025]
allcounts = []
for company in companies:
    records = [(sid, c) for sid, c in changed.items() if c['company'] == company]
    totals = np.array([sum(c['year'] == str(y) for _, c in records) for y in years])
    values = np.array([sum((theta[doc_of[sid]] for sid, c in records if c['year'] == str(y)), start=np.zeros(k)) for y in years])
    for y, n, row in zip(years, totals, values):
        allcounts.append([company, y, n, *row])
    dense = np.linspace(2022, 2025, 240)
    vals = np.maximum(PchipInterpolator(years, values, axis=0)(dense), 0)
    fig = plt.figure(figsize=(14, 8)); ax = fig.add_axes([.07, .32, .88, .49])
    ax.stackplot(dense, vals.T, baseline='sym', colors=colors[:k], alpha=.9)
    ax.set_xticks(years); ax.set_yticks([]); ax.tick_params(axis='x', length=0, pad=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    for y, n in zip(years, totals):
        ax.text(y, 1.03, f'{n}문장', transform=ax.get_xaxis_transform(), ha='center', color='#64757d')
    fig.text(.07, .95, company + '  |  변화 문장 LDA 토픽 흐름', fontsize=19, weight='bold')
    fig.text(.07, .90, 'TOPIC RIVER  ·  2022–2025', fontsize=10, color='#687980')
    fig.text(.07, .865, '띠의 두께: 변화 문장별 토픽 확률 합계', fontsize=10, color='#687980')
    fig.legend([Patch(facecolor=colors[i]) for i in range(k)], labels, loc='lower left', bbox_to_anchor=(.07, .07),
               ncol=3, frameon=False, fontsize=8.5, labelspacing=.8)
    fig.text(.07, .025, '산업 공동 LDA · 학습 문서는 문단 내 변화 문장만 묶음 · 연도 사이 곡선은 보간', fontsize=9, color='#687980')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{company}_river.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    flows = {y: np.zeros((k, k)) for y in range(2021, 2025)}
    paircount = {y: 0 for y in range(2021, 2025)}
    for (old, new), r in pairs.items():
        if r['company'] != company or old not in oldtheta:
            continue
        y = int(r['year_old'])
        if y not in flows: continue
        flows[y] += np.outer(oldtheta[old], theta[doc_of[new]]); paircount[y] += 1
    with (out / f'{company}_transitions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['year_old', 'year_new', 'old_topic', 'new_topic', 'expected_pair_mass'])
        for y, m in flows.items():
            for i in range(k):
                for j in range(k):
                    w.writerow([y, y + 1, i + 1, j + 1, m[i, j]])
    inc = np.zeros((5, k)); outgoing = np.zeros((5, k))
    for t, y in enumerate(range(2021, 2025)):
        outgoing[t] = flows[y].sum(axis=1); inc[t + 1] = flows[y].sum(axis=0)
    sizes = np.maximum(inc, outgoing); gap = .2
    height = max(max(sum(sizes[t]) + gap * (k - 1) for t in range(5)), 1)
    bottom = np.zeros((5, k))
    for t in range(5):
        start = (height - sum(sizes[t]) - gap * (k - 1)) / 2
        for i in range(k):
            bottom[t, i] = start + sum(sizes[t, :i]) + i * gap
    fig = plt.figure(figsize=(15, 9)); ax = fig.add_axes([.08, .32, .84, .48]); width = .03
    for t, y in enumerate(range(2021, 2025)):
        left = bottom[t].copy(); right = bottom[t + 1].copy()
        for i in range(k):
            for j in range(k):
                h = flows[y][i, j]
                if h <= 0: continue
                a = left[i]; b = right[j]; left[i] += h; right[j] += h; x0 = t + width; x1 = t + 1 - width
                verts = [(x0, a), (x0 + .38, a), (x1 - .38, b), (x1, b), (x1, b + h), (x1 - .38, b + h), (x0 + .38, a + h), (x0, a + h), (x0, a)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colors[i], alpha=.35, edgecolor='none'))
    for t, y in enumerate(range(2021, 2026)):
        for i in range(k):
            if sizes[t, i] > 0:
                ax.add_patch(Rectangle((t - width, bottom[t, i]), 2 * width, sizes[t, i], color=colors[i], lw=0))
                if sizes[t, i] > .35:
                    ax.text(t + width + .03, bottom[t, i] + sizes[t, i] / 2, 'T' + str(i + 1), va='center', fontsize=8)
        ax.text(t, height + .6, str(y), ha='center', fontsize=11, weight='bold')
    for t, y in enumerate(range(2021, 2025)):
        ax.text(t + .5, -.55, f'{paircount[y]}쌍', ha='center', va='top', fontsize=10)
    ax.set_xlim(-.15, 4.28); ax.set_ylim(-1.1, height + 1.3); ax.axis('off')
    fig.text(.08, .95, company + '  |  대응 문장의 토픽 연결', fontsize=19, weight='bold')
    fig.text(.08, .90, 'TOPIC TRANSITION SANKEY  ·  2021–2025', fontsize=10, color='#687980')
    fig.text(.08, .865, '연결선: 대응 문장 쌍의 토픽 확률을 곱한 기대 연결량', fontsize=10, color='#687980')
    fig.legend([Patch(facecolor=colors[i]) for i in range(k)], labels, loc='lower left', bbox_to_anchor=(.08, .07),
               ncol=3, frameon=False, fontsize=8.5, labelspacing=.8)
    fig.text(.08, .025, '신규·삭제 제외 · 확률 연결량은 실제 주제 이동이나 동일 문장의 연속 이동을 뜻하지 않음', fontsize=9, color='#687980')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{company}_sankey.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(company, 'sentences', totals.tolist(), 'pairs', paircount, flush=True)

with (out / 'year_topic_mass.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['company', 'year', 'changed_sentences'] + [f'topic_{i+1}_mass' for i in range(k)])
    w.writerows(allcounts)
