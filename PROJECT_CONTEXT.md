# Thesis Project Context Document
**Dutch Newspaper Framing of Migration and Climate Change, 2014–2023**
*For Claude cowork context and Data/Methods reference*

---

## 1. Research Question

> How have Dutch newspapers framed migration and climate change over time, and is there evidence of discursive convergence or cross-crisis frame borrowing?

**Argument to test:** Climate journalism may borrow security/humanitarian frames from migration coverage (or vice versa) as the two crises become conceptually linked in public discourse. The pipeline is designed to detect this asymmetrically — it does not assume symmetric convergence.

---

## 2. Corpus Overview

### Source
Nexis Uni full-text exports. Six Dutch national newspapers; two topical queries.

### Outlets (as they appear in `outlet_clean`)
| Outlet | Type |
|---|---|
| AD | Popular/tabloid-leaning broadsheet |
| Telegraaf | Right-leaning tabloid |
| Volkskrant | Centre-left quality paper |
| Trouw | Protestant/progressive quality paper |
| NRC | Liberal quality paper |
| FD | Financial/business paper |

**Note for methods:** AD, Telegraaf, Volkskrant, and Trouw are the primary analytical outlets. NRC and FD were retained in the corpus but are smaller in volume.

### Time window
1 January 2014 – 31 December 2023 (10 years)

### Raw file sizes
| File | Size | Rows |
|---|---|---|
| `data/raw/climate.csv` | 122 MB | 24,397 |
| `data/raw/migration.csv` | 822 MB | 111,948 |

---

## 3. Pipeline Architecture

The pipeline has four sequential stages. Stages 1–2 are complete; Stages 3–4 are in progress.

```
Stage 1: Preprocessing     raw CSVs → climate_clean.csv, migration_clean.csv
Stage 2: Corpus building   clean files → corpus_merged.csv → corpus_sampled.csv
Stage 3: BERTopic          corpus_sampled.csv → topic assignments + labels   ← COMPLETE
Stage 4: LLM annotation    ~1,000 sampled articles → frame codes              ← PENDING
Stage 5: Temporal shift    key term tracking across annual slices              ← PENDING
Stage 6: Dashboard         Dash/Plotly on HuggingFace Spaces                  ← PENDING
```

---

## 4. Stage 1: Preprocessing (`src/preprocess_final.py`)

### Steps (in order)

1. **Load raw files** — read both CSVs with `low_memory=False`, log shape and column names
2. **Drop retired BERTopic columns** — removes prior pipeline artefacts: `processed`, `nouns`, `adjectives`, `verbs`, `topic`, `topic_norm`, `topic_label`, `topic_meta`, `topic_meta_original`, `probability`
3. **Fix doubled titles** — Nexis exported many titles duplicated in the body field (e.g. *"Klimaat loopt vast Klimaat loopt vast"*). Fixed by splitting at the midpoint character and taking the first half when the first and second halves are identical.
4. **Strip Nexis boilerplate** — regex-based removal of metadata lines appended to article bodies (e.g. classification lines, `Load-Date:`, `End of Document`). These lines inflate word counts and contain outlet/author identifiers that must not appear in LLM-submitted text.
5. **Parse and clip dates** — `pd.to_datetime(errors='coerce')`, filter to `2014-01-01 ≤ date ≤ 2023-12-31`. Articles outside this window are dropped.
6. **Normalise outlet names** — raw Nexis outlet strings are inconsistent (`"de Volkskrant"`, `"De Volkskrant"`, `"volkskrant"`, etc.). Mapped to a clean `outlet_clean` column via lookup dictionary. Unmapped strings are flagged.
7. **Deduplicate migration corpus by edition** — Nexis exported both Print and Online editions of many articles. Where both exist for the same title+date+outlet combination, the Online edition is retained and the Print edition dropped. This step applies only to migration (climate raw file did not have this duplication pattern).
8. **Drop short articles** — articles under 100 words (post-boilerplate-strip) are removed. Threshold chosen to eliminate press release fragments and brief news items that carry no substantive frame.
9. **Flag cross-corpus overlap** — articles appearing in both corpora are identified by fuzzy title+date+outlet matching and marked `corpus_overlap = True`. These 695 articles are retained in both corpora but flagged so they can be excluded from cross-corpus comparisons if needed.
10. **Select and standardise output columns** — output schema is identical for both corpora:
    `title`, `body`, `date`, `outlet_clean`, `word_count`, `corpus`, `corpus_overlap`, `year`, `source_file`, `article_id`

### Key constants
```python
RANDOM_SEED     = 42
DATE_MIN        = pd.Timestamp("2014-01-01")
DATE_MAX        = pd.Timestamp("2023-12-31")
MIN_WORD_COUNT  = 100
```

### Output statistics
| File | Rows | Notes |
|---|---|---|
| `data/processed/climate_clean.csv` | 22,266 | 10 columns |
| `data/processed/migration_clean.csv` | 68,760 | 12 columns (extra cols from edition deduplication) |

- **Cross-corpus overlap:** 695 articles appear in both corpora (`corpus_overlap = True`)
- Raw → clean retention: climate 91.3% (24,397 → 22,266); migration 57.9% (111,948 → 68,760; large drop due to edition deduplication + boilerplate)

---

## 5. Stage 2: Corpus Building and Sampling

### Merge (`src/build_corpus.py`)
Merges `climate_clean.csv` and `migration_clean.csv` on their 10 shared columns.

Output: `data/processed/corpus_merged.csv` — **91,026 rows**

### Stratified Annual Sampling (`sample_corpus.py`)

**Why sample?** The full 91,026-article corpus caused RAM exhaustion on Google Colab T4 (12 GB) during multi-run BERTopic coherence scoring. Additionally, BERTopic on an unsampled decade-long corpus produced ~2,000 topics dominated by temporal drift rather than thematic structure — this was flagged by the supervisor as an analytical problem.

**Sampling strategy:**
- Each corpus (migration, climate) sampled independently
- Target: **500 articles per year** per corpus
- Within each year, sampled **proportionally by outlet** (stratified)
- If a year has fewer than 500 articles, the **full year is taken**
- Random seed: 42 (reproducible)

**Output:** `data/processed/corpus_sampled.csv`

| Corpus | Articles | Notes |
|---|---|---|
| Migration | 5,000 | 500/year × 10 years, all years met target |
| Climate | 4,948 | 2014 had only 448 articles; full year taken |
| **Total** | **9,948** | |

**Date range:** 2014–2023 for both corpora
**Shuffle:** combined file shuffled (seed 42) before saving so corpus label is not sorted

---

## 6. Stage 3: BERTopic Modelling

Run in **Google Colab** (T4 GPU, ~15 GB RAM). Notebook: `bertopic_dutch_news.ipynb` (generated from `build_notebook.py`).

### Architecture

```
Text → Sentence Embeddings → UMAP → HDBSCAN → c-TF-IDF topic representation
```

**Component choices:**

| Component | Choice | Rationale |
|---|---|---|
| Embedding model | `paraphrase-multilingual-mpnet-base-v2` (SBERT) | Multilingual, strong Dutch performance, 768-dim |
| Dimensionality reduction | UMAP | Standard BERTopic default; preserves local structure |
| Clustering | HDBSCAN | Density-based, handles non-globular clusters, produces outliers (topic = -1) |
| Topic representation | c-TF-IDF | Class-based TF-IDF; identifies words distinctive to each topic |
| Vectorizer | `CountVectorizer` with Dutch stopwords | Prevents function words dominating topic keywords |

**Dutch stopwords:** loaded from `spacy.load("nl_core_news_sm").Defaults.stop_words`. Applied to `CountVectorizer` in both `build_model()` and `update_topics()`. Without stopwords, all topic keywords were Dutch function words (de, het, van, en, is, dat) and coherence was 0.35.

### Hyperparameters (final values)
```python
EMBEDDING_MODEL    = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE         = 64
N_NEIGHBORS        = 15       # UMAP
N_COMPONENTS       = 5        # UMAP output dimensions
MIN_DIST           = 0.0      # UMAP (tighter clusters)
MIN_CLUSTER_SIZE   = 30       # HDBSCAN
MIN_SAMPLES        = 5        # HDBSCAN
CLUSTER_SELECTION  = "eom"    # HDBSCAN (Excess of Mass — larger, more stable clusters)
N_TOP_WORDS        = 10       # c-TF-IDF keywords per topic
MIN_DF             = 5        # CountVectorizer minimum document frequency
NGRAM_RANGE        = (1, 2)   # unigrams + bigrams
SEEDS              = [42, 123, 456, 789, 1024]   # 5 runs for stability assessment
```

**Why these hyperparameters:** `MIN_CLUSTER_SIZE=30` was reduced from 50 (used on the full corpus) because the sampled corpus has only ~5,000 documents per corpus. `eom` selection was chosen over `leaf` to produce broader, more interpretable topic groupings. `MIN_DIST=0.0` forces tighter UMAP clusters, which feeds cleaner inputs to HDBSCAN.

### Embedding caching
Embeddings are computed once and saved to `sampled_embeddings.npy` on Google Drive to avoid recomputation across sessions. The file has shape `(9948, 768)`. **Important:** the cache file must be named for the sampled corpus — an earlier bug loaded an old `all_embeddings.npy` (91,026 rows) into a 9,948-row corpus, causing an `IndexError`.

### Multi-run stability protocol
Five independent runs were performed per corpus using different random seeds. Models were compared using:

| Metric | How computed |
|---|---|
| Gensim c_v coherence | `CoherenceModel(model='c_v', texts=tokenized_docs, ...)` on top-10 words per topic |
| Jaccard similarity | Pairwise topic overlap on top-10 word sets, matched with Hungarian algorithm |
| Cosine similarity | Pairwise similarity on topic word-score vectors |
| Jensen-Shannon distance | Pairwise distance on topic word probability distributions |

**Why Hungarian algorithm matching:** BERTopic assigns arbitrary topic IDs per run. Topics must be matched across runs before any pairwise comparison. The Hungarian algorithm finds the optimal 1:1 assignment minimising total distance.

### Results (seed=42 model, post-outlier-reduction)

**Migration corpus:**
| Metric | Value |
|---|---|
| Topics | 36 |
| Initial outliers (topic = -1) | 1,873 (37.5%) |
| Outliers after reduction | 0 (0%) |
| Mean c_v coherence (5 runs) | 0.644 |
| Mean pairwise Jaccard stability | 0.715 |
| Mean pairwise cosine stability | ~0.83 |
| Mean pairwise JS distance | ~0.24 |

**Climate corpus:**
| Metric | Value |
|---|---|
| Topics | 42 |
| Initial outliers (topic = -1) | 1,828 (37.0%) |
| Outliers after reduction | 0 (0%) |
| Mean c_v coherence (5 runs) | 0.646 |
| Mean pairwise Jaccard stability | 0.722 |
| Mean pairwise cosine stability | ~0.83 |
| Mean pairwise JS distance | ~0.23 |

### Outlier reduction (manual embeddings method)

BERTopic's built-in `reduce_outliers()` could not be used because an earlier failed run had called `update_topics()`, setting the model's internal outlier counter to 0 (causing "No outliers to reduce" error on retry).

**Method implemented (`reduce_outliers_cell.py`):**
1. Compute mean centroid embedding for each topic from its assigned documents
2. For each outlier document, compute cosine similarity to all centroids
3. Assign outlier to the nearest centroid (argmax)
4. Call `update_topics()` with the Dutch stopwords vectorizer explicitly

This is mathematically equivalent to BERTopic's built-in `strategy="embeddings"` approach.

### Topic outputs saved to Google Drive
```
/content/drive/MyDrive/thesis/bertopic_outputs/
├── migration/
│   ├── migration_topic_assignments.csv   (5,000 rows; cols: all original + topic_id, outlier_reduced)
│   └── migration_topic_labels.csv        (36 topics; cols: Topic, Count, Name, keywords, word_scores)
└── climate/
    ├── climate_topic_assignments.csv     (4,948 rows)
    └── climate_topic_labels.csv         (42 topics)
```

---

## 7. Key Topic Findings (BERTopic)

### Migration — top topics by size

| Topic | Count | % | Keywords | Interpretive label |
|---|---|---|---|---|
| T0 | 455 | 9.1% | partij, vvd, partijen, rutte, wilders, cda, pvv | Dutch electoral politics |
| T1 | 336 | 6.7% | asielzoekers, coa, opvang, gemeente, vluchtelingen, statushouders, azc | Asylum reception system |
| T7 | 303 | 6.1% | mensen, leven, jaar, film, zien, heel, wereld, maken | Human interest / personal narrative |
| T2 | 219 | 4.4% | nederland, mensen, jaar, inwoners, aantal, arbeidsmigranten, werk | Labour migration |
| T10 | 202 | 4.0% | oorlog, vader, joodse, joden, duitse, jaar, moeder, duitsers | WWII/Jewish history |
| T21 | 201 | 4.0% | asielzoekers, staatssecretaris, vluchtelingen, vvd | Policy/political discourse on asylum |
| T4 | 199 | 4.0% | woningen, arbeidsmigranten, statushouders | Housing/integration |
| T24 | 179 | 3.6% | asielzoekers, vluchtelingen, ind, opvang | IND/asylum processing |

### Climate — top topics by size

| Topic | Count | % | Keywords | Interpretive label |
|---|---|---|---|---|
| T0 | 330 | 6.7% | water, droogte, waterschap, jaar, overstromingen, dijken, klimaatverandering | Water management / physical impacts |
| T7 | 207 | 4.2% | bedrijven, jaar, nederland, nieuwe, landen, economie | Green economy / business |
| T6 | 201 | 4.1% | partij, partijen, vvd, cda, d66, rutte, groenlinks, verkiezingen | Dutch climate politics |
| T27 | 196 | 4.0% | landen, uitstoot, bedrijven, klimaatverandering, klimaat, nederland, co2 | International emissions / climate policy |
| T1 | 191 | 3.9% | soorten, dieren, natuur, insecten, vogels, biodiversiteit | Biodiversity / nature |
| T9 | 185 | 3.7% | klimaatverandering, opwarming, aarde | General climate awareness |
| T2 | 163 | 3.3% | shell, olie, fossiele, abp, aandeelhouders | Corporate responsibility / Shell litigation |
| T4 | 160 | 3.2% | trump, biden, china, amerikaanse | US/international climate politics |
| T3 | 158 | 3.2% | boeren, vlees, landbouw | Agriculture / food |

### Cross-corpus discourse finding (key analytical result)

**Cross-crisis search procedure:**
- Searched migration topic keywords for climate terms: `klimaat`, `klimaatverandering`, `opwarming`, `co2`, `uitstoot` → **zero matches**
- Searched climate topic keywords for migration terms: `vluchteling`, `migratie`, `migranten`, `asiel`, `vluchtelingen` → **one match: Topic 25**

**Climate Topic 25 — "klimaatvluchtelingen" (migration-as-climate-displacement):**
- Keywords: `migratie, migranten, vn, afrika, vluchtelingen, klimaat, landen, mensen`
- Count: 68 articles (1.4% of climate sample)
- Temporal peak: 2017 (13 articles, 2.6% of that year's climate sample)
- Declined after 2017; episodic rather than sustained

**Interpretation:** The cross-crisis discourse is **asymmetric**. Climate journalism discusses migration as a consequence of climate displacement (particularly in a 2015–2017 window coinciding with the European migration crisis). Migration journalism does not discuss climate causation. This asymmetry is itself an analytical finding.

**Manual verification of 2017 Topic 25 articles:** Most were general international affairs pieces clustering around UN/global governance vocabulary. Genuine cross-crisis framing confirmed in 1–2 articles — notably Trouw, 2017-11-01: *"Nieuw-Zeeland wil een uitweg bieden aan klimaatvluchtelingen"*.

---

## 8. Key Design Decisions and Rationale

| Decision | Alternative considered | Why this choice |
|---|---|---|
| Run BERTopic on two separate corpora (not one joint corpus) | Joint corpus BERTopic | Supervisor feedback: separate models allow corpus-specific topic structures; cross-crisis is then a comparison not an assumption |
| Stratified annual sampling at 500/year | Full corpus | RAM constraints on Colab T4; also prevents temporal drift dominating topics |
| `paraphrase-multilingual-mpnet-base-v2` | `GroNLP/bert-base-dutch-cased` | Dutch BERT had MISSING/UNEXPECTED key warnings (mismatched weights); multilingual SBERT produced no warnings and Dutch benchmarks are strong |
| Dutch spaCy stopwords in CountVectorizer | Default English stopwords or none | Without Dutch stopwords, all topic keywords were function words (de, het, van, en); coherence was 0.35 vs 0.64 with Dutch stops |
| Manual embeddings-based outlier reduction | BERTopic's built-in `reduce_outliers()` | Internal state corruption from failed prior run set outlier counter to 0; manual implementation is equivalent and bypasses the check |
| 5-seed stability assessment | Single run | Standard practice in computational social science; validates that topics are stable structures not artefacts of random initialisation |
| `eom` HDBSCAN cluster selection | `leaf` selection | Produces larger, more coherent topics appropriate for framing analysis; `leaf` over-splits into micro-topics |
| Keep corpus_overlap articles in both corpora | Remove from one | 695 is small (0.76%); flagged for exclusion in targeted analyses; removing would discard valid articles |

---

## 9. Known Issues and Resolutions

| Issue | Resolution |
|---|---|
| Embedding model key warnings (GroNLP) | Switched to `paraphrase-multilingual-mpnet-base-v2` |
| Session RAM crash during 5-run coherence scoring | Stratified sampling (9,948 vs 91,026 articles); `del model; gc.collect(); torch.cuda.empty_cache()` between runs |
| `IndexError: boolean index mismatch` when loading corpus | Old `all_embeddings.npy` (91,026 rows) cached; renamed cache to `sampled_embeddings.npy` |
| BERTopic "No outliers to reduce" error | Manual cosine-similarity outlier assignment bypassing internal state check |
| All 1,873 outliers assigned to Topic 0 | Switched from `strategy="probabilities"` (HDBSCAN soft probs all ~0) to manual embeddings centroid method |
| Stopwords reappearing after `update_topics()` | Must pass `vectorizer_model=vectorizer_clean` explicitly to `update_topics()` — not inherited from fit |
| `list[str] | None` type hint syntax error (Python 3.9) | Used `Optional[List[str]]` from `typing` module |
| `"""` docstrings breaking `r"""..."""` outer strings in build_notebook.py | Changed all inner docstrings to `'''` single-quote style |

---

## 10. Remaining Pipeline

### Stage 4: LLM Frame Annotation (PENDING)

**Goal:** Go beyond BERTopic topic labels to identify analyzable framing patterns within each topic cluster.

**Plan:**
- Sample ~1,000 articles per corpus, stratified by topic (proportional to topic size)
- **Strip all Nexis identifiers** before sending to LLM: remove `outlet_clean`, `date`, `title`, author, all boilerplate — send only anonymised body text (copyright concern, supervisor instruction)
- Annotation schema to be designed — likely deeper than standard 5-class Entman frame (Securitization / Humanitarian / Economic / Scientific / Political Conflict); should capture tone, actor prominence, causal attribution
- **Validation protocol:** multi-run consistency (same article, re-annotated), prompt sensitivity testing (paraphrase prompts, check label stability), ~100-article human validation

### Stage 5: Temporal Semantic Shift Analysis (PENDING)

**Goal:** Track how key Dutch terms change meaning/context over time.

**Key terms to track:** `vluchteling` (refugee), `klimaatvluchteling` (climate refugee), `klimaatcrisis` (climate crisis), `opvang` (reception/shelter), `grenzen` (borders)

**Method:** Annual time slices, word embedding comparison across slices, or BERTopic temporal topic modelling.

### Stage 6: Dashboard (PENDING)

Dash/Plotly app on HuggingFace Spaces using pre-computed BERTopic outputs. Planned views: topic distribution over time per corpus, cross-corpus overlap map, frame annotation summary.

---

## 11. File Structure

```
/Users/meghanreilly/Desktop/thesis/
├── data/
│   ├── raw/
│   │   ├── climate.csv                  122 MB  — Nexis export, gitignored
│   │   └── migration.csv                822 MB  — Nexis export, gitignored
│   └── processed/
│       ├── climate_clean.csv            22,266 rows × 10 cols
│       ├── migration_clean.csv          68,760 rows × 12 cols
│       ├── corpus_merged.csv            91,026 rows × 10 cols
│       └── corpus_sampled.csv           9,948 rows (5,000 mig + 4,948 clim)
├── src/
│   ├── preprocess_final.py              unified preprocessing pipeline
│   └── build_corpus.py                  merges clean corpora
├── sample_corpus.py                     stratified annual sampling
├── bertopic_dutch_news.ipynb            Colab notebook (26 cells)
├── build_notebook.py                    generates the .ipynb from Python source
├── reduce_outliers_cell.py              v3 manual outlier reduction — paste into Colab
├── requirements.txt                     pandas>=2.0
└── README.md

Google Drive (Colab outputs):
/content/drive/MyDrive/thesis/bertopic_outputs/
├── sampled_embeddings.npy               shape (9948, 768) — cached SBERT embeddings
├── migration/
│   ├── migration_topic_assignments.csv  5,000 rows + topic_id col
│   └── migration_topic_labels.csv       36 topics + keywords + word_scores
└── climate/
    ├── climate_topic_assignments.csv    4,948 rows + topic_id col
    └── climate_topic_labels.csv         42 topics + keywords + word_scores
```

---

## 12. Suggested Data/Methods Section Outline

Based on what has been built, the methods section should cover:

**2.1 Data Collection and Corpus Construction**
- Source: Nexis Uni; query terms; date range; outlets
- Preprocessing steps (boilerplate stripping, deduplication, length filter)
- Corpus statistics (pre/post cleaning; cross-corpus overlap)
- Stratified sampling rationale and procedure

**2.2 Topic Modelling**
- BERTopic overview; component stack (embedding → UMAP → HDBSCAN → c-TF-IDF)
- Model choices with rationale (multilingual SBERT, Dutch stopwords, hyperparameters)
- Stability validation (5 seeds, Jaccard/cosine/JS metrics)
- Outlier handling
- Resulting topic counts and coherence

**2.3 Cross-Corpus Discourse Analysis**
- Topic keyword search procedure
- Finding: asymmetric cross-crisis discourse; Climate Topic 25
- Temporal distribution; manual verification

**2.4 LLM Frame Annotation** *(when complete)*
- Sampling strategy; anonymisation procedure; annotation schema
- Validation: consistency re-runs, prompt sensitivity, human coding

**2.5 Temporal Semantic Shift** *(when complete)*
- Method; key terms; time slices

---

*Document generated 2026-06-08. For questions about this pipeline, see conversation history at:*
*`/Users/meghanreilly/.claude/projects/-Users-meghanreilly-Desktop-thesis/6e459aa5-d59c-4c0d-bc9b-330f1eeb36f3.jsonl`*
