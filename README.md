# Crisis Dashboard: Dutch Media Framing of Climate Change and Migration

Thesis research code for a framing analysis of Dutch national newspaper coverage of
climate change and migration, 2014–2023.

**Outlets:** AD, FD, NRC, Telegraaf, Trouw, Volkskrant
**Time window:** 1 January 2014 – 31 December 2023
**Source:** Nexis Uni (LexisNexis)

## What this does

Two pipelines, run in sequence.

**1. Corpus construction.** Raw Nexis Uni exports are cleaned, deduplicated,
outlet-normalised, and filtered to on-topic articles. Articles appearing in both
corpora are flagged (`corpus_overlap`). A centrality classifier then separates
articles *centrally about* the topic from those that merely mention it. The
resulting annotation set is **4,048 articles** — 1,070 climate and 2,978 migration.

**2. Frame annotation.** Each article is annotated by GPT-4o against a hardcoded
codebook of **16 binary frames**, organised into four passes:

| pass | frames | applies to |
|---|---|---|
| pass1 | conflict, human_interest, economic, deservingness, responsibility | all articles |
| pass2 | humanitarian, security, policy | migration + intersection |
| pass3 | scientific, crisis, solutions, victim, skepticism | climate + intersection |
| pass4 | securitization, othering, agency | all articles |

Each frame carries indicator sub-questions and a presence rule; some carry extra
categorical fields (`deservingness_direction`, `responsibility_responsible_actor`,
`agency_type`, `othering_type`). Annotation runs through the OpenAI Batch API —
12,172 requests, submitted in 26 chunks to stay under the organisation's
enqueued-token limit.

## Validation

The annotation was validated against a human-annotated benchmark of 114 articles
(100 stratified + 14 climate expansion), coded holistically without sight of model
output. Agreement is reported per label as Cohen's κ, Krippendorff's α, raw
agreement and κ_max — with κ_max included because several labels have base rates
that make κ ≥ 0.6 arithmetically unreachable, which raw κ alone would misrepresent.

The agreement tables in `data/validity/` carry the full per-label figures, and the
four-stage human reference-standard lineage in `data/benchmark/snapshots/` records
the annotation as first coded, after an adjudication pass, and after a re-check
against the revised codebook. The scripts that produced them are in `src/`.

Full methodological detail — the codebook-revision history, corrections made during
analysis, and one annotation revision discarded for breaking independence — is held
in working documents kept outside this repository.

## Repository structure

```
crisis-dashboard/
├── data/
│   ├── raw/                    gitignored — Nexis Uni exports
│   ├── processed/              gitignored — cleaned corpora
│   ├── sampled/                gitignored (except strata_report.md)
│   ├── benchmark/
│   │   ├── benchmark_ids*.csv          sample ID manifests
│   │   └── snapshots/                  human reference-standard lineage
│   ├── annotated/
│   │   ├── benchmark_stability/        X3 stability run outputs
│   │   └── final_codebook/             measured token usage from the final benchmark run
│   └── validity/               agreement tables and diagnostics
├── src/                        21 scripts (below)
├── notebooks/
├── requirements.txt
└── README.md
```

### `src/`

**Corpus construction**

| script | purpose |
|---|---|
| `preprocess_final.py` | raw Nexis exports → cleaned corpora |
| `sample.py` | draw the working sample from cleaned corpora |
| `flag_topics.py` | deterministic on-topic flag for both corpora |
| `check_strata.py` | on-topic distribution across outlet × year strata |
| `centrality_classifier.py` | gpt-4o-mini classifier: centrally about the topic, or passing mention |

**Annotation**

| script | purpose |
|---|---|
| `annotate.py` | the codebook: frame definitions, indicators, presence rules, prompts |
| `run_all.py` | synchronous annotation orchestration |
| `batch_pipeline.py` | Batch API pipeline — generate / submit / status / parse, chunked submission, makeup batches |
| `chunk_monitor.py` | automated sequential chunk submission with failure detection and cost tracking |

**Benchmark and validation**

| script | purpose |
|---|---|
| `benchmark_sample_x0b.py` | build the 100-article benchmark from the X0-filtered pool |
| `benchmark_expand_climate.py` | extend the climate benchmark 26 → 40 |
| `benchmark_stability_x3.py` | X3 run-to-run stability sub-study (Prompt A vs B, 3 runs each) |
| `run_final_benchmark.py` | Prompt B over all 114 benchmark articles, final codebook |
| `validity_agreement.py` | κ / α / raw agreement, Prompt A vs B, across runs |
| `validity_final.py` | final validity table with reliability tiers |
| `validity_reference_standard.py` | first-pass vs revision-1 comparison |
| `build_postfix_snapshot.py` | assemble the post-fix reference standard |
| `build_disagreement_sheet.py` | three-way disagreement review artefact |
| `build_recheck_worksheet.py` | codebook re-check worksheet |
| `diagnose_disagreements.py` | independent recomputation of cell census and disagreement direction |
| `diagnose_presence_rule.py` | presence-rule threshold diagnostics |

## What is and is not in this repository

Two separate reasons for exclusion, both applying:

**Size.** The raw and cleaned corpora, sampled corpora, and Batch API payload and
response files run from tens of megabytes to ~820 MB — well past GitHub's limits.

**Licensing.** Source articles come from Nexis Uni under licence and cannot be
republished. This constrains more than the large files: several *small* files also
carry article text and are excluded for licensing reasons alone, including the human
annotation workbooks, the full-corpus annotation output, and the raw model responses
(Prompt B's reasoning quotes source articles at length). File size is not a reliable
proxy for whether a file can be published, so exclusions are listed by name in
`.gitignore` rather than left to a blanket pattern.

**Included:** all pipeline code, the human reference-standard snapshots, agreement
and validity tables, sample ID manifests, measured token usage, the X3 stability run
outputs, the RQ1–4 aggregate tables, the interactive dashboard, and — in redacted
form — the analysis-ready annotated dataset (`data/final/annotated_tidy.csv`) and the
final-codebook benchmark run (`data/annotated/final_codebook/promptB_run1_final.csv`).

To reproduce from source: obtain the Nexis Uni exports, place them in `data/raw/`, and
run the preprocessing pipeline before the annotation pipeline.

### Reproducibility notes

**Redaction.** The two annotated datasets listed above ship with one column redacted.
`othering_othering_evidence` held a one-sentence model quotation or paraphrase of the
source article (max 25 words by prompt design); every such value is replaced with the
literal string `[redacted]`. Rows where the model supplied no evidence keep their
original empty/`null` sentinel, so presence-of-evidence remains analysable. No other
column carries article prose — neither file has a title or body column, and the
remaining free-text fields are closed categoricals or short actor labels. No script in
`src/` reads the evidence column, so redaction does not affect any reported result.

**What reruns from a clone.** Given the two redacted datasets, the entire analysis
layer reproduces without the source corpora: `build_rq1_heatmap.py`,
`build_rq2_temporal.py`, `build_rq3_outlet.py`, `build_rq4_crosscrisis.py`,
`build_dashboard.py`, `chapter5_significance_tests.py`, and `validity_final.py`
(the headline reliability figures).

**What does not rerun.** `validity_agreement.py` reads the human annotation sheet
`data/benchmark/benchmark_annotation_sheet.csv.numbers` directly. That file contains
full article text and is not published, so the script cannot be rerun from a clone.
Its outputs are committed instead and can be inspected as-is:
`data/validity/agreement_per_label.csv`, `agreement_aggregate.csv`,
`agreement_exclusions.csv`, and `agreement_report.md`. The same applies to
`build_disagreement_sheet.py`, `build_recheck_worksheet.py`, and
`build_postfix_snapshot.py`, whose committed outputs are the frozen snapshots under
`data/benchmark/snapshots/`. Stages 0–2 (cleaning, sampling, centrality
classification, annotation) require the licensed Nexis Uni exports and an OpenAI API
key.

**Dashboard.** `dashboard/dashboard.html` opens directly in a browser with no build
step and no local data files — all four RQ payloads are embedded inline. It loads
Plotly from a CDN, so it needs an internet connection to render. The four standalone
panel pages it was assembled from are included alongside it as an audit trail.

## Setup

```bash
pip install -r requirements.txt
```

Requires an OpenAI API key in `.env` as `OPENAI_API_KEY` (gitignored).

## Tooling note

Annotation is performed by OpenAI **GPT-4o** (and **gpt-4o-mini** for the centrality
classifier). Pipeline code in `src/` was developed with the assistance of
**Claude Code** (Anthropic). Model identifiers, pricing constants and API parameters
are recorded in the scripts themselves.
