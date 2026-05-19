# Crisis Dashboard: Dutch Media Coverage of Climate Change and Migration

A thesis research project analysing how Dutch national newspapers frame climate change and migration across a shared 2014–2023 corpus.

## Project overview

This repository contains the preprocessing and corpus-building pipeline for a parallel corpus of Dutch-language news articles on two crisis topics — climate change and migration — drawn from six national newspapers. The goal is to examine framing patterns and cross-crisis discourse across a decade of coverage.

**Outlets:** AD, FD, NRC, Telegraaf, Trouw, Volkskrant  
**Time window:** 1 January 2014 – 31 December 2023  
**Source:** Nexis Uni

## Repository structure

```
crisis-dashboard/
├── data/
│   ├── raw/                    # gitignored — source exports from Nexis Uni
│   └── processed/              # gitignored — derived clean and merged corpora
├── notebooks/                  # exploratory analysis (forthcoming)
├── src/
│   ├── preprocess_final.py     # full pipeline: raw CSVs → cleaned corpora
│   └── build_corpus.py         # merges cleaned corpora into a joint corpus
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

The raw data (`climate.csv`, `migration.csv`) are full-text article exports from [Nexis Uni](https://www.lexisnexis.com/en-us/products/nexis-uni.page) and are **not included in this repository** due to file size and licensing restrictions.

To reproduce the pipeline, obtain the exports from Nexis Uni and place them in `data/raw/` before running the scripts.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Running the pipeline

Run scripts from the project root (not from inside `src/`):

```bash
python src/preprocess_final.py   # step 1: raw → data/processed/climate_clean.csv
                                  #                 data/processed/migration_clean.csv
python src/build_corpus.py        # step 2: → data/processed/corpus_merged.csv
```

`preprocess_final.py` must complete before running `build_corpus.py`.

### What each script does

**`preprocess_final.py`** runs a 12-step pipeline on both raw corpora:
- Fixes Nexis Uni title-doubling artefacts
- Strips boilerplate metadata lines from article bodies
- Parses and clips dates to the shared 2014–2023 window
- Normalises outlet names to a consistent `outlet_clean` column
- Deduplicates migration articles (Online edition preferred over Print)
- Drops articles under 100 words
- Flags articles that appear in both corpora (`corpus_overlap`)

**`build_corpus.py`** merges the two cleaned files on their shared columns into a single `corpus_merged.csv`, reports schema alignment, and validates critical columns.

## Output files

| File | Description |
|------|-------------|
| `data/processed/climate_clean.csv` | Cleaned climate corpus |
| `data/processed/migration_clean.csv` | Cleaned migration corpus |
| `data/processed/corpus_merged.csv` | Joint corpus (shared columns only) |

All output files are gitignored due to size.