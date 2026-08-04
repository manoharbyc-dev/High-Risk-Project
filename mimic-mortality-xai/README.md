# MIMIC-III Explainable & Fair ICU Mortality Prediction

A graduate-level project analyzing in-hospital mortality prediction from the first 24 hours of ICU admission using MIMIC-III data, with explainability (SHAP) and fairness/subgroup disparity analysis.

## Setup

### Prerequisites
- Google Cloud account with access to MIMIC-III on BigQuery (via PhysioNet DUA)
- Python 3.11 (installed via `brew install python@3.11` on macOS)
- Homebrew (macOS) for gcloud CLI installation

### Installation

1. **Set up Google Cloud access:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project <YOUR_PROJECT_ID>
   ```

2. **Create and activate virtual environment (using Python 3.11):**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Pinned Versions

The `requirements.txt` pins key ML library versions for reproducibility and compatibility:
- **pandas 2.0.3**, **numpy 1.24.4** — stable with proven wheel support
- **scikit-learn 1.3.2** — widely compatible
- **xgboost 2.0.3** — recent but not bleeding edge
- **matplotlib 3.8.2** — stable for plotting

**Deferred:** shap 0.44.0 (deferred due to numba/llvmlite compilation issues on macOS). Will install in Stage 2 once all cohort extraction is complete. SHAP is needed only for explainability analysis, not for data extraction.

These versions were chosen for wheel availability and stability on Python 3.11. If you update any of these, test model training and inference carefully.

## Project Structure

```
data/           Raw and processed MIMIC data (gitignored)
sql/            BigQuery cohort extraction SQL
src/            Python modules (data, features, models, explain, fairness)
notebooks/      Exploration (outputs stripped before commit)
figures/        Generated figures (aggregate-only, never patient-level)
results/        Summary metrics and tables (aggregate-only)
docs/           LIMITATIONS.md (design decisions), notes
requirements.txt   Pinned dependencies
```

## Running the Project

(To be filled in as stages complete.)

## Important: Data Privacy

This repository is public. **Never commit:**
- Patient-level data (MIMIC extracts, cohort CSVs, individual records)
- Notebooks with rendered data cells
- Credentials or service-account keys

Only aggregate summaries (counts, mean age, AUROC) may be committed.

---

See `docs/LIMITATIONS.md` for design decisions and known limitations.
