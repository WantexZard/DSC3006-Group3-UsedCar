# Environment (per member, as recorded)

The project did not use one identical environment across all members. Do not claim a single pinned environment.

| Part | Python | numpy | pandas | scikit-learn | other |
|---|---:|---:|---:|---:|---|
| A data/EDA | 3.12.12 | 2.5.4 | 3.0.3 | 1.9.1 | scipy 1.18.1, matplotlib 3.10.9 |
| B linear | 3.12.11 | 1.26.4 | 2.2.3 | 1.4.0 | — |
| C forest | 3.12.8 | 2.5.4 | 3.1.0 | 1.9.1 | scipy 1.16.3, matplotlib 3.10.9 |
| D angle 5 | 3.12 | — | — | Historical version not recorded; rerun pin 1.9.1 (code needs ≥1.2) | see `D_angle5/requirements.txt` |
| E angle 4 | 3.12.14 | 2.3.5 | 2.2.3 | 1.8.0 | matplotlib 3.10.8 |

Canonical details: each part's `model_config.json` / `README.md`; E's `code/requirements.txt`. The frozen predictions are reused; this archive does not claim a single re-trained environment.
