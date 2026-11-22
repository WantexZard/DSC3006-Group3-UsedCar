# A — Data build, split, preprocessing, EDA

Code:

- `code/build_dataset.py`: builds `member-a-v1` from the official CSV; strict parsing, fixed attribute groups, fixed 4,789/1,208 split and ten folds.
- `code/shared_preprocessing.py`: fold-local imputation/encoding, loaders, feature mapping, unknown-category diagnostics.
- `code/eda.py`: training-only EDA tables/figures.

The shared data are at `../data/`. The split uses `group_id`, never random row splitting. Do not edit `group_id` or `cv_fold` when reproducing.
