# Data

Source and attribution (unchanged): Md Saikot Hossain (2026), *Used Car Price Prediction Dataset*, Version 2, Mendeley Data, DOI [`10.17632/8d38h82fyt.2`](https://doi.org/10.17632/8d38h82fyt.2), CC BY 4.0.

The two files in this directory are derived from that source for the coursework analysis:

- `cleaned.csv` is a re-serialized analysis view of the same 5,997 listings. Changes are limited to parsing and normalization: category strings are NFKC/whitespace/case normalized, fuel labels are stored as sorted `|`-joined tokens, prices/mileage/capacity/age are parsed to integers, and `brand_model`, `row_id`, `source_line` and `group_id` are derived columns. Two suspect `150 cc` capacities and pure-electric capacities are set missing; one whitelisted `1000cc` text value is recovered. No listings are added or removed.
- `split_manifest.csv` is repository-added metadata: the fixed development/holdout assignment and ten development folds, derived from an attribute-group hash. `group_id` groups matching attributes and is not a verified vehicle identifier.

`cleaned.csv` SHA-256: 6f4791ce8763b6c73aa3af82be9b3d3cf5065e4cd901737211476ec5b4c5097f
`split_manifest.csv` SHA-256: eae5f790e40e356e2f17c7df4956cf85e02e6a5a7f049a2e40d8ab8ab468a179

5,997 listings; 4,789 development / 1,208 holdout; 5,608 attribute groups. The data are asking prices in BDT, not transaction prices. Holdout labels and test predictions are included for post-hoc inspection; this is not a blind benchmark. Attribute groups are not verified vehicle IDs.

See `NOTICE.md` for licence and modification details.
