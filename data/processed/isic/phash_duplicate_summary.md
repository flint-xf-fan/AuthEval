# ISIC perceptual duplicate report

This report uses 64-bit DCT pHash perceptual hashes over existing image files. It does not copy raw images or alter splits.

- Algorithm: phash_dct
- Rows scanned: 25331
- Images hashed: 25331
- Hash failures: 0
- Hamming threshold: 4
- Transform size: 32
- Exact hash duplicate groups: 95
- Exact hash duplicate images: 191
- Exact hash cross-split groups: 0
- Exact hash cross-split images: 0
- Near-duplicate components: 594
- Near-duplicate images: 1777
- Cross-split near-duplicate components: 122
- Component report: `data/processed/isic/phash_duplicate_report.csv`
- All exact duplicate components: `data/processed/isic/phash_exact_duplicate_components.csv`
- Exact cross-split report: `data/processed/isic/phash_final_exact_cross_split_duplicates.csv`
- Failure report: `data/processed/isic/phash_duplicate_failures.csv`

- Split metadata source: `data/processed/isic/all_clean.csv`

Exact pHash cross-split groups are zero. Cross-split near-duplicate components remain diagnostic at the configured Hamming threshold.
