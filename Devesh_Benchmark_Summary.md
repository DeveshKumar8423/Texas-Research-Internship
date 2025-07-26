# Benchmarking Summary (Motion Code - Parkinson Dataset)

## Datasets Evaluated:
- PD Setting 1
- PD Setting 2

## Commands Used:
```bash
python parkinson_data_processing.py
python motion_code.py --dataset parkinson_pd1
python motion_code.py --dataset parkinson_pd2

Results:

Dataset	Accuracy (%)
PD Setting 1	71.12
PD Setting 2	54.31

---

## Benchmark comparison of Time Series classification models across multiple datasets:

```bash
python collect_all_benchmarks.py
