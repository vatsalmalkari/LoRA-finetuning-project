# Exported experiment results

Written by `src/export_results.py` (notebook section 9). Re-running the export overwrites this folder.

| Path | What it is |
|---|---|
| `runs/<run>.json` | one finished run: config, accuracy, params, memory, timing, latency (`lora_exp.run`) |
| `runs/<run>.predictions.json` | every test (or val) item: model response, strict/lenient correctness |
| `tables/summary_test.csv` | `lora_exp.summarize`: one row per configuration, mean and sd over seeds |
| `tables/summary_val.csv` | same for the learning-rate grid on the validation split |
| `tables/accuracy.csv` | accuracy by question type, MC split into rules/history, with 95% bootstrap CI |
| `tables/comparisons.csv` | paired comparisons: difference, 95% item-bootstrap CI, sign-flip permutation p |
| `tables/trend.json` | slope of accuracy on log10(trainable params), CI and p |
| `tables/stats_closeness.csv` | stats questions: share within 1/5/10%, median relative error |
| `tables/mc_baselines.csv` | chance / always-A / longest-answer baselines |
| `tables/subspace_<pair>_<side>.csv` | subspace similarity summaries (paper Section 7.2) |
| `figures/*.png` | figures saved by the notebook |

Accuracy is the *strict* metric unless a column says lenient (strict equals lenient for every run so far).
