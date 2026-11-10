# Cold-Start Prediction & Provisioning Optimization (Azure Functions 2019)

Replication of *"Cold Start Prediction and Provisioning Optimization in Serverless Computing Using
Deep Learning"* (ODL-CSP), on the public Azure Functions 2019 trace instead of the paper's AWS Lambda data.

## Main pipeline: paper replication (`run_paper.py`)

Built from the paper's abstract (the methods section is still to be cross-checked):

| Paper | This implementation |
|---|---|
| Attention-based BiGRU (ABiGRU) | `coldstart/abigru.py`: BiGRU + additive attention pooling |
| RMSProp + hyperparameter tuning | `torch.optim.RMSprop`; `--tune` grid over lr x hidden |
| Predict cold start of each incoming invocation | per-invocation dataset (`coldstart/events.py`); label = cold under a 10-min keep-alive; **only past invocations are used** |
| Features: timestamp, invoke frequency, cold-start indicator, duration, previous/historical cold starts, event type, time since last cold start, consecutive cold starts | all of them, per event, over the last 16 invocations |
| Accuracy / Precision / Recall / F1 / MCC (90.36 / 91.87 / 90.42 / 90.28 / 82.28) | same metrics, on the natural test distribution **and** a balanced test subset |
| Provisioned concurrency ("function warmer") | warmer keeps the function provisioned 60 min instead of 10 when a cold start is predicted |

Baselines: BiGRU (no attention), GRU, LSTM, logistic regression, gradient boosting, "previous call was cold".

```bash
python run_paper.py --tune --epochs 5      # outputs/paper/{classification_test,warmer_test}.csv, report.json
```

Differences from the paper to keep in mind:
- Azure trace has per-minute counts, so an "invocation" is an active minute; cold starts are derived
  from the keep-alive rule, not measured.
- Only ~4.5% of invocations are cold in this trace, so accuracy is high by default. MCC/F1 and the
  balanced-subset numbers are the fair comparison with the paper.

## Results

1,000 random Azure functions, 10 days, seed 42. Ground-truth cold start = idle > 10 min.

### A. Paper protocol (`run_paper_protocol.py`)
Reconstructed from the paper's confusion matrix ([[1242, 0], [241, 1017]] reproduces its five
numbers exactly): 10,000 balanced invocations, stratified 75/25 split (7,500 / 2,500), macro
metrics, RMSProp (lr 1e-3, hidden 64, batch 64), 100 epochs, best-validation-loss checkpoint.

| Model (test) | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| **ABiGRU (ours)** | **95.84** | **95.92** | **95.84** | **95.84** | **91.76** |
| BiGRU | 96.16 | 96.17 | 96.16 | 96.16 | 92.33 |
| GRU | 95.92 | 95.97 | 95.92 | 95.92 | 91.89 |
| LSTM | 96.00 | 96.03 | 96.00 | 96.00 | 92.03 |
| KNN | 95.56 | 95.57 | 95.56 | 95.56 | 91.13 |
| Logistic regression | 96.20 | 96.22 | 96.20 | 96.20 | 92.42 |
| Gradient boosting | 96.00 | 96.03 | 96.00 | 96.00 | 92.03 |
| ODL-CSP (paper, AWS data) | 90.36 | 91.87 | 90.42 | 90.28 | 82.28 |

ABiGRU test confusion matrix (actual Cold/Warm x predicted Cold/Warm): [[1224, 26], [78, 1172]].

![comparison](docs/figures/comparison.png)
![confusion](docs/figures/cm_test.png) ![pr-roc](docs/figures/pr_roc_test.png)
![curves](docs/figures/train_curves.png)

**Overfitting:** with only 6,750 training samples, validation loss is lowest around epochs 5-15
and rises afterwards; the reported model is the best-validation checkpoint (the paper's own
validation curve is similarly unstable). Early stopping is recommended.

### B. Realistic time-split protocol (`run_paper.py --tune`)
Train days 2-7 (400k invocations), validate day 8, test days 9-10 (395k invocations, natural
4.5% cold-start rate). RMSProp grid search picked lr 1e-3, hidden 64. Positive class = cold.

| Model | Acc | Prec | Rec | F1 | MCC | AUC | Bal. Acc* | Bal. MCC* |
|---|---|---|---|---|---|---|---|---|
| **ABiGRU** | 98.47 | 85.65 | 78.84 | **82.11** | 81.38 | 99.32 | **89.10** | **79.91** |
| BiGRU | 98.48 | 90.86 | 73.30 | 81.14 | 80.86 | 99.32 | 86.46 | 75.59 |
| GRU | 98.49 | 89.74 | 74.78 | 81.58 | 81.17 | 99.32 | 87.17 | 76.72 |
| LSTM | 98.51 | 88.52 | 76.52 | 82.09 | **81.55** | 99.34 | 88.01 | 78.10 |
| Logistic regression | 98.43 | 86.34 | 77.03 | 81.42 | 80.75 | 99.26 | 88.22 | 78.42 |
| Gradient boosting | 98.50 | 87.86 | 77.09 | **82.12** | 81.53 | 99.31 | 88.30 | 78.61 |
| "previous call was cold" | 97.69 | 74.10 | 74.05 | 74.08 | 72.86 | 86.42 | 86.39 | 75.10 |

\* balanced subset of the test set (all cold + equal random warm), threshold 0.5.

**Function warmer** (provision 60 min instead of 10 when a cold start is predicted, test days):

| Policy | Cold starts | Cold-start rate | Reduction | Warm memory (GB-h) |
|---|---|---|---|---|
| Keep-alive 10 min (no warmer) | 17,652 | 4.46% | - | 1,641 |
| **ABiGRU warmer** | **5,435** | **1.37%** | **69.2%** | 2,408 |
| LSTM warmer | 5,701 | 1.44% | 67.7% | 2,378 |
| Gradient-boosting warmer | 5,663 | 1.43% | 67.9% | 2,387 |
| Keep-alive 60 min for every function | 2,389 | 0.60% | 86.5% | 2,601 |
| Oracle warmer (upper bound) | 2,389 | 0.60% | 86.5% | 2,286 |

![warm-rate](docs/figures/warm_rate.png)

### Takeaways
- Under the paper's protocol we exceed its reported numbers (95.8 vs 90.4 accuracy, 91.8 vs 82.3 MCC),
  but every model, including logistic regression, lands within ~1 point: the paper's features
  (cold-start history, idle gap) make the balanced task easy.
- Under the realistic protocol ABiGRU gives the best warmer (69.2% fewer cold starts) and the best
  balanced accuracy/MCC, but its margin over LSTM and gradient boosting is small (1-2 points).
- The paper's latency figure is not reproduced: the Azure trace has no cold-start latency.

All tables: `docs/results/*.csv`.

## Earlier exploratory pipeline (`run_pipeline.py`): minute-level forecasting

Result (test days, thresholds picked on validation within the 10-min keep-alive memory budget):
LSTM cuts the cold-start rate from 4.46% to 0.60%, gradient boosting to 0.49%, and the
hybrid-histogram policy to 0.52% (`docs/results/minute_level_summary.csv`,
`docs/figures/minute_level_tradeoff.png`).

## Finalized idea

1. **Prediction (deep learning).** For every function *f* and minute *t*, predict
   `P(f is invoked at minute t | history up to t-1)` with an LSTM.
   - Sequence input (last 60 min): `log1p(invocations)`, sin/cos time-of-day.
   - Static input: minutes since last invocation, invocations at the same time yesterday,
     24 h and 60 min activity fraction, trigger type (http/timer/queue/...).
2. **Provisioning optimization.** Turn the predictions into a keep-warm decision per minute:
   keep a container warm when `p >= threshold` (plus 1 minute after use). The threshold is
   **chosen on the validation day** to minimize cold starts subject to a memory budget equal to
   Azure's default fixed 10-minute keep-alive.
3. **Evaluation (minute-level simulator).**
   - Cold start = invoked in a minute with no warm container.
   - Cost = warm-but-idle minutes x app memory (MB), reported as GB-hours.
   - Baselines: fixed keep-alive (1/5/10/20/60 min), hybrid histogram (Shahrad et al., ATC'20),
     gradient-boosting predictor (classical ML vs DL), and the oracle.

## Data
`C:\Users\Durga\Downloads\azurefunctions_dataset2019_azurefunctions-dataset2019`
- 1,000 functions sampled at random from day 1 (seed 42); days 1-10 loaded.
- Day 1 = history only, days 2-7 = train, day 8 = validation (threshold selection), days 9-10 = test.
- App memory (`app_memory_percentiles d01`) weights the wasted-memory cost.

## Run
```bash
pip install torch pandas numpy scikit-learn matplotlib
python run_pipeline.py                 # full run (1000 functions)
python run_pipeline.py --functions 200 --epochs 1   # quick smoke test
```
Outputs in `outputs/`: `report.json`, `summary_test.csv`, `policies_test.csv` (all thresholds),
`tradeoff_test.png` (cold-start rate vs wasted memory), `lstm.pt`.

## Layout
- `coldstart/data.py`: trace -> (functions x minutes) matrix, cached in `cache/`
- `coldstart/features.py`: leak-free features (only minutes < t)
- `coldstart/models.py`: LSTM predictor + training loop
- `coldstart/policies.py`: keep-alive, hybrid-histogram, predictive policies + scoring
- `run_pipeline.py`: end to end
