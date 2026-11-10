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

## Earlier exploratory pipeline (`run_pipeline.py`): minute-level forecasting

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
