# surveycast

Calibrated answer distributions for survey respondents nobody has interviewed yet. Entry for SimulacraBench (NeurIPS 2026 competition track).

For each held-out respondent the model sees only the demographic (GIVEN) block and returns a probability distribution over every hidden answer. The metric is a log score normalised so that the crowd marginal scores 0 and a perfect forecast scores 1, averaged over the UNHCR, UNICEF and World Bank instruments.

## What it does

- Fits everything inside `predict()` on the visible rows it receives. The microdata are never released, so nothing is trained offline and the archive carries no weights.
- Stacks three model families: Dirichlet-smoothed backoff tables, multi-task softmax regression and a multi-task MLP (five seeds) with per-item early stopping.
- Combines them with per-item linear-pool weights fitted by EM on out-of-fold predictions.
- Applies the instrument's skip logic: exact gates when the parent is a GIVEN item, averaged gate recursion when the parent is itself predicted.
- Splits the run-time budget over the three calls and drops components that would overrun, so the run always finishes inside the limit.

## Results

Exact (unnoised) skill on the phase-2 split: TRAIN and DEV visible, TEST scored. Computed with the starter kit's own scoring functions.

| dataset | skill |
|---|---|
| official sandbox, UNICEF | 0.3958 |
| official sandbox, World Bank | 0.2888 |
| official sandbox, UNHCR | 0.2312 |
| Afrobarometer R6 / R8 / R9 (proxy) | 0.0950 / 0.1799 / 0.1056 |
| Argentina EPH household / person (proxy) | 0.3436 / 0.3860 |

The sandboxes have invented dependencies, so model choices were made on the mean over the sandboxes and five proxy datasets built in the kit's format from real public microdata (`proxy/`). The MLP is the best single component; the stack beats it on all eight datasets. Training is stochastic on CPU threads, so expect agreement to about the third decimal.

## Try it

Python 3.11 or newer. The scripts expect the official starter kit in `./kit` (or set `KIT=/path/to/kit`).

```bash
git clone https://github.com/SituatedEvals/public kit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# generate the three official sandboxes and score the model on each (phase 2)
scripts/score_sandboxes.sh
```

Each line prints a JSON record with `skill`, `std_error`, `log_score` and `seconds`. Per-item results go to `work/results/`. `PHASE=1 scripts/score_sandboxes.sh` scores the development split instead (TRAIN visible, DEV scored).

`--no-deadline` lifts the run-time guard, so the full model runs even on a slow CPU. Without it the guard targets the hosted GPU budget and may drop the neural components on a CPU. The full phase-2 run takes roughly 5 to 15 minutes per instrument on 8 idle CPU threads.

Score the zip itself with the kit's official scorer (it builds its own environment per run):

```bash
scripts/package.sh                      # builds and validates surveycast.zip
mkdir -p work/unzipped && python -m zipfile -e surveycast.zip work/unzipped
(cd kit && python score.py --submission ../work/unzipped --data ../work/sbx/unicef \
   --schema data/unicef.json --phase 1)
```

### Proxy datasets

```bash
python proxy/build_proxies.py   # downloads public microdata from Hugging Face into work/proxy_raw
python proxy/subsample.py       # cuts the three large proxies to 12,000 rows
scripts/score_proxies.sh
```

Sources: Afrobarometer rounds 6, 8 and 9 (CC-BY-4.0 repackaging on Hugging Face, `electricsheepafrica/africa-afrobarometer-<country>`) and INDEC's Encuesta Permanente de Hogares (`Matiasdn/eph-microdata-parquet`). The builder infers skip gates from the data and writes `respondents.parquet` and `schema.json` per dataset.

## How it works

```
visible rows ──► encode GIVEN items (one level per option + unseen)
                 │
                 ├─ backoff tables     (leave-one-out predictions)
                 ├─ softmax regression ┐
                 └─ MLP x 5 seeds      ┘ 5-fold cross-fitting
                 │
                 ▼
     per-item EM linear pool on out-of-fold predictions
                 │
                 ▼
     skip-logic gates  ──►  distribution per (respondent, item)
```

Backoff tables rank GIVEN items by mutual information with each answer and shrink each cell toward its parent cell, with the Dirichlet strength chosen by leave-one-out likelihood. The softmax regression and the MLP share one padded multi-task output (items by max options, masked logits) trained on the mean cross-entropy per cell, which matches the metric's equal weight per cell. Each MLP seed uses its own fold split. Pooling weights per item get a Dirichlet prior toward the instrument-wide weights, so items with few rows fall back to the global mix.

Things that did not help on local validation: LightGBM as a fourth component, backoff averaged over several feature orders, 10 folds instead of 5, per-item temperature scaling, and one global set of pooling weights instead of per-item weights.

## Layout

```
surveycast/              the submission: main.py, requirements.txt, LICENSE
scripts/evaluate.py      in-process scorer using the kit's score.py functions
scripts/score_*.sh       score sandboxes or proxies
scripts/package.sh       build and validate the upload archive
proxy/                   build proxy datasets from public surveys
```

## Built with

Python, NumPy, pandas, PyTorch. Scoring and sandboxes from the SimulacraBench starter kit.

## License

MIT, see `LICENSE`.
