# AMLAgent — Autonomous ML Pipeline Agent

An autonomous multi-agent system that takes a raw dataset and a task description, then runs the full ML pipeline: EDA, cleaning, feature engineering, model selection, training, evaluation in a self-directed loop until it converges on the best model. Every decision is reasoned, logged, and explained.

## What makes it different

Most automated ML tools treat model selection as a search problem, they try many things in parallel, return the best. AMLAgent treats it as a reasoning problem. Every iteration produces a written hypothesis about why the last attempt failed, a specific next action based on that diagnosis, and a log that builds into a full reasoning trace. The output isn't just the best model, it's an explanation of how it was found.


## What it does

You give it a CSV and one sentence:

```
"Predict customer churn. Target column: churned. Optimize for F1."
```

It gives you back the best model it found, a full reasoning trace of every experiment it ran, and a report explaining why the winning model works.

## How it works

Four agents. One loop.

```
Orchestrator
    │
    ├──▶ Data Agent         fetches data, EDA, cleaning, feature engineering
    │         │
    │         └──▶ Python sandbox (Docker)
    │
    ├──▶ Model Agent        writes training code, runs it, returns metrics
    │         │
    │         └──▶ Python sandbox (Docker)
    │
    └──▶ Eval Agent         diagnoses results, forms hypothesis, recommends next action
              │
              └──▶ Experiment memory (JSON log)
```

The orchestrator runs the loop. It calls Data Agent once at the start, then cycles Model Agent → Eval Agent → Model Agent → Eval Agent until the stop condition is met.


## The reasoning loop

What separates this from AutoGluon or H2O is that every iteration produces a hypothesis, a written reason for why the last experiment failed and what the next one should try. This trace is preserved.

```json
{
  "iteration": 3,
  "hypothesis": "Val loss 0.91 vs train loss 0.23 is severe overfitting. Features age_income_interaction and zipcode_encoded are likely noise. Dropping them and adding L2 regularization.",
  "model": "LogisticRegression(C=0.1)",
  "features_used": ["age", "income", "tenure_months"],
  "train_f1": 0.84,
  "val_f1": 0.79,
  "gap": 0.05,
  "diagnosis": "Overfitting reduced from gap=0.31 to gap=0.05. Val F1 improved from 0.61 to 0.79. Continuing this direction — try XGBoost with same reduced feature set.",
  "next_action": "XGBoost with max_depth=4, learning_rate=0.05, same features"
}
```

The final report contains every iteration's reasoning trace, not just the final model.


## Architecture

### Orchestrator

Holds experiment memory. Never writes code. Decides what to do next by reading the full history of what was tried and what each attempt taught. Calls the other agents as tools.

Knows when to stop:
- Val score has not improved for 3 consecutive iterations
- Val score crossed the target threshold you specified
- Max iterations reached (default: 15)

### Data Agent

Single responsibility: understand and prepare the data.

- Writes Python to load, profile, and clean the dataset
- Detects column types (categorical, numerical, datetime, free text)
- Handles missing values with reasoned strategy per column (not just fill with mean)
- Engineers features — interaction terms, log transforms, encodings — based on what the data actually looks like, not a fixed recipe
- Reports back a structured summary: shape, dtypes, missing value treatment, features created, final feature matrix description

Runs inside Docker. No network access. 5 minute timeout.

### Model Agent

Single responsibility: implement and train what the orchestrator specifies.

- Receives: feature matrix description, task type, model hypothesis from orchestrator
- Writes complete training code from scratch each iteration (no templates)
- Runs it in Docker sandbox
- Reads stdout: loss curves, sklearn report, warnings
- Returns: train/val metrics, any runtime errors, first 50 lines of training output

Does not decide what model to try. That's the orchestrator's job.

### Eval Agent

Single responsibility: make sense of what just happened.

- Reads current metrics + full experiment history
- Diagnoses the failure mode (overfitting, underfitting, wrong metric, data leakage, class imbalance, wrong feature set)
- Each diagnosis maps to a set of possible next actions
- Recommends one specific next action with written reasoning
- Flags if the agent appears to be in a loop (same experiment repeated)

This is the agent with the most carefully engineered prompt. The quality of this agent's reasoning determines the quality of the whole system.


## Experiment memory

A single JSON file. Append-only during a run. Every iteration adds one entry. The orchestrator reads the full file before each decision. This is the agent's working memory — without it the loop has no sense of history.

```
experiments/
└── run_2027_07_19_churn/
    ├── memory.json          full iteration log
    ├── data_summary.json    output from Data Agent
    ├── best_model.pkl       serialized winner
    ├── best_code.py         training code that produced the winner
    └── report.md            final human-readable report
```


## Sandboxed execution

Every Python file the agents write runs inside a Docker container:

```
memory:   4GB max
cpu:      2 cores
network:  none
timeout:  300 seconds
disk:     read-only except /tmp
```

The container is torn down after each run. No state leaks between iterations. If the agent writes code that causes an OOM or infinite loop, the container dies, the error message goes back to the orchestrator, and the loop continues.

This is not optional. Agents write code you have not reviewed. Run it in a box.


## Stack

- LLM API
- Agent Loop (Raw Python)
- Docker, Subprocess
- JSON for experiment memory
- ML Libraries (sklearn, xgboost, lightbm, pandas, numpy and more)
- API layer (FastAPI, Flask if needed)
- Streamlit for Live view of current iteration, hypothesis, score chart


## Eval harness

The agent is evaluated on a benchmark of 10 Kaggle tabular datasets with known leaderboard scores. For each dataset:

- Run the agent with max 15 iterations
- Record best val score achieved
- Compare to: dummy baseline, a single XGBoost with default params, top public Kaggle kernel

| Dataset | Dummy baseline | Default XGBoost | AMLAgent | Kaggle top 10% |
|---|---|---|---|---|
| Titanic | 0.62 | 0.81 | — | 0.84 |
| House Prices | 0.00 RMSE | 0.14 RMSE | — | 0.11 RMSE |
| Credit fraud | 0.50 F1 | 0.86 F1 | — | 0.89 F1 |

*(scores filled in as benchmark runs complete)*

The agent is succeeding if it consistently beats default XGBoost. It is doing something interesting if its reasoning traces identify the correct failure mode more than 70% of the time, regardless of whether the fix worked.


## Output

A run produces three things:

**1. Best model** - serialized, ready to load and run inference

**2. Reasoning trace** - the full iteration log in human-readable form. What was tried, why, what was learned. This is what you show people.

**3. Final report** - structured markdown:
- Dataset summary (shape, types, key statistics)
- Feature engineering decisions and rationale
- Experiments run (table: iteration, model, hypothesis, val score)
- Winner analysis (why this model, what features mattered, confidence)
- Failure analysis (what didn't work and why)

## What this is not

- Not a Jupyter notebook wrapper. The agent writes fresh code every iteration based on its current hypothesis.
- Not a chatbot that helps you with ML. You don't talk to it. It runs autonomously.
- Not a fixed pipeline. There is no predetermined sequence of steps. The orchestrator decides what to do next based on what it learned from the last iteration.


## Research direction

The hypothesis-diagnosis loop is the novel contribution. The open questions:

- How accurately does the Eval Agent identify the true failure mode? (Requires labeled failure mode dataset to measure)
- Does reasoning about failure modes lead to faster convergence than blind iteration? (Measure iterations-to-target across benchmark datasets)
- Can the agent generalize hypotheses across datasets? (Does "log-transform skewed features" learned on Dataset A transfer to Dataset B?)
- Does the reasoning trace produced by the Eval Agent match the diagnosis a human ML engineer would make? (Human study on 20 failure cases)

If the answer to the second question is yes and the margin is significant, this is publishable.


## Status

- [ ] Docker sandbox setup
- [ ] Data Agent (EDA + cleaning + feature engineering)
- [ ] Model Agent (code generation + execution)  
- [ ] Eval Agent (diagnosis + hypothesis)
- [ ] Orchestrator loop
- [ ] Experiment memory schema
- [ ] FastAPI endpoint
- [ ] Streamlit dashboard
- [ ] Benchmark evaluation
- [ ] Final report generation


## Author

MIT  
[Ganesh Kumar](https://byganesh.com)
