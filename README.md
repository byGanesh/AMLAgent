# AMLAgent - Autonomous ML Pipeline Agent

An autonomous ML agent that takes a dataset and a task description, then runs the full pipeline: data preparation, model training, evaluation, and iterative improvement until it finds the best model.

## What makes it different

Most automated ML tools treat model selection as a search problem, they try many things in parallel, return the best. AMLAgent treats it as a reasoning problem. Every iteration produces a written hypothesis about why the last attempt failed, a specific next action based on that diagnosis, and a log that builds into a full reasoning trace. The output isn't just the best model, it's an explanation of how it was found.

## Quick start

```bash
# Build the sandbox image (one-time)
docker build -t amlagent-sandbox sandbox/

# Install deps
pip install -r requirements.txt

# Set up .env (copy from .env.example or edit directly)
#   LLM_API_KEY=your_key
#   LLM_BASE_URL=https://api.
#   LLM_MODEL=

# Drop your CSV into workspace/
# Run
python main.py
```

## Usage

```
You > I have heart.csv, Predict whether a patient has heart disease.
```

The agent will:
1. Analyze and clean the data, engineer features
2. Train a baseline model
3. Diagnose results and try better models
4. Report the best model with code and metrics

## Project structure

```
├── main.py              # CLI entry point
├── src/
│   ├── pipeline.py      # Pipeline orchestration (data → model loop → report)
│   ├── sandbox.py       # Docker sandbox for safe code execution
│   └── llm.py           # LLM API wrapper
├── sandbox/
│   └── Dockerfile       # Python + ML libraries for sandbox
├── workspace/           # Put your datasets here
└── runs/                # Experiment results saved here
```

## How it works

A single `Pipeline` class drives three LLM-powered steps in a loop:

1. **Data prep** — LLM writes Python to load, clean, and feature-engineer the dataset, saves `cleaned_data.csv`.
2. **Model training** — LLM writes model training/eval code based on a hypothesis, runs it, returns metrics.
3. **Evaluation** — LLM analyzes results, diagnoses failure modes, suggests the next model to try.

Steps 2–3 repeat until the model stops improving or max iterations are reached.

## Sandbox

All generated code runs in an isolated Docker container:
- No network access
- 4GB RAM / 2 CPU limit
- Container destroyed after each run

## Requirements

- Docker
- Python 3.10+
- An LLM API key (Groq, OpenAI, etc.)

## Author

MIT  
[Ganesh Kumar](https://byganesh.com)
