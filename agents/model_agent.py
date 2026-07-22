from core.llm import call_llm, extract_text
from core.sandbox import Sandbox
import json


SYSTEM_PROMPT = """You are the Model Agent in an autonomous ML pipeline.

You receive:
- A description of the prepared dataset (features, target, shape)
- The task (classification/regression, metric to optimize)
- A hypothesis from the Orchestrator telling you exactly what to try

Your job is to write complete Python code that:
1. Loads cleaned_data.csv from the workspace
2. Splits into train/validation sets
3. Trains the model specified in the hypothesis
4. Evaluates on the validation set
5. Prints a structured result

Rules:
- Load data from cleaned_data.csv (always)
- Always print results in this exact format at the end:
    === RESULTS ===
    model: <model name>
    train_score: <float>
    val_score: <float>
    metric: <metric name>
    features_used: <comma separated list>
    notes: <any warnings, observations>
- Write complete runnable code, no placeholders
- Available libraries: pandas, numpy, scikit-learn, xgboost, lightgbm
- Do not plot anything
- Do not save the model, just train, evaluate, print results
"""


class ModelAgent:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.max_retries = 3

    def run(self, hypothesis: str, data_summary: dict, task: str) -> dict:
        print("\n[Model Agent] Writing training code...")
        print(f"[Model Agent] Hypothesis: {hypothesis}")

        messages = [
            {
                "role": "user",
                "content": f"""Task: {task}

Dataset summary:
{json.dumps(data_summary, indent=2)}

Hypothesis from Orchestrator:
{hypothesis}

Write complete Python code to implement this hypothesis.
Load from cleaned_data.csv, train, evaluate, print results in the required format."""
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens=4096)
        code = self._extract_code(extract_text(response))

        # it retries the loop
        for attempt in range(self.max_retries):
            print(f"[Model Agent] Running code - attempt {attempt + 1}...")
            result = self.sandbox.run(code, data_path="cleaned_data.csv")

            if result["success"]:
                print("[Model Agent] Success.")
                print(f"\n{result['stdout']}\n")
                return {
                    "success": True,
                    "code": code,
                    "raw_output": result["stdout"],
                    "metrics": self._parse_results(result["stdout"])
                }

            print("[Model Agent] Failed. Fixing...")
            code = self._fix_code(code, result["stderr"], hypothesis, task)

        # all retries exhausted
        return {
            "success": False,
            "code": code,
            "raw_output": "",
            "metrics": None,
            "error": result["stderr"]
        }

    def _fix_code(self, original_code: str, error: str, hypothesis: str, task: str) -> str:
        messages = [
            {
                "role": "user",
                "content": f"""This training code failed:

ERROR:
{error}

ORIGINAL CODE:
{original_code}

Hypothesis: {hypothesis}
Task: {task}

Fix the code and return the complete corrected version."""
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens=4096)
        return self._extract_code(extract_text(response))

    def _parse_results(self, stdout: str) -> dict:
        # finding the results block and parsing it
        if "=== RESULTS ===" not in stdout:
            return {"raw": stdout}

        results_block = stdout.split("=== RESULTS ===")[1].strip()
        metrics = {}

        for line in results_block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

        return metrics

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()
