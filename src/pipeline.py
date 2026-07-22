import json
from datetime import datetime
from pathlib import Path
from src.sandbox import Sandbox
from src.llm import call_llm, extract_text


DATA_SYSTEM_PROMPT = """You write Python code for data preparation.
Given a dataset path and task, write code that:
1. Loads the dataset
2. Prints shape, column dtypes, missing values, basic statistics
3. Handles missing values with a reasoned strategy per column
4. Engineers relevant new features based on the task
5. Saves the cleaned dataset as cleaned_data.csv in the current directory
6. Prints a JSON summary at the end with keys: rows, columns, features, target_column, dropped_features, notes

Rules:
- Write complete runnable code, no placeholders
- Use pandas, numpy. No plotting.
- The file path is relative to the current directory."""

MODEL_SYSTEM_PROMPT = """You write Python code for ML model training.
Given a dataset summary and a hypothesis, write code that:
1. Loads cleaned_data.csv from the current directory
2. Splits into train/validation sets
3. Trains the model specified in the hypothesis
4. Evaluates on the validation set
5. Prints a JSON result at the end with keys: model, train_score, val_score, metric, features_used, notes

Rules:
- Write complete runnable code, no placeholders
- Available: pandas, numpy, scikit-learn, xgboost, lightgbm
- Do not save any files, just print the JSON result
- The validation metric should match the task (accuracy/F1/RMSE etc.)"""

EVAL_SYSTEM_PROMPT = """You are an ML experiment evaluator.
Given the experiment history, analyze the results and decide what to try next.

Return a JSON object with these exact keys:
{
    "diagnosis": "what happened",
    "failure_mode": "overfitting|underfitting|wrong_features|convergence_failure|data_leakage|class_imbalance|null",
    "hypothesis": "exactly what model to try next with specific params",
    "should_stop": true/false,
    "stop_reason": "why stopping or null",
    "confidence": 0.0-1.0
}

Stop if: val_score has not improved for 3 iterations, target threshold crossed, or data leakage detected."""


class Pipeline:

    def __init__(self, workspace: str = "workspace", max_iterations: int = 10):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(exist_ok=True)
        self.max_iterations = max_iterations
        self.sandbox = Sandbox()

    def run_data_step(self, file_name: str, task: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"Dataset: {file_name}\nTask: {task}\nWrite Python code to prepare this data."
            }
        ]
        response = call_llm(DATA_SYSTEM_PROMPT, messages, max_tokens=4096)
        code = extract_text(response)
        code = self._extract_code(code)

        result = self.sandbox.run(code, str(self.workspace))
        if not result["success"]:
            messages.append({"role": "assistant", "content": code})
            messages.append({
                "role": "user",
                "content": f"This code failed:\n{result['stderr']}\nFix it and return the complete corrected version."
            })
            response = call_llm(DATA_SYSTEM_PROMPT, messages, max_tokens=4096)
            code = self._extract_code(extract_text(response))
            result = self.sandbox.run(code, str(self.workspace))

        if not result["success"]:
            return {"success": False, "error": result["stderr"]}

        summary = self._parse_json(result["stdout"])
        return {"success": True, "summary": summary, "code": code, "raw_output": result["stdout"]}

    def run_model_step(self, hypothesis: str, task: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"Task: {task}\nHypothesis: {hypothesis}\nWrite Python code to train and evaluate this model."
            }
        ]
        response = call_llm(MODEL_SYSTEM_PROMPT, messages, max_tokens=4096)
        code = self._extract_code(extract_text(response))

        for attempt in range(3):
            result = self.sandbox.run(code, str(self.workspace))
            if result["success"]:
                metrics = self._parse_json(result["stdout"])
                return {"success": True, "code": code, "metrics": metrics, "raw_output": result["stdout"]}
            messages.append({"role": "assistant", "content": code})
            messages.append({
                "role": "user",
                "content": f"Code failed:\n{result['stderr']}\nFix it."
            })
            response = call_llm(MODEL_SYSTEM_PROMPT, messages, max_tokens=4096)
            code = self._extract_code(extract_text(response))

        return {"success": False, "code": code, "error": result["stderr"]}

    def evaluate(self, history: list, task: str, iteration: int) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"Task: {task}\nIteration: {iteration}\nHistory:\n{json.dumps(history, indent=2)}\nAnalyze and return JSON."
            }
        ]
        response = call_llm(EVAL_SYSTEM_PROMPT, messages, max_tokens=1024)
        raw = extract_text(response).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "diagnosis": "Parse error", "failure_mode": None,
                "hypothesis": "Try RandomForestClassifier with n_estimators=100",
                "should_stop": False, "stop_reason": None, "confidence": 0.3
            }

    def run(self, file_name: str, task: str) -> dict:
        run_id = f"{Path(file_name).stem}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = Path("runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRun ID: {run_id}")
        print(f"Task: {task}")
        print(f"Max iterations: {self.max_iterations}\n")

        data_result = self.run_data_step(file_name, task)
        if not data_result["success"]:
            print(f"[Error] Data prep failed: {data_result['error']}")
            return {"success": False, "error": data_result["error"]}

        print("[Data] Done.\n")

        hypothesis = f"Start with LogisticRegression baseline. Target column: {data_result['summary'].get('target_column', 'unknown')}."
        history = []
        best = {"val_score": -1, "result": None}
        no_improve = 0

        for i in range(1, self.max_iterations + 1):
            print(f"{'─' * 50}")
            print(f"Iteration {i}/{self.max_iterations}")
            print(f"{'─' * 50}")
            print(f"Hypothesis: {hypothesis}")

            model_result = self.run_model_step(hypothesis, task)

            if model_result["success"]:
                metrics = model_result["metrics"]
                val_score = metrics.get("val_score", -1)
                if isinstance(val_score, (int, float)) and val_score > best["val_score"]:
                    best = {"val_score": val_score, "result": model_result}
                    no_improve = 0
                    (run_dir / "best_code.py").write_text(model_result["code"])
                    (run_dir / "best_output.txt").write_text(model_result["raw_output"])
                else:
                    no_improve += 1
                entry = {"iteration": i, "success": True, **metrics}
            else:
                print(f"[Model] Failed: {model_result['error']}")
                entry = {"iteration": i, "success": False, "error": model_result["error"]}

            history.append(entry)

            decision = self.evaluate(history, task, i)
            print(f"[Eval] {decision['diagnosis']}")
            if decision.get("failure_mode"):
                print(f"[Eval] Failure: {decision['failure_mode']}")

            if decision.get("should_stop"):
                print(f"[Eval] Stopping. {decision.get('stop_reason', '')}")
                break

            if no_improve >= 3:
                print("[Pipeline] No improvement for 3 iterations. Stopping.")
                break

            hypothesis = decision.get("hypothesis", hypothesis)

        self._print_report(best, history, run_id)
        return {
            "success": True, "run_id": run_id,
            "best_val_score": best["val_score"],
            "iterations": len(history)
        }

    def _print_report(self, best: dict, history: list, run_id: str):
        print(f"\n{'═' * 50}")
        print("RESULTS")
        print(f"{'═' * 50}")
        if best["result"]:
            m = best["result"]["metrics"]
            print(f"Best model:    {m.get('model', 'unknown')}")
            print(f"Best val score: {best['val_score']}")
            print(f"Metric:        {m.get('metric', 'unknown')}")
        else:
            print("No successful model.")
        print(f"Iterations:    {len(history)}")
        print(f"Saved to:      runs/{run_id}/")
        print(f"{'═' * 50}\n")

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    def _parse_json(self, text: str) -> dict:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}
