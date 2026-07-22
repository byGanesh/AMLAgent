from agents.data_agent import DataAgent
from agents.model_agent import ModelAgent
from agents.eval_agent import EvalAgent
from core.memory import ExperimentMemory
from core.sandbox import Sandbox
from pathlib import Path
import json


class Orchestrator:
    def __init__(self, workspace: str = "workspace", max_iterations: int = 15):
        self.workspace = Path(workspace)
        self.max_iterations = max_iterations
        self.sandbox = Sandbox()

        self.data_agent = DataAgent(self.sandbox)
        self.model_agent = ModelAgent(self.sandbox)
        self.eval_agent = EvalAgent()

    def run(self, file_name: str, task: str) -> dict:
        file_path = self.workspace / file_name
        run_id = self._make_run_id(file_name)
        run_dir = Path("runs") / run_id
        memory = ExperimentMemory(str(run_dir))

        print(f"\nRun ID: {run_id}")
        print(f"Task: {task}")
        print(f"File: {file_path}")
        print(f"Max iterations: {self.max_iterations}\n")

        # phase 1: data
        data_result = self.data_agent.run(str(file_path), task)

        if not data_result["success"]:
            print("\n[Orchestrator] Data Agent failed. Aborting.")
            print(f"Error: {data_result['error']}")
            return {"success": False, "error": data_result["error"]}

        data_summary = data_result["summary"]

        # saving data summary
        with open(run_dir / "data_summary.json", "w") as f:
            json.dump(data_summary, f, indent=2)

        # phase 2: model iteration loop
        # first hypothesis, always start simple
        hypothesis = self._first_hypothesis(task, data_summary)
        best_result = None
        best_val_score = -1
        no_improvement_count = 0

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─' * 50}")
            print(f"Iteration {iteration}/{self.max_iterations}")
            print(f"{'─' * 50}")

            # model agent trains
            model_result = self.model_agent.run(hypothesis, data_summary, task)

            if not model_result["success"]:
                print(f"[Orchestrator] Model Agent failed on iteration {iteration}.")
                memory.log({
                    "hypothesis": hypothesis,
                    "success": False,
                    "error": model_result["error"]
                })

                decision = self.eval_agent.run(
                    {"error": model_result["error"]},
                    memory.to_string(),
                    task,
                    iteration
                )
            else:
                metrics = model_result["metrics"]

                # tracking best
                val_score = metrics.get("val_score", -1)
                if isinstance(val_score, float) and val_score > best_val_score:
                    best_val_score = val_score
                    best_result = model_result
                    no_improvement_count = 0
                    self._save_best(run_dir, model_result)
                else:
                    no_improvement_count += 1

                # eval agent diagnoses
                decision = self.eval_agent.run(
                    metrics,
                    memory.to_string(),
                    task,
                    iteration
                )

                # log everything to memory
                memory.log({
                    "hypothesis": hypothesis,
                    "model": metrics.get("model", "unknown"),
                    "train_score": metrics.get("train_score"),
                    "val_score": metrics.get("val_score"),
                    "metric": metrics.get("metric"),
                    "diagnosis": decision["diagnosis"],
                    "failure_mode": decision["failure_mode"],
                    "confidence": decision["confidence"],
                    "next_hypothesis": decision["hypothesis"]
                })

            # it checks stop conditions
            if decision["should_stop"]:
                print(f"\n[Orchestrator] {decision['stop_reason']}")
                break

            if no_improvement_count >= 3:
                print("\n[Orchestrator] No improvement for 3 iterations. Stopping.")
                break

            # next hypothesis comes from eval agent
            hypothesis = decision["hypothesis"]

        # phase 3: report
        self._print_final(best_result, best_val_score, run_id, memory)

        return {
            "success": True,
            "run_id": run_id,
            "best_val_score": best_val_score,
            "best_result": best_result,
            "iterations": len(memory.get_history())
        }

    def _first_hypothesis(self, task: str, data_summary: dict) -> str:

        task_lower = task.lower()
        if any(word in task_lower for word in ["classify", "classification", "predict", "churn", "survival", "fraud"]):
            return (
                f"Start with a baseline LogisticRegression — "
                f"simple, interpretable, sets the floor. "
                f"Use all features from the dataset. "
                f"Target column: {data_summary.get('target_column', 'unknown')}."
            )
        else:
            return (
                f"Start with a baseline LinearRegression — "
                f"simple, interpretable, sets the floor. "
                f"Use all features from the dataset. "
                f"Target column: {data_summary.get('target_column', 'unknown')}."
            )

    def _save_best(self, run_dir: Path, model_result: dict):
        # saving the code that produced the best result
        best_code_path = run_dir / "best_code.py"
        best_code_path.write_text(model_result["code"])

        # saving the raw output
        best_output_path = run_dir / "best_output.txt"
        best_output_path.write_text(model_result["raw_output"])

    def _print_final(self, best_result: dict, best_val_score: float, run_id: str, memory: ExperimentMemory):
        print("\n{'═' * 50}")
        print("DONE")
        print(f"{'═' * 50}")

        if best_result:
            metrics = best_result["metrics"]
            print(f"Best model:    {metrics.get('model', 'unknown')}")
            print(f"Best val score: {best_val_score}")
            print(f"Metric:        {metrics.get('metric', 'unknown')}")
        else:
            print("No successful model found.")

        print(f"Iterations run: {len(memory.get_history())}")
        print(f"Results saved to: runs/{run_id}/")
        print(f"{'═' * 50}\n")

    def _make_run_id(self, file_name: str) -> str:
        from datetime import datetime
        stem = Path(file_name).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{timestamp}"
