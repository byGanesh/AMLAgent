import json
from pathlib import Path
from datetime import datetime

class ExperimentMemory:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.run_dir / "memory.json"
        self.iterations = []

        if self.memory_file.exists():
            with open(self.memory_file) as f:
                self.iterations = json.load(f)

        def log(self, entry:dict):
            entry["iteration"] = len(self.iterations) + 1
            entry["timestamp"] = datetime.now().isoformat()
            self.iterations.append(entry)
            self._save()

        def get_history(self) -> list:
            return self.iterations

        def get_last(self) -> dict | None:
            if not self.iteratios:
                return None
            return self.iterations[-1]

        def get_best(self, metric: str, higher_is_better: bool = True) -> dict | None:
            if not self.iteratins:
                return None
            valid = [i for i in self.iterations if metric in i]

            if not valid:
                return None

            return max(valid, key=lambda x: x[metric]) if higher_is_better else min(valid, key=lambda x:x[metric])

        def to_string(self) -> str:
            if not self.iterations:
                return "No experiments run yet"

            lines = []
            for entry in self.iterations:
                lines.append(f"--- Iteration {entry['iteration']} ---")
                for key, value in entry.items():
                    if key not in ("iteration", "timestamp"):
                        lines.append(f"{key} : {value}")
                lines.append("")

            return "\n".join(lines)

        def _save(self):
            with open(self.memory_file, "w") as f:
                json.dump(self.iterations, f, indent=2)
