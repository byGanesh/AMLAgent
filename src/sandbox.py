import subprocess
import shutil
from pathlib import Path


class Sandbox:
    def __init__(self, image: str = "amlagent-sandbox", timeout: int = 300):
        self.image = image
        self.timeout = timeout

    def run(self, code: str, workspace: str) -> dict:
        workspace = Path(workspace).resolve()
        runfile = workspace / "_run.py"
        try:
            runfile.write_text(code)
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{workspace}:/workspace",
                "-w", "/workspace",
                "--network", "none",
                "--memory", "4g", "--memory-swap", "4g",
                "--cpus", "2", "--pids-limit", "64",
                self.image,
                "python", "_run.py"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Execution timed out."}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}
        finally:
            if runfile.exists():
                runfile.unlink()
