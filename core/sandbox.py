import subprocess
import tempfile
import shutil
import os
from pathlib import Path

class Sandbox:
    def __init__(self, image: str = "amlagent-sandbox", timeout: int = 300):
        self.image = image
        self.timeout = timeout

    def run(self, code:str, data_path: str = None) -> dict:

        # creating a temp dir on machine, this gets mounted into the container so the code can read data
        tmpdir = tempfile.mkdtemp()

        try:
            # this writes the agent's code into the temp dir
            copy_file = Path(tmpdir) / "run.py"
            copy_file.write_text(code)
            if data_path:
                shutil.copy(data_path, tmpdir)

            # docker command
            cmd = [
                "docker", "run",
                "--rm",             # it deletes container after it exits
                "--network", "none", # it prevents internet access
                "--memory", "4g",    # maximum 4GB RAM
                "--memory-swap", "4g",  # swapping is prevented
                "--cpus", "2",     # max 2 CPU cores
                "--pids-limit", "64",  # max 64 processes (it prevents fork bombs)
                "-v", f"{tmpdir}:/workspace", # it mounts temp dir as /workspace inside container, anything outside it is not visible to the agent
                "-w", "/workspace", # setting working dir to /workspace
                self.image,   # docker image to use
                "python", "run.py" # it commands to run inside
            ]


            result = subprocess.run(
                cmd,
                capture_output=True,
                text = True,
                timeout=self.timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:8000],  # 8000 chars
                "stderr": result.stderr[:2000],
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout} seconds.",
                "returncode": -1
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Sandbox error: {str(e)}",
                "returncode": -1
            }

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True) # it always cleans up temp dir, even if something crashed
