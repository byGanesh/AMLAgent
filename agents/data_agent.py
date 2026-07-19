from core.llm import call_llm, extract_text
from core.sandbox import Sandbox
import json

SYSTEM_PROMPT = """
You are the Data agent in an autonomous ML pipeline.
Your job is to write Python code that:
1. Loads the dataset from the given file path
2. Profiles it - shape, dtypes, missing values, basic statistics
3. Cleans it - handle missing values with reasoned strategy per column
4. Engineers features - create new meaningful features based on what you see
5. Saves the cleaned dataframe as cleaned_data.csv in the same directory
6. Prints a structure summary of everything you did

Rules:
- Always save the cleaned data as clean_data.csv
- Always print a clear summary at the end
- The summary must include: final shape, features created, features dropped, missing value treatment
- Write complete, runnable Python code - no placeholders, no comments saying 'add code here'
- Use pandas, numpy, scikit-learn only - nothing else
- Do not use any plotting libraries - no matplotlib, no seaborn
- The file path will be provided to you - use it exactly as given
"""

class DataAgent:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def run(self, file_path: str, task: str) -> dict:
        print("\n[Data Agent] Analyzing dataset and writing EDA code...")

        # step 1. asking LLM to write the data preparation code
        messages = [
            {
                "role": "user",
                "content": f"""Dataset path: {file_path}
Task: {task}
Write complete Python code to load, profile, clean, and engineer features for this dataset.
Remember to save cleaned data as cleaned_data.csv or whatever the type is in the same directory as the input file.
Print a detailed summary of what you found and what you did.
                """
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens = 4096)
        code = extract_text(response)

        code = self._extract_code(code)  # it extracts code from markdown

        print("[Data Agent] Running code in sandbox...")

        # step 2. running the code in the sandbox
        result = self.sandbox.run(code, data_path=file_path)

        # step 3. if it failed, try once to fix it
        if not result["success"]:
            print("[Data Agent] Code failed. Attempting fix...")
            result = self._fix_and_retry(code, result["stderr"], file_path, task)

        if not result["success"]:
            return {
                "success": False,
                "error": result["stderr"],
                "summary": None,
                "code": code
            }

        print("[Data Agent] Done.")
        print(f"\n {result['stdout']}\n")

        # step 4. asking LLM to parse its own output into structured summary
        summary = self._parse_summary(result["stdout"], task)

        return {
            "success": True,
            "summary": summary,
            "raw_output": result["stdout"],
            "code": code
        }


    def _fix_and_retry(self, original_code: str, error: str, file_path: str, task: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""This code failed with the following error:

ERROR:
{error}

ORIGINAL CODE:
{original_code}

Fix the code and return the complete corrected version.
Dataset path: {file_path}
Task: {task}"""
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens=4096)
        fixed_code = self._extract_code(extract_text(response))
        return self.sandbox.run(fixed_code, data_path=file_path)

    def _parse_summary(self, stdout: str, task: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": f"""This is the output from a data preparation script:

{stdout}

Extract and return ONLY a JSON object with these exact keys:
{{
    "rows": <int>,
    "columns": <int>,
    "target_column": "<string>",
    "features": ["list", "of", "final", "feature", "names"],
    "missing_value_treatment": "<brief description>",
    "engineered_features": ["list", "of", "new", "features", "created"],
    "dropped_features": ["list", "of", "dropped", "features"],
    "notes": "<anything important the model agent should know>"
}}

Task context: {task}
Return only the JSON. No explanation. No markdown fences."""
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens=1024)
        raw = extract_text(response).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": stdout}

    def _extract_code(self, text: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()
