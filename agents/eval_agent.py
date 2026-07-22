from core.llm import call_llm, extract_text
import json


SYSTEM_PROMPT = """You are the Eval Agent in an autonomous ML pipeline.

You receive the full experiment history and the latest result.
Your job is to:
1. Diagnose why the last experiment succeeded or failed
2. Identify the specific failure mode if it failed
3. Recommend exactly what to try next
4. Decide whether to stop or continue

Failure modes you understand:
- Overfitting: train_score >> val_score (gap > 0.1)
- Underfitting: both scores are low, model too simple
- Wrong features: scores plateau despite model changes
- Class imbalance: val_score low despite good train_score, check notes
- Data leakage: train_score suspiciously high (> 0.99)
- Wrong metric: model optimizing something other than requested metric
- Convergence failure: model didn't converge, check notes

Stop conditions:
- Val score has not improved for 3 consecutive iterations
- Val score crossed the target threshold
- Max iterations reached
- Data leakage detected — stop immediately and flag it

Your response must always be a JSON object with these exact keys:
{
    "diagnosis": "<what happened and why>",
    "failure_mode": "<one of the failure modes above or null if success>",
    "hypothesis": "<exactly what to try next - be specific, name the model, parameters, features>",
    "reasoning": "<why you think this will help>",
    "should_stop": <true or false>,
    "stop_reason": "<why stopping, or null if continuing>",
    "confidence": <float 0-1, how confident you are this next step will improve things>
}

Be specific in hypothesis. Not 'try a better model'. Say exactly:
'XGBoostClassifier with max_depth=3, learning_rate=0.05, subsample=0.8, using features: age, income, tenure'
"""


class EvalAgent:
    def __init__(self):
        pass

    def run(self, metrics: dict, history: str, task: str, iteration: int) -> dict:
        print(f"\n[Eval Agent] Diagnosing iteration {iteration} results...")

        messages = [
            {
                "role": "user",
                "content": f"""Task: {task}
Current iteration: {iteration}

Latest results:
{json.dumps(metrics, indent=2)}

Full experiment history:
{history}

Diagnose the results and decide what to do next.
Return only the JSON object, no explanation, no markdown fences."""
            }
        ]

        response = call_llm(SYSTEM_PROMPT, messages, max_tokens=1024)
        raw = extract_text(response).strip()

        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:

            print("[Eval Agent] Warning: could not parse response. Continuing with generic hypothesis.")
            decision = {
                "diagnosis": "Could not parse eval response.",
                "failure_mode": None,
                "hypothesis": "Try RandomForestClassifier with n_estimators=100, default params.",
                "reasoning": "Fallback hypothesis due to parse error.",
                "should_stop": False,
                "stop_reason": None,
                "confidence": 0.3
            }

        self._print_decision(decision, metrics)
        return decision

    def _print_decision(self, decision: dict, metrics: dict):
        val_score = metrics.get("val_score", "N/A")
        train_score = metrics.get("train_score", "N/A")

        print(f"[Eval Agent] Train: {train_score} | Val: {val_score}")
        print(f"[Eval Agent] Diagnosis: {decision['diagnosis']}")

        if decision["failure_mode"]:
            print(f"[Eval Agent] Failure mode: {decision['failure_mode']}")

        if decision["should_stop"]:
            print(f"[Eval Agent] Stopping, {decision['stop_reason']}")
        else:
            print(f"[Eval Agent] Next: {decision['hypothesis']}")
            print(f"[Eval Agent] Confidence: {decision['confidence']}")
