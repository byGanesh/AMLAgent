import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL"),
)

MODEL = os.getenv("LLM_MODEL")

def call_llm(system: str, messages: list, tools: list = [], max_tokens: int = 4096) -> dict:
    all_messages = [
        {
            "role": "system",
            "content": system,
        }
    ] + messages

    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": all_messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return response

def extract_text(response) -> str:
    return response.choices[0].message.content or ""

def extract_tool_use(response) -> tuple | None:
    message = response.choice[0].message
    if message.tool_calls:
        tool = message.tool_calls[0]
        return tool.function.name, json.loads(tool.function.arguments)
    return None

def get_stop_reason(response) -> str:
    return response.choices[0].finish_reason # reasons: "stop", "tool_calls", "length"
