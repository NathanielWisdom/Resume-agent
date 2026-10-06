"""Multi-server MCP agent: resume + memory over stdio, Groq tool-calling loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq
from groq import RateLimitError
from mcp import types
from mcp.client.session_group import ClientSessionGroup
from mcp.client.stdio import StdioServerParameters

load_dotenv()

ROOT = Path(__file__).resolve().parent
RESUME_SERVER = ROOT / "resume_server.py"

SYSTEM_PROMPT = """You are a helpful resume assistant with access to MCP tools.

- Use get_resume_text (or resume-prefixed equivalent) for facts from the user's live Google Doc resume.
- Use memory graph tools (search_nodes, create_entities, add_observations, read_graph, etc.) to remember user preferences and context across conversations.
- Be concise and accurate. When citing resume content, base answers on tool results, not guesses.
"""

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 20


def component_name_hook(name: str, server_info: types.Implementation) -> str:
    server_label = (server_info.name or "server").replace(" ", "_")
    return f"{server_label}__{name}"


def _stdio_env() -> dict[str, str]:
    """Environment for MCP stdio child processes (secrets are not inherited by default)."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if value is not None:
            env[key] = value

    for key in (
        "GOOGLE_DOC_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "SERVICE_ACCOUNT_PATH",
        "GOOGLE_SUBJECT_EMAIL",
        "MEMORY_FILE_PATH",
        "PATH",
        "SystemRoot",
        "PATHEXT",
    ):
        if key in os.environ:
            env[key] = os.environ[key]

    if "SERVICE_ACCOUNT_PATH" not in env and "GOOGLE_APPLICATION_CREDENTIALS" not in env:
        default_sa = str(ROOT / "service_account.json")
        if Path(default_sa).is_file():
            env["SERVICE_ACCOUNT_PATH"] = default_sa

    return env


def mcp_tools_to_groq(group_tools: dict[str, types.Tool]) -> list[dict[str, Any]]:
    groq_tools: list[dict[str, Any]] = []
    for name, tool in group_tools.items():
        schema = getattr(tool, "inputSchema", None)
        if schema is None:
            schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = getattr(tool, "parameters", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}

        groq_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": schema,
                },
            }
        )
    return groq_tools


def tool_result_to_text(result: types.CallToolResult) -> str:
    parts: list[str] = []

    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", False)
    if is_error:
        parts.append("[tool error]")

    content = getattr(result, "content", []) or []
    for block in content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        parts.append(json.dumps(structured, indent=2))

    return "\n".join(parts) if parts else "(empty tool result)"


def assistant_message_from_choice(choice_message: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": choice_message.content or ""}
    if choice_message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in choice_message.tool_calls
        ]
    return msg


class ResumeAgent:
    def __init__(self, model: str, groq_api_key: str) -> None:
        self.model = model
        self.client = AsyncGroq(api_key=groq_api_key)
        self.group: ClientSessionGroup | None = None

    async def connect(self) -> None:
        if not RESUME_SERVER.is_file():
            raise FileNotFoundError(f"Resume server not found: {RESUME_SERVER}")

        self.group = ClientSessionGroup(component_name_hook=component_name_hook)
        await self.group.__aenter__()

        env = _stdio_env()
        await self.group.connect_to_server(
            StdioServerParameters(
                command=sys.executable,
                args=[str(RESUME_SERVER)],
                env=env,
            )
        )
        await self.group.connect_to_server(
            StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
                env=env,
            )
        )

        names = sorted(self.group.tools.keys())
        print(f"Connected. {len(names)} tools: {', '.join(names)}")

    async def close(self) -> None:
        if self.group is not None:
            await self.group.__aexit__(None, None, None)
            self.group = None

    async def process_query(self, query: str) -> str:
        if self.group is None:
            raise RuntimeError("Not connected. Call connect() first.")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        groq_tools = mcp_tools_to_groq(self.group.tools)

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=groq_tools,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                )
            except RateLimitError:
                return (
                    "Groq rate limit reached (free tier). Wait a moment and try again, "
                    "or reduce request frequency."
                )

            choice = response.choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                return (choice.content or "").strip() or "(No response text)"

            messages.append(assistant_message_from_choice(choice))

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                if not isinstance(fn_args, dict):
                    fn_args = {}

                result = await self.group.call_tool(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_to_text(result),
                    }
                )

        return "Stopped after maximum tool rounds. Try a simpler question."


async def run_repl(agent: ResumeAgent) -> None:
    print("\nMCP Resume Agent (Groq). Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer = await agent.process_query(query)
            print(f"\n{answer}\n")
        except Exception as err:
            print(f"\nError: {err}\n")


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Multi-server MCP resume agent")
    parser.add_argument("query", nargs="*", help="Optional single-shot query (default: interactive REPL)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Set GROQ_API_KEY in .env or the environment.", file=sys.stderr)
        return 1

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    agent = ResumeAgent(model=model, groq_api_key=api_key)
    try:
        await agent.connect()
        if args.query:
            print(await agent.process_query(" ".join(args.query)))
        else:
            await run_repl(agent)
    finally:
        await agent.close()
    return 0


def main() -> None:
    import asyncio

    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
