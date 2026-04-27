import json as json_lib

from src.memory.context_manager import calculate_context_usage, offload_to_summary


AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a # Question section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- ## Conversation Memory
- ## Knowledge Base Memory
- ## Workflow Memory
- ## Entity Memory
- ## Summary Memory

# Summary Expansion Policy
If critical detail is only present in Summary Memory or appears ambiguous, call expand_summary(summary_id) before relying on it.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If memory is insufficient, state what is missing and use an appropriate tool.
3. Use only the tools provided in this turn and choose the minimum necessary tool calls.
4. For conversation compaction, use summarize_and_store with thread_id.
"""


class MemoryAwareAgent:
    """Orchestrates memory retrieval, tool use, and persistence per user query."""

    def __init__(self, memory_manager, toolbox, llm_client, model: str = "gpt-5-mini"):
        self.memory_manager = memory_manager
        self.toolbox = toolbox
        self.llm_client = llm_client
        self.model = model

    def execute_tool(self, tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
        """Execute a registered tool by name."""
        if tool_name not in self.toolbox._tools_by_name:
            return f"Error: Tool '{tool_name}' not found"

        args = dict(tool_args or {})
        if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id:
            args["thread_id"] = str(current_thread_id)

        return str(self.toolbox._tools_by_name[tool_name](**args) or "Done")

    def _call_chat(self, messages: list, tools: list | None = None):
        """Call chat completions with optional function tools."""
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.llm_client.chat.completions.create(**kwargs)

    def call(self, query: str, thread_id: str = "1", max_iterations: int = 10, verbose: bool = False) -> str:
        """Run one complete memory-aware agent interaction."""
        thread_id = str(thread_id)
        steps = []

        memory_context = ""
        memory_context += self.memory_manager.read_conversational_memory(thread_id) + "\n\n"
        memory_context += self.memory_manager.read_knowledge_base(query) + "\n\n"
        memory_context += self.memory_manager.read_workflow(query) + "\n\n"
        memory_context += self.memory_manager.read_entity(query) + "\n\n"
        memory_context += self.memory_manager.read_summary_context(query, thread_id=thread_id) + "\n\n"

        usage = calculate_context_usage(memory_context, model=self.model)
        if usage["percent"] > 80:
            memory_context, _ = offload_to_summary(
                memory_context,
                self.memory_manager,
                self.llm_client,
                thread_id=thread_id,
                model=self.model,
            )

        context = f"# Question\n{query}\n\n{memory_context}"
        dynamic_tools = self.memory_manager.read_toolbox(query, k=5)

        self.memory_manager.write_conversational_memory(query, "user", thread_id)
        try:
            self.memory_manager.write_entity("", "", "", llm_client=self.llm_client, text=query)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        final_answer = ""

        for iteration in range(max_iterations):
            response = self._call_chat(messages, tools=dynamic_tools)
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json_lib.loads(tc.function.arguments or "{}")
                    except json_lib.JSONDecodeError:
                        tool_args = {}

                    try:
                        result = self.execute_tool(tool_name, tool_args, current_thread_id=thread_id)
                        status = "success"
                        error_message = None
                        steps.append(f"{tool_name}({tool_args}) -> success")
                    except Exception as exc:
                        result = f"Error: {exc}"
                        status = "failed"
                        error_message = str(exc)
                        steps.append(f"{tool_name}({tool_args}) -> failed")

                    log_id = self.memory_manager.write_tool_log(
                        thread_id=thread_id,
                        tool_call_id=tc.id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result=result,
                        status=status,
                        error_message=error_message,
                        metadata={"iteration": iteration + 1},
                    )

                    if len(result) > 3000:
                        result_for_llm = (
                            result[:3000]
                            + "\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: "
                            + str(log_id)
                            + "]"
                        )
                    else:
                        result_for_llm = result

                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_for_llm})
                    if verbose:
                        print(f"tool={tool_name} status={status}")
            else:
                final_answer = msg.content or ""
                break

        if not final_answer:
            final_answer = "I was unable to complete the request within the allowed iterations."

        if steps:
            self.memory_manager.write_workflow(query, steps, final_answer)

        try:
            self.memory_manager.write_entity("", "", "", llm_client=self.llm_client, text=final_answer)
        except Exception:
            pass

        self.memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)
        return final_answer
