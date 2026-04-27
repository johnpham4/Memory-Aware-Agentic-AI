import uuid


MODEL_TOKEN_LIMITS = {
    "gpt-5-mini": 256000,
    "gpt-5": 256000,
}


def calculate_context_usage(context: str, model: str = "gpt-5-mini") -> dict:
    """Estimate context usage percentage based on character length."""
    estimated_tokens = len(context or "") // 4
    max_tokens = MODEL_TOKEN_LIMITS.get(model, 128000)
    percentage = (estimated_tokens / max_tokens) * 100 if max_tokens else 0
    return {"tokens": estimated_tokens, "max": max_tokens, "percent": round(percentage, 1)}


def monitor_context_window(context: str, model: str = "gpt-5-mini") -> dict:
    """Return context usage and status classification."""
    result = calculate_context_usage(context, model)

    if result["percent"] < 50:
        result["status"] = "ok"
    elif result["percent"] < 80:
        result["status"] = "warning"
    else:
        result["status"] = "critical"

    return result


def _message_text(response) -> str:
    """Parse response text from chat completions response variants."""
    if not getattr(response, "choices", None):
        return ""

    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


def summarise_context_window(
    content: str,
    memory_manager,
    llm_client,
    model: str = "gpt-5-mini",
    thread_id: str | None = None,
) -> dict:
    """Summarize content and store summary metadata in summary memory."""
    cleaned = (content or "").strip()
    if not cleaned:
        return {"status": "nothing_to_summarize"}

    summary_prompt = f"""You are creating durable memory for an AI research assistant.
Summarize this conversation so it can be resumed accurately later.

Output with exactly these headings:
### Technical Information
### Emotional Context
### Entities & References
### Action Items & Decisions

Rules:
- Keep concrete details (names, dates, APIs, errors, decisions).
- Separate confirmed facts from open questions where relevant.
- Do not invent information.
- Keep it concise and useful for continuation.

Conversation:
{cleaned[:6000]}"""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        max_completion_tokens=4000,
    )
    summary = _message_text(response)

    if not summary:
        retry_prompt = f"""Summarize this conversation in <= 180 words using these headings:
### Technical Information
### Emotional Context
### Entities & References
### Action Items & Decisions

Conversation:
{cleaned[:6000]}"""
        retry_response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            max_completion_tokens=4000,
        )
        summary = _message_text(retry_response)

    if not summary:
        excerpt = cleaned[:500].replace("\n", " ").strip()
        summary = (
            "### Technical Information\n"
            f"{excerpt or '(No content provided.)'}\n\n"
            "### Emotional Context\n"
            "Not available from model output.\n\n"
            "### Entities & References\n"
            "Not available from model output.\n\n"
            "### Action Items & Decisions\n"
            "Not available from model output."
        )

    desc_prompt = f"""Create a short 8-12 word label for this summary.
Return ONLY the label.

Summary:
{summary}"""

    desc_response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": desc_prompt}],
        max_completion_tokens=2000,
    )
    description = _message_text(desc_response) or "Conversation summary"

    summary_id = str(uuid.uuid4())[:8]
    memory_manager.write_summary(summary_id, cleaned, summary, description, thread_id=thread_id)

    return {"id": summary_id, "description": description, "summary": summary}


def summarize_conversation(
    thread_id: str,
    memory_manager,
    llm_client,
    model: str = "gpt-5-mini",
) -> dict:
    """Summarize unsummarized messages in a thread and mark those rows."""
    thread_id = str(thread_id)

    with memory_manager.conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, role, content, timestamp
            FROM {memory_manager.conversation_table}
            WHERE thread_id = %s AND summary_id IS NULL
            ORDER BY timestamp ASC
            """,
            (thread_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return {"status": "nothing_to_summarize"}

    message_ids = []
    transcript_lines = []
    for msg_id, role, content, timestamp in rows:
        message_ids.append(msg_id)
        ts_text = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "Unknown"
        transcript_lines.append(f"[{ts_text}] [{str(role).upper()}] {content}")

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        return {"status": "nothing_to_summarize"}

    result = summarise_context_window(
        transcript,
        memory_manager,
        llm_client,
        model=model,
        thread_id=thread_id,
    )
    if result.get("status") == "nothing_to_summarize":
        return result

    summary_id = result["id"]

    with memory_manager.conn.cursor() as cur:
        cur.executemany(
            f"""
            UPDATE {memory_manager.conversation_table}
            SET summary_id = %s
            WHERE id = %s AND summary_id IS NULL
            """,
            [(summary_id, msg_id) for msg_id in message_ids],
        )

    memory_manager.conn.commit()
    result["num_messages_summarized"] = len(message_ids)

    return result


def offload_to_summary(
    context: str,
    memory_manager,
    llm_client,
    thread_id: str | None = None,
    model: str = "gpt-5-mini",
) -> tuple[str, list[dict]]:
    """Compact conversation-heavy context and attach summary references."""
    raw_context = (context or "").strip()

    if thread_id:
        result = summarize_conversation(thread_id, memory_manager, llm_client, model=model)
    else:
        result = summarise_context_window(raw_context, memory_manager, llm_client, model=model)

    if result.get("status") == "nothing_to_summarize":
        return raw_context, []

    summary_ref = f"[Summary ID: {result['id']}] {result['description']}"
    conversation_stub = (
        "## Conversation Memory\n"
        "Older conversation content was summarized to reduce context size.\n"
        "Use Summary Memory references + expand_summary(id) for full detail."
    )

    compact_context = raw_context
    if "## Conversation Memory" in compact_context:
        lines = compact_context.splitlines()
        rebuilt = []
        in_conversation = False
        inserted_stub = False

        for line in lines:
            if line.startswith("## "):
                heading = line.strip()
                if heading == "## Conversation Memory":
                    in_conversation = True
                    if not inserted_stub:
                        if rebuilt and rebuilt[-1].strip():
                            rebuilt.append("")
                        rebuilt.extend(conversation_stub.splitlines())
                        rebuilt.append("")
                        inserted_stub = True
                    continue
                in_conversation = False

            if not in_conversation:
                rebuilt.append(line)

        compact_context = "\n".join(rebuilt).strip()
    else:
        compact_context = f"{conversation_stub}\n\n{compact_context}".strip()

    if "## Summary Memory" in compact_context:
        compact_context = f"{compact_context}\n{summary_ref}".strip()
    else:
        compact_context = (
            f"{compact_context}\n\n"
            "## Summary Memory\n"
            "Use expand_summary(id) to retrieve full underlying content.\n"
            f"{summary_ref}"
        ).strip()

    return compact_context, [result]
