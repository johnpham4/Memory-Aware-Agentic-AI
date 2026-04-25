import os
import sys
import time
import warnings
import json as json_lib
import uuid
import inspect

from datetime import datetime
from typing import Callable, Optional, Union

import psycopg2
import psycopg2.extras

from loguru import logger
from pydantic import BaseModel

from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.utils import DistanceStrategy

from src.config.settings import settings


def suppress_warnings():
    warnings.filterwarnings("ignore")


def connect_to_postgres(max_retries=3, retry_delay=5, connect_timeout=5):
    """Connect to PostgreSQL database with retry logic."""
    for attempt in range(1, max_retries + 1):
        conn = None
        try:
            logger.info(f"Connection attempt {attempt}/{max_retries}...")

            conn = psycopg2.connect(
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                connect_timeout=connect_timeout
            )

            conn.autocommit = True
            logger.info("Connected successfully!")

            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                logger.info(f"DB version: {version}")

            return conn

        except psycopg2.OperationalError as e:
            logger.warning(f"Connection failed (attempt {attempt}/{max_retries})")
            logger.error(f"Error: {e}")

            if conn:
                conn.close()

            if attempt < max_retries:
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if conn:
                conn.close()
            raise

    raise ConnectionError("Failed to connect after all retries")


def table_exists(conn, table_name):
    """Check if a table exists in PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            );
        """, (table_name,))
        return cur.fetchone()[0]


class MemoryManager:
    """Memory manager for AI agents using PostgreSQL + pgvector."""

    def __init__(
        self,
        conn,
        conversation_table: str,
        knowledge_base_vs,
        workflow_vs,
        toolbox_vs,
        entity_vs,
        summary_vs,
        tool_log_table: str | None = None
    ):
        self.conn = conn
        self.conversation_table = conversation_table
        self.tool_log_table = tool_log_table
        self.knowledge_base_vs = knowledge_base_vs
        self.workflow_vs = workflow_vs
        self.toolbox_vs = toolbox_vs
        self.entity_vs = entity_vs
        self.summary_vs = summary_vs

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Write a message to conversational memory."""
        msg_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.conversation_table} (id, thread_id, role, content)
                VALUES (%s, %s, %s, %s)
            """, (msg_id, thread_id, role, content))
        self.conn.commit()
        return msg_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        """Read conversation history for a thread."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT role, content, timestamp
                FROM {self.conversation_table}
                WHERE thread_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (thread_id, limit))
            rows = cur.fetchall()

        if not rows:
            return ""

        lines = []
        for row in reversed(rows):
            lines.append(f"{row['role']}: {row['content']}")
        return "\n".join(lines)

    def mark_as_summarized(self, thread_id: str, summary_id: str):
        """Mark messages as summarized."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self.conversation_table}
                SET summary_id = %s
                WHERE thread_id = %s AND summary_id IS NULL
            """, (summary_id, thread_id))
        self.conn.commit()

    def write_tool_log(
        self,
        thread_id: str,
        tool_name: str,
        tool_args,
        result: str,
        status: str = "success",
        tool_call_id: str | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """Write tool execution log."""
        if not self.tool_log_table:
            return None

        log_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.tool_log_table}
                (id, thread_id, tool_name, tool_args, result, status, tool_call_id, error_message, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                log_id, thread_id, tool_name,
                json_lib.dumps(tool_args) if tool_args else None,
                result, status, tool_call_id, error_message,
                json_lib.dumps(metadata) if metadata else None
            ))
        self.conn.commit()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Read tool execution logs for a thread."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {self.tool_log_table}
                WHERE thread_id = %s
                ORDER BY executed_at DESC
                LIMIT %s
            """, (thread_id, limit))
            rows = cur.fetchall()

        return [dict(row) for row in rows] if rows else []

    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        """Write to knowledge base vector store."""
        if isinstance(text, str):
            text = [text]
            metadata = [metadata] if isinstance(metadata, dict) else [metadata]

        self.knowledge_base_vs.add_texts(text, metadatas=metadata)

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        """Search knowledge base."""
        docs = self.knowledge_base_vs.similarity_search(query, k=k)
        return "\n\n".join([f"Source: {doc.metadata}\n{doc.page_content}" for doc in docs])

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        """Store workflow execution pattern."""
        content = f"Query: {query}\nSteps: {json_lib.dumps(steps)}\nFinal Answer: {final_answer}"
        metadata = {"success": success, "query": query}
        self.workflow_vs.add_texts([content], metadatas=[metadata])

    def read_workflow(self, query: str, k: int = 3) -> str:
        """Retrieve similar workflow patterns."""
        docs = self.workflow_vs.similarity_search(query, k=k)
        return "\n\n".join([f"Workflow: {doc.page_content}" for doc in docs])

    def write_toolbox(self, text: str, metadata: dict):
        """Store tool in toolbox."""
        self.toolbox_vs.add_texts([text], metadatas=[metadata])

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve relevant tools."""
        docs = self.toolbox_vs.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def write_entity(self, name: str, entity_type: str, description: str):
        """Store entity information."""
        content = f"{name} ({entity_type}): {description}"
        metadata = {"name": name, "type": entity_type}
        self.entity_vs.add_texts([content], metadatas=[metadata])

    def read_entity(self, query: str, k: int = 5) -> str:
        """Retrieve relevant entities."""
        docs = self.entity_vs.similarity_search(query, k=k)
        return "\n\n".join([f"Entity: {doc.page_content}" for doc in docs])

    def write_summary(
        self,
        summary_id: str,
        full_content: str,
        summary: str,
        description: str,
        thread_id: str | None = None,
    ):
        """Store summary."""
        content = f"ID: {summary_id}\nDescription: {description}\nSummary: {summary}"
        metadata = {"summary_id": summary_id, "thread_id": thread_id, "description": description}
        self.summary_vs.add_texts([content], metadatas=[metadata])

    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary."""
        docs = self.summary_vs.similarity_search(summary_id, k=1)
        return docs[0].page_content if docs else ""

    def read_summary_context(self, query: str = "", k: int = 10, thread_id: str | None = None) -> str:
        """Retrieve summary context."""
        if not query:
            query = "summary"
        docs = self.summary_vs.similarity_search(query, k=k)
        return "\n\n".join([f"Summary: {doc.page_content}" for doc in docs])


class StoreManager:
    """Manages vector stores and SQL tables."""

    def __init__(self, conn, embedding_function, table_names, distance_strategy,
                 conversational_table, tool_log_table: str | None = None):
        """Initialize all stores."""
        self.conn = conn
        self.embedding_function = embedding_function
        self.distance_strategy = distance_strategy
        self._conversational_table = conversational_table
        self._tool_log_table = tool_log_table

        connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            database=settings.POSTGRES_DB,
        )

        self._knowledge_base_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['knowledge_base'],
            connection_string=connection_string,
            distance_strategy=distance_strategy,
        )

        self._workflow_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['workflow'],
            connection_string=connection_string,
            distance_strategy=distance_strategy,
        )

        self._toolbox_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['toolbox'],
            connection_string=connection_string,
            distance_strategy=distance_strategy,
        )

        self._entity_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['entity'],
            connection_string=connection_string,
            distance_strategy=distance_strategy,
        )

        self._summary_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['summary'],
            connection_string=connection_string,
            distance_strategy=distance_strategy,
        )

    def get_knowledge_base_store(self):
        """Return the knowledge base vector store."""
        return self._knowledge_base_vs

    def get_workflow_store(self):
        """Return the workflow vector store."""
        return self._workflow_vs

    def get_toolbox_store(self):
        """Return the toolbox vector store."""
        return self._toolbox_vs

    def get_entity_store(self):
        """Return the entity vector store."""
        return self._entity_vs

    def get_summary_store(self):
        """Return the summary vector store."""
        return self._summary_vs

    def get_conversational_table(self):
        """Return the conversational history table name."""
        return self._conversational_table

    def get_tool_log_table(self):
        """Return the tool log table name."""
        return self._tool_log_table


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""
    name: str
    description: str
    signature: str
    parameters: dict
    return_type: str


class Toolbox:
    """Register and retrieve tools with semantic search."""

    def __init__(self, memory_manager, llm_client, embedding_function, model: str = "gpt-5"):
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.model = model

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        return self.embedding_function.embed_query(text)

    def _augment_docstring(self, docstring: str, source_code: str = "") -> str:
        """Use LLM to improve docstring for better retrieval."""
        prompt = f"""Enhance this tool docstring to make it better for semantic search and tool discovery.

Original Docstring:
{docstring}

Source Code:
{source_code}

Return ONLY the improved docstring, no other text."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500
            )
            return response.choices[0].message.content
        except:
            return docstring

    def _generate_queries(self, docstring: str, num_queries: int = 5) -> list[str]:
        """Generate synthetic queries that would trigger this tool."""
        prompt = f"""Generate {num_queries} short, diverse natural language queries that would require this tool.

Tool Docstring:
{docstring}

Return ONLY the queries, one per line, no numbering or extra text."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500
            )
            queries = response.choices[0].message.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]
        except:
            return [docstring]

    def _get_tool_metadata(self, func: Callable) -> ToolMetadata:
        """Extract metadata from a function."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or func.__name__

        return ToolMetadata(
            name=func.__name__,
            description=docstring.split('\n')[0],
            signature=str(sig),
            parameters={param.name: str(param.annotation) for param in sig.parameters.values()},
            return_type=str(sig.return_annotation) if sig.return_annotation else "Any"
        )

    def register_tool(self, func: Optional[Callable] = None, augment: bool = False) -> Union[str, Callable]:
        """Register a tool with the toolbox."""
        def decorator(f):
            metadata = self._get_tool_metadata(f)
            docstring = inspect.getdoc(f) or f.__name__

            if augment:
                docstring = self._augment_docstring(docstring, inspect.getsource(f))

            content = f"Tool: {metadata.name}\n{docstring}"
            metadata_dict = metadata.dict()

            self.memory_manager.write_toolbox(content, metadata_dict)
            logger.info(f"Registered tool: {metadata.name}")

            return f

        if func:
            return decorator(func)
        return decorator


MODEL_TOKEN_LIMITS = {
    "gpt-4": 128000,
    "gpt-5": 256000,
}


def calculate_context_usage(context: str, model: str = "gpt-5") -> dict:
    """Calculate context window usage as percentage."""
    estimated_tokens = len(context) // 4
    max_tokens = MODEL_TOKEN_LIMITS.get(model, 128000)
    percentage = (estimated_tokens / max_tokens) * 100
    return {"tokens": estimated_tokens, "max": max_tokens, "percent": round(percentage, 1)}


def monitor_context_window(context: str, model: str = "gpt-5") -> dict:
    """Monitor the current context window and return capacity utilization."""
    result = calculate_context_usage(context, model)

    if result['percent'] < 50:
        result['status'] = 'ok'
    elif result['percent'] < 80:
        result['status'] = 'warning'
    else:
        result['status'] = 'critical'

    return result


def summarise_context_window(
    content: str,
    memory_manager,
    llm_client,
    model: str = "gpt-5",
    thread_id: str | None = None,
) -> dict:
    """Summarise content using LLM and store in summary memory."""
    cleaned = (content or "").strip()
    if not cleaned:
        return {"status": "nothing_to_summarize"}

    def _message_text(resp) -> str:
        if hasattr(resp, 'choices') and len(resp.choices) > 0:
            choice = resp.choices[0]
            if hasattr(choice, 'message'):
                return choice.message.content
            elif hasattr(choice, 'text'):
                return choice.text
        return ""

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
        max_completion_tokens=4000
    )
    summary = _message_text(response)

    if not summary:
        return {"status": "nothing_to_summarize"}

    desc_prompt = f"""Create a short 8-12 word label for this summary.
Return ONLY the label.

Requirements:
- Be specific about the topic/outcome.
- Include a concrete signal (entity, task, or issue).
- Do not use generic labels like "Conversation summary".

Summary:
{summary}"""

    desc_response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": desc_prompt}],
        max_completion_tokens=2000
    )
    description = _message_text(desc_response).strip().strip('"').strip("'")
    if not description or description.lower() in {
        "conversation summary",
        "summary",
        "chat summary",
        "thread summary",
    }:
        description = "Summarized conversation"

    summary_id = str(uuid.uuid4())[:8]
    memory_manager.write_summary(summary_id, cleaned, summary, description, thread_id=thread_id)

    return {"id": summary_id, "description": description, "summary": summary}


def offload_to_summary(context: str, memory_manager, llm_client, thread_id: str | None = None) -> tuple[str, list[dict]]:
    """Simple context compaction."""
    raw_context = (context or "").strip()

    if thread_id is not None:
        result = summarize_conversation(thread_id, memory_manager, None)
    else:
        result = summarise_context_window(raw_context, memory_manager, None, thread_id=thread_id)

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
        parts = compact_context.split("## Conversation Memory", 1)
        if len(parts) == 2:
            compact_context = parts[0] + "## Conversation Memory\n" + conversation_stub + parts[1].split("##", 1)[-1]
    else:
        compact_context += "\n" + conversation_stub

    if "## Summary Memory" in compact_context:
        compact_context = compact_context.replace("## Summary Memory", f"## Summary Memory\n{summary_ref}\n\n## Summary Memory")
    else:
        compact_context += f"\n\n## Summary Memory\n{summary_ref}"

    return compact_context, [result]


def summarize_conversation(thread_id: str, memory_manager, llm_client) -> dict:
    """Summarize all unsummarized messages in a thread and mark them."""
    thread_id = str(thread_id)
    with memory_manager.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(f"""
            SELECT id, role, content, timestamp
            FROM {memory_manager.conversation_table}
            WHERE thread_id = %s AND summary_id IS NULL
            ORDER BY timestamp ASC
        """, (thread_id,))
        rows = cur.fetchall()

    if not rows:
        return {"status": "nothing_to_summarize"}

    message_ids: list[str] = []
    transcript_lines: list[str] = []
    for row in rows:
        msg_id = row['id']
        role = row['role']
        content = row['content']
        message_ids.append(msg_id)
        transcript_lines.append(f"{role}: {content}")

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        return {"status": "nothing_to_summarize"}

    result = summarise_context_window(transcript, memory_manager, llm_client, thread_id=thread_id)
    if result.get("status") == "nothing_to_summarize":
        return result

    summary_id = result["id"]
    with memory_manager.conn.cursor() as cur:
        for msg_id in message_ids:
            cur.execute(f"""
                UPDATE {memory_manager.conversation_table}
                SET summary_id = %s
                WHERE id = %s
            """, (summary_id, msg_id))
    memory_manager.conn.commit()

    result["num_messages_summarized"] = len(message_ids)

    logger.info(f"Conversation summarized: [Summary ID: {summary_id}]")
    logger.info(f"Description: {result['description']}")
    logger.info(f"Messages marked summarized: {len(message_ids)}")

    return result


def register_summary_tools(toolbox, memory_manager, llm_client):
    """Register summary-related tools with the toolbox."""

    def expand_summary(summary_id: str, thread_id: str = None) -> str:
        """Expand a compressed summary to see full details."""
        expanded = memory_manager.read_summary_memory(summary_id, thread_id=thread_id)
        return expanded or "Summary not found"

    def summarize_and_store(text: str = "", thread_id: str = None) -> str:
        """Summarize given text or current thread and store in memory."""
        if text:
            result = summarise_context_window(text, memory_manager, llm_client)
        elif thread_id:
            result = summarize_conversation(thread_id, memory_manager, llm_client)
        else:
            return "Error: provide either text or thread_id"

        if result.get("status") == "nothing_to_summarize":
            return "Nothing to summarize"

        return f"Summarized as: {result['description']}"

    toolbox.register_tool(expand_summary, augment=True)
    toolbox.register_tool(summarize_and_store, augment=True)

    registered_tools = {
        "expand_summary": expand_summary,
        "summarize_and_store": summarize_and_store,
    }

    logger.info(f"Registered {len(registered_tools)} summary tools: {list(registered_tools.keys())}")

    return registered_tools
