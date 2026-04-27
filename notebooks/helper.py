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
from psycopg2.extras import Json

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
    """
    A simplified memory manager for AI agents using Postgres Vector Database.

    Manages 7 types of memory:
    - Conversational: Chat history per thread (SQL table)
    - Tool Log: Raw tool execution outputs and metadata (SQL table)
    - Knowledge Base: Searchable documents (Vector store)
    - Workflow: Execution patterns (Vector store)
    - Toolbox: Available tools (Vector store)
    - Entity: People, places, systems (Vector store)
    - Summary: Storing compressed context window
    """

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

    # ==================== CONVERSATIONAL MEMORY (SQL) ====================

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Store a message in conversation history."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.conversation_table}
                (thread_id, role, content, metadata, timestamp)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (thread_id, role, content, Json({}),))

            record_id = cur.fetchone()[0]
        self.conn.commit()
        return record_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> str:
        """Read conversation history for a thread (excludes summarized messages)."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT role, content, timestamp
                FROM {self.conversation_table}
                WHERE thread_id = %s AND summary_id IS NULL
                ORDER BY timestamp DESC
                LIMIT %s
            """, (thread_id, limit))
            rows = cur.fetchall()

        if not rows:
            return ""
        # Keep newest-N selection but display in chronological order for readability.
        messages = [
            f"[{ts.strftime('%H:%M:%S')}] [{role}] {content}"
            for role, content, ts in reversed(rows)
        ]
        messages_formatted = "\n".join(messages)
        if not messages_formatted:
            messages_formatted = "(No unsummarized messages found for this thread.)"
        return f"""## Conversation Memory
### What this memory is
Chronological, unsummarized messages from the current thread. This memory captures user intent, constraints, and commitments made in recent turns.
### How you should leverage it
- Preserve continuity with prior decisions, terminology, and user preferences.
- Resolve references like "that", "previous step", or "the paper above" using earlier turns.
- If older context conflicts with newer user instructions, prioritize the latest user direction.
### Retrieved messages

{messages_formatted}"""


    def mark_as_summarized(self, thread_id: str, summary_id: str):
        """Mark all unsummarized messages in a thread as summarized."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self.conversation_table}
                SET summary_id = %s
                WHERE thread_id = %s AND summary_id IS NULL
            """, (summary_id, thread_id))
        self.conn.commit()
        logger.info(f"  📦 Marked messages as summarized (summary_id: {summary_id})")


    # ==================== TOOL LOG MEMORY (SQL) ====================

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
        """Persist raw tool execution logs for auditing and just-in-time retrieval."""
        if not self.tool_log_table:
            return None

        if isinstance(tool_args, (dict, list)):
            tool_args_payload = tool_args
        else:
            tool_args_payload = {"value": "" if tool_args is None else str(tool_args)}

        result_str = "" if result is None else str(result)
        metadata_payload = metadata if isinstance(metadata, dict) else {}

        with self.conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.tool_log_table}
                (thread_id, tool_call_id, tool_name, tool_args, result, status, error_message, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                thread_id,
                tool_call_id,
                tool_name,
                Json(tool_args_payload),
                result_str,
                status,
                error_message,
                Json(metadata_payload),
            ))
            log_id = cur.fetchone()[0]
        self.conn.commit()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Read recent tool logs for a thread, newest first."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, tool_call_id, tool_name, tool_args, result, status, error_message, metadata, timestamp
                FROM {self.tool_log_table}
                WHERE thread_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (thread_id, limit))
            rows = cur.fetchall()

        logs = []
        for log_id, tool_call_id, tool_name, tool_args, result_text, status, error_message, metadata, ts in rows:
            logs.append({
                "id": log_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_preview": result_text,
                "status": status,
                "error_message": error_message,
                "metadata": metadata,
                "timestamp": ts.isoformat() if ts else None,
            })
        return logs

    # ==================== KNOWLEDGE BASE (Vector Store) ====================


    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        """
        Store knowledge-base content with metadata.

        Supports:
        - Single record: text=str, metadata=dict
        - Batch insert: text=list[str], metadata=list[dict]
        """
        if isinstance(text, list):
            texts = [str(t) for t in text]
            if isinstance(metadata, list):
                metadatas = metadata
            else:
                metadatas = [metadata for _ in texts]

            if len(texts) != len(metadatas):
                raise ValueError(
                    f"Knowledge-base batch length mismatch: {len(texts)} texts vs {len(metadatas)} metadata rows"
                )
            self.knowledge_base_vs.add_texts(texts, metadatas)
            return

        self.knowledge_base_vs.add_texts([str(text)], [metadata if isinstance(metadata, dict) else {}])

    def read_knowledge_base(self, query: str, k: int = 3) -> str:
        """Search knowledge base for relevant content."""
        results = self.knowledge_base_vs.similarity_search(query, k=k)
        content = "\n".join([doc.page_content for doc in results])
        if not content:
            content = "(No relevant knowledge base passages found.)"
        return f"""## Knowledge Base Memory
### What this memory is
Retrieved background documents and previously ingested reference material relevant to the current query.
### How you should leverage it
- Ground responses in these passages when making factual or technical claims.
- Prefer concrete details from this memory over unsupported assumptions.
- If evidence is missing or ambiguous, state uncertainty and request clarification or additional retrieval.
### Retrieved passages

{content}"""

    # ==================== WORKFLOW (Vector Store) ====================

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        """Store a completed workflow pattern for future reference."""
        # Format steps as text
        steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
        text = f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"
        metadata = {
            "query": query,
            "success": success,
            "num_steps": len(steps),
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_vs.add_texts([text], [metadata])


    def read_workflow(self, query: str, k: int = 3) -> str:
        """Search for similar past workflows with at least 1 step."""
        # Filter to only include workflows that have steps (num_steps > 0)
        results = self.workflow_vs.similarity_search(
            query,
            k=k,
            filter={"num_steps": {"$gt": 0}}
        )
        if not results:
            return """## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows
(No relevant workflows found.)"""

        content = "\n---\n".join([doc.page_content for doc in results])
        return f"""## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows

{content}"""

    # ==================== TOOLBOX (Vector Store) ====================

    def write_toolbox(self, text: str, metadata: dict):
        """Store a tool definition in the toolbox."""
        self.toolbox_vs.add_texts([text], [metadata])

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Find relevant tools and return OpenAI-compatible schemas."""
        results = self.toolbox_vs.similarity_search(query, k=k)
        tools = []
        seen_tool_names: set[str] = set()

        for doc in results:
            meta = doc.metadata
            tool_name = meta.get("name", "tool")
            if tool_name in seen_tool_names:
                continue

            seen_tool_names.add(tool_name)
            # Extract parameters from metadata and convert to OpenAI format
            stored_params = meta.get("parameters", {})
            properties = {}
            required = []

            for param_name, param_info in stored_params.items():
                # Convert stored param info to OpenAI schema format
                if isinstance(param_info, dict):
                    param_type = param_info.get("type", "string")
                    has_default = "default" in param_info
                else:
                    param_type = str(param_info or "string")
                    has_default = False
                # Map Python types to JSON schema types
                type_mapping = {
                    "<class 'str'>": "string",
                    "<class 'int'>": "integer",
                    "<class 'float'>": "number",
                    "<class 'bool'>": "boolean",
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean"
                }
                json_type = type_mapping.get(param_type, "string")
                properties[param_name] = {"type": json_type}

                # If no default, it's required
                if not has_default:
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": meta.get("description", ""),
                    "parameters": {"type": "object", "properties": properties, "required": required}
                }
            })
        return tools
    # ==================== ENTITY (Vector Store) ====================

    def extract_entities(self, text: str, llm_client) -> list[dict]:
        """Use llm to extract entities (people, places, systems) from text."""
        if not text or len(text.strip()) < 5:
            return []

        prompt = f'''Extract entities from "{text[:500]}"
return JSON [{{"name": "X", "type": "PERSON|PLACE|SYSTEM", "description": "brief"}}]
If none: []'''

        try:
            response = llm_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2000
            )
            result = response.choices[0].message.content.strip()

            # Extract JSON array from response
            start, end = result.find("["), result.rfind("]")
            if start == -1 or end == -1:
                return []

            parsed = json_lib.loads(result[start:end+1])
            return [{"name": e["name"], "type": e.get("type", "UNKNOWN"), "description": e.get("description", "")}
                    for e in parsed if isinstance(e, dict) and e.get("name")]

        except Exception:
            return []


    def write_entity(self, name: str, entity_type: str, description: str, llm_client=None, text: str = None):
        """Store an entity OR extract and store entities from text."""
        if text and llm_client:
            # Extract and store entities from text
            entities = self.extract_entities(text, llm_client)
            for e in entities:
                self.entity_vs.add_texts(
                    [f"{e['name']} ({e['type']}): {e['description']}"],
                    [{"name": e['name'], "type": e['type'], "description": e['description']}]
                )
            return entities
        else:
            # store single entity directly
            self.entity_vs.add_texts(
                [f"{name} ({entity_type}): {description}"],
                [{"name": name, "type": entity_type, "description": description}]
            )

    def read_entity(self, query: str, k: int = 5) -> str:
        """Search for relevant entities."""
        results = self.entity_vs.similarity_search(query, k=k)
        if not results:
            return """## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities
(No entities found.)"""

        entities = [f"• {doc.metadata.get('name', '?')}: {doc.metadata.get('description', '')}"
                    for doc in results if hasattr(doc, 'metadata')]

        entities_formatted = '\n'.join(entities)
        return f"""## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities

{entities_formatted}"""


    # ==================== SUMMARY (Vector Store) ====================

    def write_summary(
        self,
        summary_id: str,
        full_content: str,
        summary: str,
        description: str,
        thread_id: str | None = None,
    ):
        """Store a summary with its original content."""
        metadata = {
            "id": summary_id,
            "full_content": full_content,
            "summary": summary,
            "description": description,
        }
        if thread_id is not None:
            metadata["thread_id"] = str(thread_id)
        self.summary_vs.add_texts(
            [f"{summary_id}: {description}"],
            [metadata]
        )
        return summary_id


    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary by ID (just-in-time retrieval)."""
        filters = {"id": summary_id}
        if thread_id is not None:
            filters["thread_id"] = str(thread_id)

        results = self.summary_vs.similarity_search(
            summary_id,
            k=5,
            filter=filters
        )
        if not results:
            if thread_id is not None:
                return f"Summary {summary_id} not found for thread {thread_id}."
            return f"Summary {summary_id} not found."
        doc = results[0]
        return doc.metadata.get('summary', 'No summary content.')


    def read_summary_context(self, query: str = "", k: int = 10, thread_id: str | None = None) -> str:
        """Get available summaries for context window (IDs + descriptions only)."""
        filters = None
        if thread_id is not None:
            filters = {"thread_id": str(thread_id)}
        results = self.summary_vs.similarity_search(query or "summary", k=k, filter=filters)
        if not results:
            scope_note = (
                f"(No summaries available for thread {thread_id}.)"
                if thread_id is not None
                else "(No summaries available.)"
            )
            return """## Summary Memory
### What this memory is
Compressed snapshots of older conversation windows preserved to retain long-range context.
### How you should leverage it
- Use summaries to maintain continuity when full historical messages are not in the active context window.
- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.
### Available summaries
""" + scope_note

        lines = [
            "## Summary Memory",
            "### What this memory is",
            "Compressed snapshots of older conversation windows preserved to retain long-range context.",
            "### How you should leverage it",
            "- Use summaries to maintain continuity when full historical messages are not in the active context window.",
            "- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.",
            "### Available summaries",
            "Use expand_summary(id) to retrieve the detailed underlying conversation."
        ]
        if thread_id is not None:
            lines.append(f"Scope: thread_id = {thread_id}")
        for doc in results:
            sid = doc.metadata.get('id', '?')
            desc = doc.metadata.get('description', 'No description')
            lines.append(f"  • [ID: {sid}] {desc}")
        return "\n".join(lines)

    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        """
        Retrieve all original conversations that were summarized with a given summary_id.
        Returns conversations in order of occurrence with timestamps.

        Args:
            summary_id: The ID of the summary to expand

        Returns:
            Formatted string with original conversations and timestamps
        """
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, role, content, timestamp
                FROM {self.conversation_table}
                WHERE summary_id = %s
                ORDER BY timestamp ASC
            """, (summary_id,))
            results = cur.fetchall()

        if not results:
            return f"No conversations found for summary_id: {summary_id}"

        # Format conversations with timestamps
        lines = [f"## Expanded Conversations for Summary ID: {summary_id}"]
        lines.append(f"Total messages: {len(results)}\n")

        for msg_id, role, content, timestamp in results:
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
            lines.append(f"[{ts_str}] [{role.upper()}]")
            lines.append(f"{content}")
            lines.append("")  # Empty line between messages

        return "\n".join(lines)


class StoreManager:
    """Manages vector stores and SQL tables."""

    def __init__(self, conn, embedding_function, table_names, distance_strategy,
                 conversational_table, tool_log_table: str | None = None):
        """Initialize all stores."""
        self.conn = conn
        self.embedding_function = embedding_function
        if hasattr(distance_strategy, "value"):
            normalized_distance = str(distance_strategy.value)
        else:
            normalized_distance = str(distance_strategy)
        self.distance_strategy = normalized_distance
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
            distance_strategy=normalized_distance,
        )

        self._workflow_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['workflow'],
            connection_string=connection_string,
            distance_strategy=normalized_distance,
        )

        self._toolbox_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['toolbox'],
            connection_string=connection_string,
            distance_strategy=normalized_distance,
        )

        self._entity_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['entity'],
            connection_string=connection_string,
            distance_strategy=normalized_distance,
        )

        self._summary_vs = PGVector(
            embedding_function=embedding_function,
            collection_name=table_names['summary'],
            connection_string=connection_string,
            distance_strategy=normalized_distance,
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

    def set_up_hybrid_search(self, preference_name=""):
        pass


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""
    name: str
    description: str
    signature: str
    parameters: dict
    return_type: str


class Toolbox:
    """
    A toolbox for registering, storing, and retrieving tools with LLM-powered augmentation.

    Tools are stored with embeddings for semantic retrieval, allowing the agent to
    find relevant tools based on natural language queries.
    """

    def __init__(self, memory_manager, llm_client, embedding_function, model: str = "gpt-5"):
        """
        Initialize the Toolbox.

        Args:
            memory_manager: MemoryManager instance for storing tools
            llm_client: LLM client for augmentation
            embedding_function: Embedding function/model for creating embeddings
            model: LLM model name
        """

        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.model = model
        self._tools: dict[str, Callable] = {}
        self._tools_by_name: dict[str, Callable] = {}

    def _get_embedding(self, text: str) -> list[float]:
        """
        Get the embedding for a text using the configured embedding function.
        """

        if hasattr(self.embedding_function, 'embed_query'):
            return self.embedding_function.embed_query(text)
        elif callable(self.embedding_function):
            return self.embedding_function(text)
        else:
            raise ValueError("embedding_function must be callable or have embed_query method")


    def _augment_docstring(
        self, docstring: str, source_code: str = ""
        ) -> str:
        """
        Use LLM to improve and expand a tool's docstring
        by analyzing both the original description and the
        function's source code.
        """

        if not docstring.strip() and not source_code.strip():
            return "No description provided"

        code_section = ""
        if source_code.strip():
            code_section = (
                "\n\nFunction source code:\n"
                f"```python\n{source_code}\n```"
            )

        prompt = (
            "You are a technical writer. "
            "Analyze the function's source code and its "
            "original docstring, then produce a richer, "
            "more detailed description. Include:\n"
            "1. A clear one-line summary\n"
            "2. What the function does step by step\n"
            "3. When an agent should call this function\n"
            "4. Important notes or caveats\n\n"
            f"Original docstring:\n{docstring}"
            f"{code_section}\n\n"
            "Return ONLY the improved docstring, "
            "no other text."
        )

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=2000,
        )

        return response.choices[0].message.content.strip()

    def _generate_queries(self, docstring: str, num_queries: int = 5) -> list[str]:
        """
        Generate synthetic example queries that would lead to using this tool.
        """

        prompt = f"""Based on the following tool description,
            generate {num_queries} diverse example queries
            that a user might ask when they need this tool. Make them natural and varied.

            Tool description:
            {docstring}

            Return ONLY a JSON array of strings, like: ["query1", "query2", ...]
        """

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000
        )

        try:
            queries = json_lib.loads(response.choices[0].message.content.strip())
            return queries if isinstance(queries, list) else []
        except json_lib.JSONDecodeError:
            # Fallback: extract queries from text
            return [response.choices[0].message.content.strip()]


    def _get_tool_metadata(self, func: Callable) -> ToolMetadata:
        """
        Extract metadata from a function for storage and retrieval.
        """
        sig = inspect.signature(func)

        # Extract parameter metadata with explicit fields.
        parameters = {}
        for name, param in sig.parameters.items():
            param_info = {"name": name}

            if param.annotation is not inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
            else:
                param_info["type"] = "string"

            if param.default is not inspect.Parameter.empty:
                # Store default as string for safe JSON serialization in metadata.
                param_info["default"] = str(param.default)

            parameters[name] = param_info

        # Extract return type.
        return_type = "Any"
        if sig.return_annotation is not inspect.Signature.empty:
            return_type = str(sig.return_annotation)

        description = inspect.getdoc(func) or "No description"

        return ToolMetadata(
            name=func.__name__,
            description=description,
            signature=str(sig),
            parameters=parameters,
            return_type=return_type,
        )

    def _tool_exists_in_db(self, tool_name: str) -> bool:
        """Check if a tool with the given name already exists in the toolbox store."""
        try:
            # Keep this DB check backend-agnostic by using vector-store metadata filters.
            results = self.memory_manager.toolbox_vs.similarity_search(
                query=tool_name,
                k=1,
                filter={"name": tool_name},
            )
            if not results:
                return False

            first = results[0]
            return (first.metadata or {}).get("name") == tool_name
        except Exception:
            return False

    def register_tool(
        self, func: Optional[Callable] = None, augment: bool = False
        ) -> Union[str, Callable]:
        """
        Register a function as a tool in the toolbox.

        If a tool with the same name already exists in the database,
        the callable is registered in memory but no duplicate row is
        written to the vector store.
        """

        def decorator(f: Callable) -> str:
            tool_name = f.__name__

            # In-memory dedupe for current process.
            if tool_name in self._tools_by_name:
                logger.info(f"Tool '{tool_name}' already registered in runtime")
                return tool_name

            # Deduplication: skip DB write if tool already stored
            if self._tool_exists_in_db(tool_name):
                self._tools_by_name[tool_name] = f
                logger.info(f"Tool '{tool_name}' already in toolbox (skipping DB write)")
                return tool_name

            docstring = inspect.getdoc(f) or f.__name__
            signature = str(inspect.signature(f))
            object_id = uuid.uuid4()
            object_id_str = str(object_id)

            if augment:
                # Use LLM to enhance the tool's discoverability
                try:
                    source_code = inspect.getsource(f)
                except (OSError, TypeError):
                    source_code = ""
                augmented_docstring = self._augment_docstring(
                    docstring, source_code
                )
                queries = self._generate_queries(augmented_docstring)

                # Create rich embedding text combining all information
                embedding_text = f"{f.__name__} {augmented_docstring} {signature} {' '.join(queries)}"
                embedding = self._get_embedding(embedding_text)

                tool_data = self._get_tool_metadata(f)
                tool_data.description = augmented_docstring  # Use augmented description

                tool_dict = {
                    "_id": object_id_str,  # Use string, not UUID object
                    "embedding": embedding,
                    "queries": queries,
                    "augmented": True,
                    **tool_data.model_dump(),
                }
            else:
                # Basic registration without augmentation
                embedding = self._get_embedding(f"{f.__name__} {docstring} {signature}")
                tool_data = self._get_tool_metadata(f)

                tool_dict = {
                    "_id": object_id_str,  # Use string, not UUID object
                    "embedding": embedding,
                    "augmented": False,
                    **tool_data.model_dump(),
                }

            # Store the tool in the toolbox memory for retrieval
            # The embedding enables semantic search to find relevant tools
            self.memory_manager.write_toolbox(
                f"{f.__name__} {docstring} {signature}",
                tool_dict
            )

            # Keep reference to the callable for execution
            self._tools[object_id_str] = f
            self._tools_by_name[f.__name__] = f  # Also store by name for easy lookup
            return object_id_str

        if func is None:
            return decorator
        return decorator(func)


# =============================================================================
# Context Window Management and Summarization
# =============================================================================


MODEL_TOKEN_LIMITS = {
    "gpt-5": 256000,
}


def calculate_context_usage(context: str, model: str = "gpt-5") -> dict:
    """Calculate context window usage as percentage."""
    estimated_tokens = len(context) // 4  # ~4 chars per token
    max_tokens = MODEL_TOKEN_LIMITS.get(model, 128000)
    percentage = (estimated_tokens / max_tokens) * 100
    return {"tokens": estimated_tokens, "max": max_tokens, "percent": round(percentage, 1)}


def monitor_context_window(context: str, model: str = "gpt-5") -> dict:
    """
    Monitor the current context window and return capacity utilization.

    Args:
        context: The current context string to measure
        model: The model being used (to determine max tokens)

    Returns:
        dict with tokens, max, percent, and status ('ok', 'warning', 'critical')
    """
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
