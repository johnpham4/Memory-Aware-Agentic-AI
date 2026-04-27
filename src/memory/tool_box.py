import inspect
import json as json_lib
import uuid
from typing import Callable, Optional, Union
from loguru import logger

from src.models.tool import ToolMetadata

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