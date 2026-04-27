
from langchain_community.vectorstores import PGVector

from src.config.settings import settings


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

    def set_up_hybrid_search(self, preference_name=""):
        pass