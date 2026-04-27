from openai import OpenAI
from langchain_community.vectorstores.utils import DistanceStrategy

from src.api.agent_runtime import MemoryAwareAgent
from src.config.settings import settings
from src.infrastructure.db.postgres import connect_to_postgres
from src.infrastructure.embedding.embedding_model import get_embedding_model
from src.memory.memory_manager import MemoryManager
from src.memory.store_manager import StoreManager
from src.memory.tool_box import Toolbox


CONVERSATIONAL_TABLE = "conversational_memory"
KNOWLEDGE_BASE_TABLE = "semantic_memory"
WORKFLOW_TABLE = "workflow_memory"
TOOLBOX_TABLE = "toolbox_memory"
ENTITY_TABLE = "entity_memory"
SUMMARY_TABLE = "summary_memory"
TOOL_LOG_TABLE = "tool_log_memory"


def build_agent() -> MemoryAwareAgent:
    """Build and return a memory-aware agent runtime from source modules."""
    conn = connect_to_postgres()
    embedding_model = get_embedding_model()
    llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    store_manager = StoreManager(
        conn=conn,
        embedding_function=embedding_model,
        table_names={
            "knowledge_base": KNOWLEDGE_BASE_TABLE,
            "workflow": WORKFLOW_TABLE,
            "toolbox": TOOLBOX_TABLE,
            "entity": ENTITY_TABLE,
            "summary": SUMMARY_TABLE,
        },
        distance_strategy=DistanceStrategy.COSINE,
        conversational_table=CONVERSATIONAL_TABLE,
        tool_log_table=TOOL_LOG_TABLE,
    )

    memory_manager = MemoryManager(
        conn=conn,
        conversation_table=store_manager.get_conversational_table(),
        knowledge_base_vs=store_manager.get_knowledge_base_store(),
        workflow_vs=store_manager.get_workflow_store(),
        toolbox_vs=store_manager.get_toolbox_store(),
        entity_vs=store_manager.get_entity_store(),
        summary_vs=store_manager.get_summary_store(),
        tool_log_table=store_manager.get_tool_log_table(),
    )

    toolbox = Toolbox(memory_manager=memory_manager, llm_client=llm_client, embedding_function=embedding_model)

    return MemoryAwareAgent(
        memory_manager=memory_manager,
        toolbox=toolbox,
        llm_client=llm_client,
        model="gpt-5-mini",
    )


def main():
    """Simple interactive loop for manual runtime checks."""
    agent = build_agent()
    thread_id = "local-dev"

    print("Memory-aware agent is ready. Type 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        answer = agent.call(query, thread_id=thread_id, verbose=True)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
