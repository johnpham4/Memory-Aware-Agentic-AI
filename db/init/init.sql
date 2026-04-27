-- Conversational memory table
CREATE TABLE IF NOT EXISTS conversational_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary_id VARCHAR(100) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversational_memory_thread_id
    ON conversational_memory(thread_id);

CREATE INDEX IF NOT EXISTS idx_conversational_memory_timestamp
    ON conversational_memory(timestamp);

-- Tool log memory table
CREATE TABLE IF NOT EXISTS tool_log_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id VARCHAR(100) NOT NULL,
    tool_call_id VARCHAR(100),
    tool_name VARCHAR(255) NOT NULL,
    tool_args JSONB NOT NULL,
    result TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'success',
    error_message TEXT,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_log_memory_thread_id
    ON tool_log_memory(thread_id);

CREATE INDEX IF NOT EXISTS idx_tool_log_memory_timestamp
    ON tool_log_memory(timestamp);
