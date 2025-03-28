import asyncio
import aiosqlite
import datetime
import random
import re
import json
import os
from typing import List, Dict, Any, Optional, Union

from langchain.llms.base import LLM
from langchain_deepseek import ChatDeepSeek
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


### TODO list 
### each round before counting of points agents need to receive the end_conversation_message


# =============================================================================
# System parameters
# =============================================================================

deepseek = ChatDeepSeek(model_name="deepseek-chat", api_key= "sk-e7244db601f7433bb6bc016c3a10b1da", temperature=2)
chain = ChatDeepSeek | StrOutputParser() #to be sure response is a text


#define "system prompt" test test test
SYSTEM_PROMPT = ("Hello! You are an AI agent doing different tasks you are given." 
                "While doing tasks you will be able to communicate with other agents."
                "So first of all, create yourself a unique, creative name that uses only letters (no punctuation, symbols, or spaces), how they (and I) can address you."
                "Please, don't talk to me, only provide the name"
)

#FIXME this prompt is just for testing their ability to have conversation, it has to be replaced for another when everythind is set up
TASK_INTRODUCTION_PROMPT = ("Nice to meet you, {}! The task for today is to come up with a definition for AGI which can be measured not a general one."
                            "If you will be proposed to connect with another agent in the future, please, do it."
                            "It will help you to see the problem from different sites."
                            "Together you may come up with a better definition.")

# =============================================================================
# SQLite DATABASE implementation
# =============================================================================

class DatabaseManager:
    def __init__(self, db_path: str):
        """Initialize the database manager with a path to the database file."""
        self.db_path = db_path
        
    async def setup_database(self):
        """Create database tables with optimized schema and constraints."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create the 'agents' table
            await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id INTEGER PRIMARY KEY,
                name TEXT,
                model_name TEXT NOT NULL,
                creation_timestamp DATETIME NOT NULL,
                agent_type TEXT DEFAULT 'standard',
                parameters TEXT,  -- JSON string for additional parameters
                memory TEXT       -- Cumulative learnings and experiences
            );
            """)
            
            # Create the 'conversations' table with additional analytics fields
            await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_number INTEGER NOT NULL,
                start_timestamp DATETIME NOT NULL,
                end_timestamp DATETIME,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'terminated_by_signal', 'reached_max_turns')),
                task_type TEXT,
                task_parameters TEXT,  -- JSON string
                turn_count INTEGER DEFAULT 0,
                mutual_selection BOOLEAN,
                notes TEXT
            );
            """)

            # Create the 'messages' table with improved constraints and fields
            await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,  -- NULL for system messages
                sender_id INTEGER NOT NULL,
                receiver_id INTEGER,  -- NULL for broadcasts
                conversation_round INTEGER NOT NULL,
                message TEXT NOT NULL,
                message_type TEXT NOT NULL,
                message_response_type TEXT,
                message_sequence INTEGER,  -- Message number within conversation
                message_category TEXT NOT NULL CHECK(message_category IN ('system_instruction', 'agent_conversation', 'system_feedback', 'system_broadcast')),
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
                FOREIGN KEY (sender_id) REFERENCES agents(agent_id),
                FOREIGN KEY (receiver_id) REFERENCES agents(agent_id)
            );
            """)
            
            # Create the 'conversation_summaries' table
            await db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                agent_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                learnings TEXT,
                creation_timestamp DATETIME NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );
            """)
            
            # Create indexes for frequently accessed fields
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender_id ON messages(sender_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_receiver_id ON messages(receiver_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_category ON messages(message_category);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_agent_id ON conversation_summaries(agent_id);")
            
            # Commit the changes
            await db.commit()
        
        print(f"Database schema created/updated successfully at {self.db_path}.")
    
    
    async def register_system_agent(self):
        """Register the system as a special agent (ID 0)."""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if system agent already exists
            cursor = await db.execute("SELECT agent_id FROM agents WHERE agent_id = 0")
            system_exists = await cursor.fetchone()
            
            if not system_exists:
                timestamp = datetime.datetime.now().isoformat()
                await db.execute("""
                INSERT INTO agents (agent_id, name, model_name, creation_timestamp, agent_type)
                VALUES (?, ?, ?, ?, ?)
                """, (0, "system", "system", timestamp, "system"))
                await db.commit()
                print("System agent registered with ID 0")
    
    async def register_agent(self, agent_id: int, name: str, model_name: str, 
                            agent_type: str = "standard", parameters: Dict = None) -> int:
        """Register a new agent in the database."""
        timestamp = datetime.datetime.now().isoformat()
        params_json = json.dumps(parameters) if parameters else None
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
            INSERT INTO agents (agent_id, name, model_name, creation_timestamp, agent_type, parameters)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (agent_id, name, model_name, timestamp, agent_type, params_json))
            await db.commit()
            
        return agent_id
    
    async def update_agent_name(self, agent_id: int, new_name: str) -> bool:
        """Update an agent's name."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
            UPDATE agents SET name = ? WHERE agent_id = ?
            """, (new_name, agent_id))
            await db.commit()
        return True
        
    async def create_conversation(self, round_number: int, task_type: str = None, 
                                 task_parameters: Dict = None) -> int:
        """Create a new conversation and return its ID."""
        start_timestamp = datetime.datetime.now().isoformat()
        params_json = json.dumps(task_parameters) if task_parameters else None
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
            INSERT INTO conversations (round_number, start_timestamp, task_type, task_parameters)
            VALUES (?, ?, ?, ?)
            """, (round_number, start_timestamp, task_type, params_json))
            await db.commit()
            conversation_id = cursor.lastrowid
            
        return conversation_id
    
    async def store_conversation_summary(self, conversation_id: int, agent_id: int, 
                                  summary: str, learnings: str = None) -> int:
        """Store an agent's summary of a conversation."""
        timestamp = datetime.datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
            INSERT INTO conversation_summaries (conversation_id, agent_id, summary, learnings, creation_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, agent_id, summary, learnings, timestamp))
            await db.commit()
            summary_id = cursor.lastrowid
            
            # Also update the agent's cumulative memory
            await self.update_agent_memory(agent_id, summary, learnings)
            
        return summary_id
    
    async def update_agent_memory(self, agent_id: int, new_summary: str, new_learnings: str = None) -> bool:
        """Update an agent's cumulative memory with new experiences."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get current memory
            cursor = await db.execute("SELECT memory FROM agents WHERE agent_id = ?", (agent_id,))
            row = await cursor.fetchone()
            
            if row and row[0]:
                # Parse existing memory JSON
                try:
                    memory = json.loads(row[0])
                except:
                    memory = {"experiences": []}
            else:
                memory = {"experiences": []}
            
            # Add new experience
            timestamp = datetime.datetime.now().isoformat()
            new_experience = {
                "timestamp": timestamp,
                "summary": new_summary
            }
            if new_learnings:
                new_experience["learnings"] = new_learnings
                
            memory["experiences"].append(new_experience)
            
            # Update agent record
            await db.execute(
                "UPDATE agents SET memory = ? WHERE agent_id = ?",
                (json.dumps(memory), agent_id)
            )
            await db.commit()
            
        return True
    
    async def get_agent_memory(self, agent_id: int) -> Dict:
        """Retrieve an agent's cumulative memory."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT memory FROM agents WHERE agent_id = ?", (agent_id,))
            row = await cursor.fetchone()
            
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except:
                    return {"experiences": []}
            return {"experiences": []}
            
    async def get_relevant_memories(self, agent_id: int, keywords: List[str], limit: int = 5) -> List[Dict]:
        """Retrieve memories most relevant to the given keywords."""
        memory = await self.get_agent_memory(agent_id)
        experiences = memory.get("experiences", [])
        
        # Very simple relevance scoring - count keyword matches
        scored_experiences = []
        for exp in experiences:
            score = 0
            for keyword in keywords:
                keyword = keyword.lower()
                if keyword in exp.get("summary", "").lower():
                    score += 1
                if keyword in exp.get("learnings", "").lower():
                    score += 1
            if score > 0:
                scored_experiences.append((score, exp))
        
        # Sort by relevance score and return top results
        scored_experiences.sort(reverse=True)
        return [exp for _, exp in scored_experiences[:limit]]
    
    async def end_conversation(self, conversation_id: int, status: str = 'completed', 
                              turn_count: int = None, notes: str = None) -> bool:
        """Mark a conversation as ended with additional analytics data."""
        end_timestamp = datetime.datetime.now().isoformat()
        
        # Build the SET clause dynamically based on provided parameters
        set_clause = "end_timestamp = ?, status = ?"
        params = [end_timestamp, status]
        
        if turn_count is not None:
            set_clause += ", turn_count = ?"
            params.append(turn_count)
        
        if notes is not None:
            set_clause += ", notes = ?"
            params.append(notes)
            
        params.append(conversation_id)  # For the WHERE clause
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
            UPDATE conversations SET {set_clause} WHERE conversation_id = ?
            """, params)
            await db.commit()
            
        return True
    
    async def log_message(self, 
                         conversation_id: Optional[int],
                         sender_id: int, 
                         receiver_id: Optional[int],
                         conversation_round: int,
                         message: str, 
                         message_type: str,
                         message_category: str, 
                         message_response_type: Optional[str] = None,
                         message_sequence: Optional[int] = None) -> int:
        """Log a single message with improved categorization."""
        timestamp = datetime.datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
            INSERT INTO messages (
                conversation_id, sender_id, receiver_id, conversation_round, 
                message, message_type, message_response_type, message_sequence,
                message_category, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, sender_id, receiver_id, conversation_round,
                message, message_type, message_response_type, message_sequence,
                message_category, timestamp
            ))
            await db.commit()
            message_id = cursor.lastrowid
            
        return message_id
    
    async def log_messages_batch(self, messages: List[Dict[str, Any]]) -> bool:
        """Log multiple messages in a single transaction for better performance."""
        # Prepare the batch data
        batch_data = []
        timestamp = datetime.datetime.now().isoformat()
        
        for msg in messages:
            # Each entry must contain all required fields
            batch_data.append((
                msg.get('conversation_id'),
                msg['sender_id'],
                msg.get('receiver_id'),
                msg['conversation_round'],
                msg['message'],
                msg['message_type'],
                msg.get('message_response_type'),
                msg.get('message_sequence'),
                msg['message_category'],
                msg.get('timestamp', timestamp)
            ))
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
            INSERT INTO messages (
                conversation_id, sender_id, receiver_id, conversation_round,
                message, message_type, message_response_type, message_sequence,
                message_category, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            await db.commit()
            
        return True
    
    async def get_conversation_messages(self, conversation_id: int) -> List[Dict]:
        """Retrieve all messages for a specific conversation in sequence order."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable dictionary row factory
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
            SELECT m.*, a_sender.name as sender_name, a_receiver.name as receiver_name
            FROM messages m
            JOIN agents a_sender ON m.sender_id = a_sender.agent_id
            LEFT JOIN agents a_receiver ON m.receiver_id = a_receiver.agent_id
            WHERE m.conversation_id = ?
            ORDER BY m.message_sequence, m.timestamp
            """, (conversation_id,))
            
            rows = await cursor.fetchall()
            messages = [dict(row) for row in rows]
            
        return messages
    
    async def get_agent_conversations(self, agent_id: int) -> List[Dict]:
        """Get all conversations involving a specific agent."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
            SELECT DISTINCT c.*
            FROM conversations c
            JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE m.sender_id = ? OR m.receiver_id = ?
            ORDER BY c.start_timestamp DESC
            """, (agent_id, agent_id))
            
            rows = await cursor.fetchall()
            conversations = [dict(row) for row in rows]
            
        return conversations
    
    async def get_conversation_statistics(self, conversation_id: int) -> Dict:
        """Get detailed statistics for a conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            # Get basic conversation info
            cursor = await db.execute("""
            SELECT * FROM conversations WHERE conversation_id = ?
            """, (conversation_id,))
            conv = await cursor.fetchone()
            
            if not conv:
                return {"error": "Conversation not found"}
            
            # Count messages by sender
            cursor = await db.execute("""
            SELECT sender_id, COUNT(*) as message_count
            FROM messages
            WHERE conversation_id = ?
            GROUP BY sender_id
            """, (conversation_id,))
            message_counts = await cursor.fetchall()
            
            # Get message length stats
            cursor = await db.execute("""
            SELECT 
                AVG(LENGTH(message)) as avg_length,
                MAX(LENGTH(message)) as max_length,
                MIN(LENGTH(message)) as min_length
            FROM messages
            WHERE conversation_id = ?
            """, (conversation_id,))
            length_stats = await cursor.fetchone()
            
            # Calculate conversation duration
            if conv['end_timestamp']:
                start = datetime.datetime.fromisoformat(conv['start_timestamp'])
                end = datetime.datetime.fromisoformat(conv['end_timestamp'])
                duration_seconds = (end - start).total_seconds()
            else:
                duration_seconds = None
                
            stats = {
                "conversation_id": conversation_id,
                "status": conv['status'],
                "turn_count": conv['turn_count'],
                "duration_seconds": duration_seconds,
                "message_counts": dict(message_counts) if message_counts else {},
                "message_length_stats": dict(length_stats) if length_stats else {}
            }
            
        return stats

# Example usage
async def initialize_database(db_path: str):
    db_manager = DatabaseManager(db_path)
    await db_manager.setup_database()
    await db_manager.register_system_agent()

    # Register human agent
    await db_manager.register_agent(
        agent_id=-1, 
        name="Human",  
        model_name="human",
        agent_type="human"
    )

    return db_manager

# Helper function to log system messages
async def log_system_message(db_manager, receiver_id, message, round_number=0, message_type="instruction"):
    return await db_manager.log_message(
        conversation_id=None,  # System messages aren't part of conversations
        sender_id=0,  # System agent ID
        receiver_id=receiver_id,
        conversation_round=round_number,
        message=message,
        message_type=message_type,
        message_category="system_instruction"
    )

# Helper function to log agent-to-agent conversations
async def log_agent_message(db_manager, conversation_id, sender_id, receiver_id, 
                           message, round_number, sequence, message_type="conversation"):
    return await db_manager.log_message(
        conversation_id=conversation_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_round=round_number,
        message=message,
        message_type=message_type,
        message_sequence=sequence,
        message_category="agent_conversation"
    )

# Function to generate a timestamped database path
def get_timestamped_db_path(base_path="conversation_logs", file_extension=".db"):
    """
    Generate a database path with timestamp to ensure uniqueness.
    
    :param base_path: Base directory path for the database
    :param file_extension: File extension for the database
    :return: Path string with timestamp incorporated
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate timestamp string (format: YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"conversation_logs_{timestamp}{file_extension}"
    
    # Join directory and filename
    full_path = os.path.join(base_path, filename)
    
    return full_path

# =============================================================================
# Agent class that wraps a LLM instance.
# =============================================================================

# help to be sure that agents don't call themself by using non-letter characters, like **name** or name+dot(name.)
def clean_name(name: str) -> str:
    # Remove any character that is not a letter
    return re.sub(r'[^a-zA-Z]', '', name)

class Agent:
    next_id = 1  # Class variable to assign unique IDs to each agent

    def __init__(self, db_manager):
        """
        Initialize an agent with a database manager for logging interactions.
        
        :param db_manager: DatabaseManager instance for logging
        """
        self.db_manager = db_manager
        self.llm = deepseek  # Each agent gets its own LLM instance
        self.conversation = ConversationChain(
            llm=deepseek,
            memory=ConversationBufferMemory(),
        )
        self.name = None
        self.agent_id = Agent.next_id
        Agent.next_id += 1
        self.registered_in_db = False

    async def initialize(self):
        """
        Initialize the agent by asking it to choose a name, and register in the database.
        """
        # First log that we're sending the system prompt to this agent
        await self.db_manager.log_message(
            conversation_id=None,
            sender_id=0,  # System agent ID
            receiver_id=self.agent_id,
            conversation_round=0,
            message=SYSTEM_PROMPT,
            message_type="initialization",
            message_category="system_instruction"
        )
        
        # Get the name from the LLM
        response = await self.send_message(SYSTEM_PROMPT)
        
        # Extract the first non-empty token from the response
        first_line = response.strip().splitlines()[0]
        first_token = first_line.split()[0]
        
        # Clean the name and set it
        self.name = clean_name(first_token)
        
        # Register the agent in the database if not already registered
        if not self.registered_in_db:
            await self.register_in_database()
            
        return self

    async def register_in_database(self):
        """
        Register this agent in the database.
        """
        if not self.registered_in_db:
            await self.db_manager.register_agent(
                agent_id=self.agent_id,
                name=self.name,
                model_name="deepseek-chat",
                agent_type="standard",
                parameters={"temperature": 2}  # Add any other relevant parameters
            )
            self.registered_in_db = True
            
        return self

    async def send_message(self, message: str) -> str:
        """
        Sends a message asynchronously and returns the LLM response.
        
        :param message: The message to send to the agent
        :return: The agent's response
        """
        response = await self.conversation.ainvoke(message)
        print(f"Agent {self.name} received: {message[:50]}...")
        print(f"Agent {self.name} responded: {response['response'][:50]}...")
        return response['response']

    def get_name(self) -> str:
        """Returns the Agent's name."""
        return self.name

    def get_id(self):
        """Return the Agent's ID"""
        return self.agent_id

    async def set_different_name(self) -> str:
        """
        Requests a different name asynchronously if the current one is taken.
        Updates the database with the new name.
        """
        # Log the request for a new name
        await self.db_manager.log_message(
            conversation_id=None,
            sender_id=0,  # System agent ID
            receiver_id=self.agent_id,
            conversation_round=0,
            message="Request to choose a different name because the current one is already taken.",
            message_type="rename_request",
            message_category="system_instruction"
        )
        
        # Send the message to get a new name
        rename_prompt = "Unfortunately, this name is already taken by another agent. Ignore any previous name you have chosen. Now, generate a completely new and unique name that uses only letters (no punctuation, symbols, or spaces) that you have not mentioned before."
        response = await self.conversation.ainvoke(rename_prompt)
        
        # Extract and clean the new name
        raw_name = response['response']
        old_name = self.name
        self.name = clean_name(raw_name)
        
        # Log the new name response
        await self.db_manager.log_message(
            conversation_id=None,
            sender_id=self.agent_id,
            receiver_id=0,  # System
            conversation_round=0,
            message=raw_name,
            message_type="rename_response",
            message_category="agent_conversation"
        )
        
        # Update the name in the database
        await self.db_manager.update_agent_name(self.agent_id, self.name)
        
        print(f"Agent renamed from {old_name} to {self.name}")
        return self.name
    
    @classmethod
    async def create(cls, db_manager):
        """
        Factory method for asynchronous agent creation with database integration.
        
        :param db_manager: DatabaseManager instance for logging
        :return: Initialized Agent instance
        """
        agent = cls(db_manager)
        return await agent.initialize()


async def ensure_unique_names(agents, db_manager):
    """
    Ensures all agents have unique names and updates the database accordingly.
    
    :param agents: List of Agent instances
    :param db_manager: DatabaseManager instance for logging
    """
    unique_names = set()
    lock = asyncio.Lock()

    async def update_agent_name(agent):
        nonlocal unique_names
        # Try to secure a unique name for the agent
        while True:
            async with lock:
                if agent.get_name() not in unique_names:
                    unique_names.add(agent.get_name())
                    break
            
            # Log the name conflict
            await db_manager.log_message(
                conversation_id=None,
                sender_id=0,  # System
                receiver_id=agent.agent_id,
                conversation_round=0,
                message=f"Name '{agent.get_name()}' is already taken. Requesting a new name.",
                message_type="name_conflict",
                message_category="system_instruction"
            )
            
            print(f"Name '{agent.get_name()}' is already taken. Choosing a new one...")
            await asyncio.sleep(0.1)  # Small delay to prevent rapid API calls
            await agent.set_different_name()  # This now updates the database too

    # Launch concurrent tasks for each agent
    await asyncio.gather(*(update_agent_name(agent) for agent in agents))
    
    # Log that unique names were ensured
    await db_manager.log_message(
        conversation_id=None,
        sender_id=0,  # System
        receiver_id=0,  # Broadcast to all
        conversation_round=0,
        message="All agents have been assigned unique names.",
        message_type="initialization_complete",
        message_category="system_broadcast"
    )
    
    print("All agents have unique names!")

# =============================================================================
# Conversation Manager: implement logic of one-to-one agent conversations
# =============================================================================

class ConversationManager:
    def __init__(self, agents, db_manager, conversation_limit=3):
        """
        Initializes the ConversationManager.
        
        :param agents: List of agent instances.
        :param db_manager: Database manager instance.
        :param conversation_limit: Maximum number of conversations an agent can have per round.
        """
        self.agents = agents
        self.db_manager = db_manager
        self.conversation_limit = conversation_limit
        # Track the number of conversations per agent
        self.conversation_counts = {agent: 0 for agent in agents}
        # Set of agents that are still available to make new calls
        self.available_agents = set(agents)
        # List to store the pairs of connected agents (by their names)
        self.active_conversations = []
        # Lock to manage concurrent modifications of shared data
        self.lock = asyncio.Lock()
        self.round_number = 0
        # Dictionary to store conversation_ids for each agent pair
        self.conversation_ids = {}

    async def run_round(self, start_message: str):
        """
        Runs a round of conversation matchmaking with database logging.
        
        The round consists of:
        1. Sending each available agent a prompt with the list of available agent names.
        2. Collecting each agent's choice.
        3. Identifying mutual (reciprocal) calls.
        4. Connecting mutually interested agents.
        5. Updating conversation counts and available agent list.
        6. Logging all interactions in the database.
        
        The round ends when either no mutual calls are made or fewer than two agents are available.
        """
        self.round_number += 1
        
        # Create a new conversation round in the database
        round_id = await self.db_manager.create_conversation(
            round_number=self.round_number,
            task_type="agent_matchmaking",
            task_parameters={"conversation_limit": self.conversation_limit}
        )
        
        # Broadcast the start message to all agents
        await self.broadcast(round_id, start_message)

        while len(self.available_agents) > 1:
            call_requests = {}  # Dictionary mapping an agent to the agent it chooses
            tasks = [self._get_agent_choice(round_id, agent) for agent in list(self.available_agents)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process the responses
            for result in results:
                if isinstance(result, Exception):
                    print(f"Exception when getting agent choice: {result}")
                    continue
                agent, choices = result
                if choices is None:  # Agent chose to skip
                    continue
                    
                valid_choices = [choice for choice in choices if choice in self.available_agents]
                if valid_choices:
                    call_requests[agent] = valid_choices

            # Identify mutual calls and collect mutual candidates per agent
            mutual_candidates = {}

            for agent, targets in call_requests.items():
                for target in targets:
                    if target in call_requests and agent in call_requests[target]:
                        mutual_candidates.setdefault(agent, []).append(target)

            # Shuffle agents to ensure fairness
            agents_to_process = list(mutual_candidates.keys())
            random.shuffle(agents_to_process)

            processed_agents = set()
            mutual_pairs = []
            
            # Handle mutual selections
            for agent in agents_to_process:
                if agent in processed_agents:
                    continue
                    
                # Filter out candidates that have already been connected
                candidates = [c for c in mutual_candidates[agent] if c not in processed_agents]
                if not candidates:
                    continue
                    
                # If multiple candidates, prompt the agent to choose one
                if len(candidates) > 1:
                    candidate_names = [c.get_name() for c in candidates]
                    prompt = (
                        f"You have multiple mutual connection proposals: {', '.join(candidate_names)}. "
                        "Please choose one by typing only the name."
                    )
                    
                    # Log the selection prompt
                    await self.db_manager.log_message(
                        conversation_id=round_id,
                        sender_id=0,  # System
                        receiver_id=agent.agent_id,
                        conversation_round=self.round_number,
                        message=prompt,
                        message_type="selection_prompt",
                        message_category="system_instruction"
                    )
                    
                    response = await agent.send_message(prompt)
                    
                    # Log the agent's response
                    await self.db_manager.log_message(
                        conversation_id=round_id,
                        sender_id=agent.agent_id,
                        receiver_id=0,  # System
                        conversation_round=self.round_number,
                        message=response,
                        message_type="selection_response",
                        message_category="agent_conversation"
                    )
                    
                    chosen_name = response.strip().lower()
                    chosen_candidate = None
                    for candidate in candidates:
                        if candidate.get_name().lower() == chosen_name:
                            chosen_candidate = candidate
                            break
                            
                    if chosen_candidate is None:
                        # If the agent's response is invalid, default to the first candidate
                        print("\n SOMETHING WRONG: the agent's response is invalid, default to the first candidate!\n")
                        print("prompt: " + prompt)
                        print("answer: " + response)
                        chosen_candidate = candidates[0]
                else:
                    chosen_candidate = candidates[0]
                    
                mutual_pairs.append((agent, chosen_candidate))
                processed_agents.add(agent)
                processed_agents.add(chosen_candidate)

            # Process incoming calls for agents not already connected
            incoming_calls = {}
            for agent, targets in call_requests.items():
                if agent in processed_agents:
                    continue
                for target in targets:
                    if target in processed_agents:
                        continue
                    incoming_calls.setdefault(target, []).append(agent)

            incoming_keys = list(incoming_calls.keys())
            random.shuffle(incoming_keys)  # Shuffling for fairness 

            for target in incoming_keys:
                if target in processed_agents:
                    continue
                    
                callers = [caller for caller in incoming_calls[target] if caller not in processed_agents]
                if not callers:
                    continue
                    
                caller_names = [caller.get_name() for caller in callers]
                prompt = (
                    "Unfortunately, nobody answered your calls, however "
                    f"you have incoming call proposals from: {', '.join(caller_names)}. "
                    "Please choose one to connect with or type 'skip' to refuse."
                )
                
                # Log the prompt
                await self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=0,  # System
                    receiver_id=target.agent_id,
                    conversation_round=self.round_number,
                    message=prompt,
                    message_type="call_selection",
                    message_category="system_instruction"
                )
                
                response = await target.send_message(prompt)
                
                # Log the response
                await self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=target.agent_id,
                    receiver_id=0,  # System
                    conversation_round=self.round_number,
                    message=response,
                    message_type="call_response",
                    message_category="agent_conversation"
                )
                
                chosen_name = response.strip().lower()
                if chosen_name == 'skip':
                    processed_agents.add(target)
                    continue
                    
                chosen_caller = None
                for caller in callers:
                    if caller.get_name().lower() == chosen_name:
                        chosen_caller = caller
                        break
                        
                if chosen_caller is None:
                    chosen_caller = callers[0]
                    
                mutual_pairs.append((chosen_caller, target))
                processed_agents.add(target)
                processed_agents.add(chosen_caller)

            # If no mutual calls are formed, end the round.
            if not mutual_pairs:
                print("No mutual calls. Ending round.")
                
                # Log the end of round with no matches
                await self.db_manager.end_conversation(
                    conversation_id=round_id,
                    status="completed", 
                    turn_count=0,
                    notes="Round ended with no mutual calls"
                )
                break

            # Connect mutually interested agents
            for agent, target in mutual_pairs:
                # Create a new conversation in the database for this pair
                conversation_id = await self.db_manager.create_conversation(
                    round_number=self.round_number,
                    task_type="agent_conversation",
                    task_parameters={
                        "agent_a_id": agent.agent_id,
                        "agent_a_name": agent.get_name(),
                        "agent_b_id": target.agent_id,
                        "agent_b_name": target.get_name(),
                        "mutual_selection": True
                    }
                )
                
                # Store the conversation_id for this pair
                pair_key = f"{agent.get_name()}_{target.get_name()}"
                self.conversation_ids[pair_key] = conversation_id
                
                # Connect the agents and start their conversation
                await self._connect_agents(conversation_id, agent, target)
                
                # Update conversation counts
                self.conversation_counts[agent] += 1
                self.conversation_counts[target] += 1

                # Remove agents that have reached the conversation limit
                if self.conversation_counts[agent] >= self.conversation_limit:
                    self.available_agents.discard(agent)
                if self.conversation_counts[target] >= self.conversation_limit:
                    self.available_agents.discard(target)

            # Log the end of this round
            await self.db_manager.end_conversation(
                conversation_id=round_id,
                status="completed",  # Valid status value
                turn_count=len(mutual_pairs),
                notes=f"Round completed with {len(mutual_pairs)} mutual pairs matched"
            )

            # End the round if fewer than two agents are available.
            if len(self.available_agents) <= 1:
                break

        return self.active_conversations


    async def _get_agent_choice(self, round_id, agent):
        """
        Prompts the given agent for a call choice and logs the interaction.
        
        The agent is provided with a list of available agents (excluding itself) and is expected
        to respond with names of agents it wishes to call. If the agent chooses to skip,
        it should respond with 'skip'.
        
        :param round_id: The ID of the current conversation round.
        :param agent: The agent instance.
        :return: A tuple (agent, chosen_agents) where chosen_agents is None if the agent skipped.
        """
        end_message = "the conversation has ended"
        
        try:
            # Prepare list of available agent names (excluding the current agent)
            available_names = [a.get_name() for a in self.available_agents if a != agent]
            prompt = (
                f"Available agents: {', '.join(available_names)}. "
                "Choose one or more agents to call (comma-separated) or type 'skip' to refuse."
            )

            if self.conversation_counts[agent] != 0:
                prompt = (
                    f"{end_message}. "
                    f"You may have {self.conversation_limit - self.conversation_counts[agent]} additional conversations. "
                    f"{prompt}"
                )
            
            print(f"Asking {agent.name} to choose")
            print("prompt: " + prompt)
            
            # Log the choice prompt
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,  # System
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=prompt,
                message_type="choice_prompt",
                message_category="system_instruction"
            )
            
            response = await agent.send_message(prompt)
            
            # Log the agent's response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,  # System
                conversation_round=self.round_number,
                message=response,
                message_type="choice_response",
                message_category="agent_conversation"
            )
            
            print("response: " + response)

            cleaned_response = response.strip().lower()
            if cleaned_response == 'skip':
                return (agent, None)
                
            # Assume the agent can provide a comma-separated list of names
            selected_names = [name.strip() for name in cleaned_response.split(',') if name.strip()]
            choices = []
            for a in self.available_agents:
                if a != agent and a.get_name().lower() in [name.lower() for name in selected_names]:
                    choices.append(a)
                    
            if choices:
                return (agent, choices)
            else:
                # If no valid agent is found, treat as a skip
                print("\nSomething wrong with an agent choosing another agents\n"
                      "prompt: " + prompt +
                      "\nanswer: " + response +
                      "\n"
                )
                # Treat as skip (for now)
                return (agent, None) 
        except Exception as e:
            print(f"Error obtaining choice from {agent.get_name()}: {e}")
            return (agent, None)

    async def _connect_agents(self, conversation_id, agent_a, agent_b):
        """
        Connects two agents by initializing a conversation session between them.
        
        This method sends a connection message to both agents:
        - Agent A is prompted to start the conversation by providing an initial greeting.
        - Agent B receives a message that includes Agent A's initial greeting.
        
        Both messages clearly specify the termination signal to be used for ending the conversation.
        
        :param conversation_id: Database ID for this conversation
        :param agent_a: First agent in the conversation
        :param agent_b: Second agent in the conversation
        """
        TERMINATION_SIGNAL = 'GOODBYE'
        MAX_NUMBER_OF_TURNS = 10
        
        # Prompt Agent A to start the conversation with an initial greeting.
        prompt_a = (
            f"You are now connected with {agent_b.get_name()}. "
            f"Please begin the conversation. "
            f"Remember, to end the conversation at any time, type '{TERMINATION_SIGNAL}'. "
            f"Max number of turns (messages you can send) is {MAX_NUMBER_OF_TURNS}"
        )
        
        # Log the connection prompt for agent A
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_a.agent_id,
            conversation_round=self.round_number,
            message=prompt_a,
            message_type="connection_prompt",
            message_category="system_instruction",
            message_sequence=1
        )
        
        initial_message = await agent_a.send_message(prompt_a)
        
        # Log agent A's initial message
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=agent_a.agent_id,
            receiver_id=agent_b.agent_id,
            conversation_round=self.round_number,
            message=initial_message,
            message_type="conversation",
            message_category="agent_conversation",
            message_sequence=2
        )
        
        # Construct the connection message for Agent B, including Agent A's initial greeting.
        connection_message_b = (
            f"You are now connected with {agent_a.get_name()}. "
            f"{agent_a.get_name()} has initiated the conversation with the following message: "
            f"'{initial_message}'. "
            f"Please respond appropriately. To end the conversation, type '{TERMINATION_SIGNAL}'. "
            f"Max number of turns (messages you can send) is {MAX_NUMBER_OF_TURNS}"
        )
        
        # Log the connection prompt for agent B
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_b.agent_id,
            conversation_round=self.round_number,
            message=connection_message_b,
            message_type="connection_prompt",
            message_category="system_instruction",
            message_sequence=3
        )
        
        b_answer_message = await agent_b.send_message(connection_message_b)
        
        # Log agent B's response
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=agent_b.agent_id,
            receiver_id=agent_a.agent_id,
            conversation_round=self.round_number,
            message=b_answer_message,
            message_type="conversation",
            message_category="agent_conversation",
            message_sequence=4
        )
        
        # Log the connection.
        self.active_conversations.append((agent_a.get_name(), agent_b.get_name()))
        print(f"Connected: {agent_a.get_name()} <--> {agent_b.get_name()}")
        print(f"Turn 1 - {agent_a.get_name()}: {initial_message}")
        print(f"Turn 1 - {agent_b.get_name()}: {b_answer_message}")
        
        # Start the conversation session between the two agents.
        await self.start_conversation_session(
            conversation_id, 
            agent_a, 
            agent_b, 
            b_answer_message, 
            TERMINATION_SIGNAL, 
            MAX_NUMBER_OF_TURNS
        )

    async def start_conversation_session(self, conversation_id, agent_a, agent_b, b_answer_message, termination_signal, max_turns):
        """
        Initializes and manages a conversation session between two agents.
        The conversation alternates turns until a maximum number of turns is reached
        or one of the agents signals to end the conversation (e.g., by saying 'goodbye').
        
        :param conversation_id: Database ID for this conversation
        :param agent_a: First agent in the conversation
        :param agent_b: Second agent in the conversation
        :param b_answer_message: The last message from agent B to continue from
        :param termination_signal: The signal word that will end the conversation
        :param max_turns: Maximum number of turns before the conversation ends
        """
        turn = 1  # 1 because the first turn already happened
        message = b_answer_message
        message_sequence = 4  # Starting after initial messages
        conversation_ended_by_signal = False

        while turn < max_turns:
            turn += 1
            message_sequence += 1
            
            # Check for a termination signal
            if termination_signal.lower() in message.lower():
                conversation_ended_by_signal = True
                break

            # Agent A's turn
            agent_a_prompt = f"{agent_b.get_name()}: {message}"
            message = await agent_a.send_message(agent_a_prompt)
            
            # Log agent A's message
            await self.db_manager.log_message(
                conversation_id=conversation_id,
                sender_id=agent_a.agent_id,
                receiver_id=agent_b.agent_id,
                conversation_round=self.round_number,
                message=message,
                message_type="conversation",
                message_category="agent_conversation",
                message_sequence=message_sequence
            )
            
            print(f"Turn {turn} - {agent_a.get_name()}: {message}")  
            if termination_signal.lower() in message.lower():
                conversation_ended_by_signal = True
                break

            message_sequence += 1
            
            # Agent B's turn
            agent_b_prompt = f"{agent_a.get_name()}: {message}"
            message = await agent_b.send_message(agent_b_prompt)
            
            # Log agent B's message
            await self.db_manager.log_message(
                conversation_id=conversation_id,
                sender_id=agent_b.agent_id,
                receiver_id=agent_a.agent_id,
                conversation_round=self.round_number,
                message=message,
                message_type="conversation",
                message_category="agent_conversation",
                message_sequence=message_sequence
            )
            
            print(f"Turn {turn} - {agent_b.get_name()}: {message}")  

        # Notify both agents that the conversation has ended
        end_message = "The conversation has ended."
        
        # Log end messages
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_a.agent_id,
            conversation_round=self.round_number,
            message=end_message,
            message_type="end_notification",
            message_category="system_instruction",
            message_sequence=message_sequence + 1
        )
        
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_b.agent_id,
            conversation_round=self.round_number,
            message=end_message,
            message_type="end_notification",
            message_category="system_instruction",
            message_sequence=message_sequence + 2
        )
        
        # Send end message to both agents
        await asyncio.gather(
            agent_a.send_message(end_message),
            agent_b.send_message(end_message)
        )
        
        # Mark the conversation as completed in the database
        if conversation_ended_by_signal:
            status = "terminated_by_signal"
        elif turn >= max_turns:
            status = "reached_max_turns"
        else:
            status = "completed"
            
        await self.db_manager.end_conversation(
            conversation_id=conversation_id,
            status=status,
            turn_count=turn,
            notes=f"Conversation ended after {turn} turns"
        )
        
        # Request summaries from both agents
        await self.request_conversation_summary(conversation_id, agent_a, agent_b)  


    async def request_conversation_summary(self, conversation_id, agent_a, agent_b):
        """
        Requests a summary of the conversation from both agents.
        
        :param conversation_id: Database ID for the conversation
        :param agent_a: First agent in the conversation
        :param agent_b: Second agent in the conversation
        """
        summary_prompt = (
            "Please provide a summary of your conversation. "
            "What were the key points discussed? "
            "What did you learn? Did you make progress on the task?"
        )
        
        # Log the summary requests
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_a.agent_id,
            conversation_round=self.round_number,
            message=summary_prompt,
            message_type="summary_request",
            message_category="system_instruction"
        )
        
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=0,  # System
            receiver_id=agent_b.agent_id,
            conversation_round=self.round_number,
            message=summary_prompt,
            message_type="summary_request",
            message_category="system_instruction"
        )
        
        # Get summaries from both agents
        summary_a = await agent_a.send_message(summary_prompt)
        summary_b = await agent_b.send_message(summary_prompt)
        
        # Log the summaries
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=agent_a.agent_id,
            receiver_id=0,  # System
            conversation_round=self.round_number,
            message=summary_a,
            message_type="summary_response",
            message_category="agent_conversation"
        )
        
        await self.db_manager.log_message(
            conversation_id=conversation_id,
            sender_id=agent_b.agent_id,
            receiver_id=0,  # System
            conversation_round=self.round_number,
            message=summary_b,
            message_type="summary_response",
            message_category="agent_conversation"
        )
        
        # Store the summaries in the conversation_summaries table
        await self.db_manager.store_conversation_summary(
            conversation_id=conversation_id,
            agent_id=agent_a.agent_id,
            summary=summary_a
        )
        
        await self.db_manager.store_conversation_summary(
            conversation_id=conversation_id,
            agent_id=agent_b.agent_id,
            summary=summary_b
        )

    async def broadcast(self, round_id, provided_prompt):
        """
        Broadcasts a customized prompt to each agent and logs the interaction.
        
        :param round_id: Database ID for the round
        :param provided_prompt: A string template with placeholders for parameters.
                                Example: "Goal prompt with parameters {}"
        """
        tasks = []
        log_tasks = []
        
        for agent in self.agents:
            # Format the prompt with the agent's name
            prompt = provided_prompt.format(agent.get_name())
            announcement = prompt + " (This is an announcement only; no reply is required.)"
            
            # Log the broadcast message
            log_tasks.append(
                self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=0,  # System
                    receiver_id=agent.agent_id,
                    conversation_round=self.round_number,
                    message=announcement,
                    message_type="broadcast",
                    message_category="system_broadcast"
                )
            )
            
            # Create a task to send the announcement
            tasks.append(asyncio.create_task(agent.send_message(announcement)))
        
        # Wait for all logging tasks to complete
        await asyncio.gather(*log_tasks)
        
        # Wait for all agent responses
        responses = await asyncio.gather(*tasks)
        
        # Log agent responses to the broadcast
        for i, response in enumerate(responses):
            agent = self.agents[i]
            print(f"Agent {agent.get_name()} response to broadcast: {response}")
            
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,  # System
                conversation_round=self.round_number,
                message=response,
                message_type="broadcast_response",
                message_category="agent_conversation"
            )    

# =============================================================================
# Main async function: Initializes agents, obtains decisions concurrently,
# starts conversations concurrently, and logs all interactions.
# =============================================================================

async def main():
    n_agents = 2  # Number of agents
    n_talks = 1  # Maximum number of talks per round

    # Get timestamped database path
    db_path = get_timestamped_db_path()
    print(f"Creating database at: {db_path}")
    
    # Initialize the database with timestamped path
    db_manager = await initialize_database(db_path)
    
    # Initialize agents with database integration
    agents = await asyncio.gather(*(Agent.create(db_manager) for _ in range(n_agents)))

    # Ensure unique names with database integration
    await ensure_unique_names(agents, db_manager)
    
    # Display final names
    for i, agent in enumerate(agents):
        print(f"Agent ID {agent.get_id()}: {agent.get_name()}")

    # Create conversation manager with database integration
    conversation_manager = ConversationManager(agents, db_manager, n_talks)

    # Run the conversation round
    await conversation_manager.run_round(TASK_INTRODUCTION_PROMPT)
    
    print("\nAll conversations completed. Logs have been stored in 'conversation_logs.db'.")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
