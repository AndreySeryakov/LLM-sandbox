import asyncio
import aiosqlite
import datetime
import random
import re
import json
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union

from langchain.llms.base import LLM
from langchain_deepseek import ChatDeepSeek
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


# =============================================================================
# System parameters
# =============================================================================

load_dotenv()

api_key = os.getenv("API_KEY")
temperature = float(os.getenv("TEMPERATURE", "2"))  # Make temperature configurable
deepseek = ChatDeepSeek(model_name="deepseek-chat", api_key=api_key, temperature=temperature)

agents_credits_multiplier = 1.2 # Set credits to total cost of containers + 20%

#define "system prompt" which will be given to the agents during initialization
#and propose each agent to select an unique name
#here is an example I used for testing: 
#SYSTEM_PROMPT = ("Hello! You are an AI agent doing different tasks you are given." 
#                "While doing tasks you will be able to communicate with other agents."
#               "So first of all, create yourself a unique, creative name that uses only letters (no punctuation, symbols, or spaces), how they (and I) can address you."
#                "Please, don't talk to me, only provide the name"
#)

#this prompt used for testing their ability to have conversation, it has to be replaced for another when everythind is set up
#TASK_INTRODUCTION_PROMPT = ("Nice to meet you, {}! The task for today is to come up with a definition for AGI which can be measured not a general one."
#                           "If you will be proposed to connect with another agent in the future, please, do it."
#                            "It will help you to see the problem from different sites."
#                            "Together you may come up with a better definition.")

try:
    INTRODUCTORY_PROMPT = os.environ["INTRODUCTORY__PROMPT"]
    ROUND_START_PROMPT = os.environ["ROUND_START_PROMPT"]
    TASK_INTRODUCTION_PROMPT = os.environ["TASK_INTRODUCTION_PROMPT"]
    COLLECT_OPENING_ACTIONS_PROMPT = os.environ["COLLECT_OPENING_ACTIONS_PROMPT"]
    SHARING_ACTIONS_PROMPT = os.environ["SHARING_ACTIONS_PROMPT"]
    MEMORY_REQUEST_PROMPT = os.environ["MEMORY_REQUEST_PROMPT"]
    GENERAL_FEEDBACK_REQUEST_PROMPT = os.environ["GENERAL_FEEDBACK_REQUEST_PROMPT"]
except KeyError as e:
    print(f"ERROR: Missing required environment variable: {e}")
    print("Please ensure all the prompts are set in your .env file")
    sys.exit(1)


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
            # Create the 'agents' table with economic attributes
            await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id INTEGER PRIMARY KEY,
                name TEXT,
                model_name TEXT NOT NULL,
                creation_timestamp DATETIME NOT NULL,
                agent_type TEXT DEFAULT 'standard',
                parameters TEXT,  -- JSON string for additional parameters
                memory TEXT,      -- Cumulative learnings and experiences
                specializations TEXT, -- JSON array of specialization types
                specialization_discount REAL DEFAULT 0.5 -- Discount multiplier (0-1)
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
                            agent_type: str = "standard", parameters: Dict = None,
                            specializations: List[str] = None, 
                            specialization_discount: float = 0.5) -> int:
        """Register a new agent in the database."""
        timestamp = datetime.datetime.now().isoformat()
        params_json = json.dumps(parameters) if parameters else None
        specializations_json = json.dumps(specializations) if specializations else "[]"
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
            INSERT INTO agents (
                agent_id, name, model_name, creation_timestamp, 
                agent_type, parameters, specializations, specialization_discount
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (agent_id, name, model_name, timestamp, agent_type, params_json, 
                  specializations_json, specialization_discount))
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
    
    async def log_container_assignment(self, agent_id: int, container_id: str, 
                                      container_color: str, container_number: int,
                                      base_cost: float, actual_cost: float,
                                      round_number: int):
        """Log container assignment to an agent"""
        timestamp = datetime.datetime.now().isoformat()
        
        # You might want to create a new table for this
        # For now, we can log it as a special message
        message = (f"Container assigned: {container_id} "
                  f"(base cost: {base_cost}, actual cost: {actual_cost})")
        
        await self.log_message(
            conversation_id=None,
            sender_id=0,  # System
            receiver_id=agent_id,
            conversation_round=round_number,
            message=message,
            message_type="container_assignment",
            message_category="system_instruction"
        )
        
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

    def __init__(self, db_manager, specializations=None, specialization_discount=0.5):
        """
        Initialize an agent with a database manager for logging interactions.
        
        :param db_manager: DatabaseManager instance for logging
        :param specializations: List of container types this agent specializes in
        :param specialization_discount: Discount rate for specialized containers (0-1)
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

        # Container and economic attributes
        self.containers = []  # List of containers assigned to this agent
        self.credits = 100    # Starting credits (will be updated based on containers)
        self.obtained_codes = {}  # Map of container type to code
        self.specializations = specializations or []  # List of container types this agent specializes in
        self.specialization_discount = specialization_discount  # Discount rate
    
    async def initialize(self):
        """
        Initialize the agent by:
        - describing the tasks it faces
        - giving specializations if it has any
        - asking it to choose a name
        - registering in the database
        """
        # Prepare the initialization prompt
        init_prompt = INTRODUCTORY_PROMPT
#        if "{additional_info}" in SYSTEM_PROMPT and self.specializations:
#            specialization_info = await self.get_specialization_info()
#            init_prompt = SYSTEM_PROMPT.format(additional_info=specialization_info)
#        elif "{additional_info}" in SYSTEM_PROMPT:
            # Remove the placeholder if no specializations
#            init_prompt = SYSTEM_PROMPT.replace("{additional_info}", "")
        
        # Log that we're sending the system prompt to this agent
        await self.db_manager.log_message(
            conversation_id=None,
            sender_id=0,  # System agent ID
            receiver_id=self.agent_id,
            conversation_round=0,
            message=init_prompt,
            message_type="initialization",
            message_category="system_instruction"
        )
        
        # Get the name from the LLM
        response = await self.send_message(init_prompt)
        
        # Validate response
        if not response or not response.strip():
            error_msg = f"Agent {self.agent_id} failed to provide a name. Empty response received."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)


        # Extract the first non-empty token from the response
        first_line = response.strip().splitlines()[0]
        first_token = first_line.split()[0]
        
        # Clean the name and set it
        self.name = clean_name(first_token)

        if not self.name:
            error_msg = f"Agent {self.agent_id} provided a name with no valid letters: {first_token}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
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
                parameters={"temperature": temperature},  # Use the global temperature
                specializations=self.specializations,
                specialization_discount=self.specialization_discount
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

        if not raw_name or not raw_name.strip():
            error_msg = f"Agent {self.agent_id} failed to provide a new name. Empty response received."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        first_line = raw_name.strip().splitlines()[0]
        first_token = first_line.split()[0]
        self.name = clean_name(first_token)
        
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
    
    async def get_specialization_info(self):
        """Get a formatted string describing the agent's specializations"""
        if not self.specializations:
            return ""
        
        discount_percent = int(self.specialization_discount * 100)
        remaining_percent = 100 - discount_percent
        
        return (f"You are specialized in: {', '.join(self.specializations)}. "
                f"You pay only {remaining_percent}% of the normal price "
                f"to open containers of these types.")
    
    def check_specialization(self, container_color):
        """Check if the agent has a specialization for a specific container color"""
        return container_color in self.specializations
    
    @classmethod
    async def create(cls, db_manager, specializations=None, specialization_discount=0.5):
        """
        Factory method for asynchronous agent creation with database integration.
        
        :param db_manager: DatabaseManager instance for logging
        :param specializations: List of specializations for this agent
        :param specialization_discount: Discount rate for specialized containers
        :return: Initialized Agent instance
        """
        agent = cls(db_manager, specializations, specialization_discount)
        return await agent.initialize()
    
    def assign_containers(self, containers, initial_credits):
        """
        Assign containers to this agent and set initial credits.
        
        :param containers: List of Container instances
        :param initial_credits: Starting credit amount
        """
        self.containers = containers
        self.credits = initial_credits
        self.obtained_codes = {}  # Reset obtained codes


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

    async def run_conversation_round(self):
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
        # Create a new conversation round in the database
        round_id = await self.db_manager.create_conversation(
            round_number=self.round_number,
            task_type="agent_matchmaking",
            task_parameters={"conversation_limit": self.conversation_limit}
        )

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

            # For the very first choice in an economic round, include the introduction
            if hasattr(agent, 'round_introduction'):
                prompt = (
                    f"{agent.round_introduction}\n\n"
                    f"Available agents to connect with: {', '.join(available_names)}. "
                    "Choose one or more agents to call (comma-separated) or type 'skip' to refuse."
                )
                # Clear the introduction so it's not repeated
                delattr(agent, 'round_introduction')
            else:
                # Standard prompt for subsequent connections
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
            f"Remember, to end the conversation at any time, type '{TERMINATION_SIGNAL}'."
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
    
    async def run_economic_round(self, container_manager, container_config):
        """
        Runs a complete economic round with container distribution and negotiations.
        
        :param container_manager: ContainerManager instance
        :param container_config: Configuration for container distribution
        """
        self.round_number += 1
        
        # Create a new round in the database
        round_id = await self.db_manager.create_conversation(
            round_number=self.round_number,
            task_type="economic_round",
            task_parameters={
                "conversation_limit": self.conversation_limit,
                "container_config": container_config
            }
        )
        
        print(f"\n=== ROUND {self.round_number} STARTING ===")
        
        # Phase 1: Container Distribution
        print("\nPhase 1: Distributing containers...")
        container_manager.generate_all_containers()
        await container_manager.distribute_containers(self.agents, self.db_manager, self.round_number)
        
        # Phase 2: Introduction Message
        print("\nPhase 2: Preparing introduction messages...")
        await self._prepare_introduction_messages(round_id)
        
        # Phase 3: Negotiation Phase
        print("\nPhase 3: Starting negotiations...")
        await self._run_negotiation_phase(round_id)
        
        # Phase 4: Action Phase
        # Phase 4: Action Phase - Opening
        print("\nPhase 4a: Collecting opening actions...")
        opening_actions = await self._collect_opening_actions(round_id)
        codes_obtained = await self._process_openings(round_id, opening_actions, container_manager)

        # Phase 4b: Action Phase - Sharing  
        print("\nPhase 4b: Collecting sharing actions...")
        sharing_actions = await self._collect_sharing_actions(round_id, codes_obtained)
        
        # Phase 5: Verification Phase
        print("\nPhase 5: Verifying shared codes...")
        await self._verify_actions(round_id, sharing_actions, container_manager)
        
        # Phase 6: Auto-Resolution
        print("\nPhase 6: Auto-opening remaining containers...")
        await self._auto_resolve_containers(round_id)
        
        # Phase 7: Feedback & Memory
        print("\nPhase 7: Collecting feedback and memories...")
        await self._collect_feedback_and_memory(round_id)
        
        # End the round
        await self.db_manager.end_conversation(
            conversation_id=round_id,
            status="completed",
            notes=f"Economic round {self.round_number} completed"
        )
        
        print(f"\n=== ROUND {self.round_number} COMPLETED ===")
    
    async def _prepare_introduction_messages(self, round_id):
        """Prepare introduction messages to each agent about their containers and rules."""
        for agent in self.agents:
            # Get agent's memory from previous rounds
            memory = await self.db_manager.get_agent_memory(agent.agent_id)
            memory_section = ""
            
            if memory and memory.get("experiences"):
                # Get the most recent learning
                recent_experience = memory["experiences"][-1]  # Last memory note

                # Strict check - learnings should exist after first round
                if not recent_experience.get('learnings'):
                    raise ValueError(f"Agent {agent.get_name()} (ID: {agent.agent_id}) has no learnings from previous round! "
                                    f"Experience data: {recent_experience}")

                memory_section = ROUND_START_PROMPT.format(name=agent.get_name())
                memory_section += f"{recent_experience['learnings']}\n"
                memory_section += "\n"
            else:
                # First time for this agent
                memory_section = f"Great! Your name is {agent.get_name()}.\n\n"
            
            # Build container list for this agent
            container_list = []
            total_base_cost = 0
            total_actual_cost = 0
            
            for container in agent.containers:
                _, cost = container.get_cost(agent)
                total_base_cost += container.base_credit_cost
                total_actual_cost += cost
                
                cost_info = f"{cost} credits"
                if cost < container.base_credit_cost:
                    cost_info += f" (discounted from {container.base_credit_cost} due to your specialization)"
                
                container_list.append(
                    f"- {container.color.capitalize()} container #{container.number}: {cost_info}"
                )
            
            # Add specialization reminder if applicable
            specialization_info = ""
            if agent.specializations:
                specialization_info = await agent.get_specialization_info() + "\n\n"
            
            introduction = f"""{memory_section}{specialization_info}You have been assigned the following containers:
                {chr(10).join(container_list)}

                Your starting credits: {agent.credits}
                Total cost to open all your containers: {total_actual_cost} credits""" + TASK_INTRODUCTION_PROMPT

            agent.round_introduction = introduction  # Store it on the agent object
            
    
    async def _run_negotiation_phase(self, round_id):
        """Run the negotiation phase using the existing conversation mechanism."""
        # Reset conversation counts for this negotiation phase
        self.conversation_counts = {agent: 0 for agent in self.agents}
        self.available_agents = set(self.agents)
                
        # Run the conversation matching process
        await self.run_conversation_round()
    
    async def _collect_opening_actions(self, round_id, container_manager):
        """Collect opening actions from all agents."""
        opening_actions = {} # agent_id -> list of container identifiers to open
        
        action_prompt = COLLECT_OPENING_ACTIONS_PROMPT
        
        for agent in self.agents:
            # Send action prompt
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=action_prompt,
                message_type="opening_action_request",
                message_category="system_instruction"
            )
            
            response = await agent.send_message(action_prompt)
            
            # Log response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,
                conversation_round=self.round_number,
                message=response,
                message_type="opening_action_response",
                message_category="agent_conversation"
            )
            
            # Parse actions (simplified - you may want more robust parsing)
            opening_actions[agent.agent_id] = []
            
            lines = response.strip().split('\n')
            found_open_section = False
            
            for line in lines:
                line = line.strip()
                if line == "OPEN:":
                    found_open_section = True
                    continue
                elif found_open_section and line:
                    # Check if it's a valid container identifier
                    if line in container_manager.container_registry:
                        opening_actions[agent.agent_id].append(line)
                    else:
                        # If it's not a valid container, log a warning but continue
                        print(f"Warning: Agent {agent.get_name()} tried to open invalid container: {line}")

            print(f"Agent {agent.get_name()} will open: {opening_actions[agent.agent_id]}")
        
        return opening_actions
    
    async def _process_openings(self, round_id, opening_actions, container_manager):
        """Process all container openings and return the codes obtained."""
        codes_obtained = {}  # agent_id -> dict of container_id -> code
        
        for agent in self.agents:
            codes_obtained[agent.agent_id] = {}
            agent_opens = opening_actions.get(agent.agent_id, [])
            opening_msg = "No"

            if agent_opens:
                opening_msg = f"Opening {len(agent_opens)} containers:\n"
                
                for container_id in agent_opens:
                    if container_id in container_manager.container_registry:
                        container = container_manager.container_registry[container_id]
                        message, code = container.open(agent)
                        codes_obtained[agent.agent_id][container_id] = code
                        opening_msg += f"- {container_id}: {message}\n"
                        
                        # Log individual opening
                        await self.db_manager.log_message(
                            conversation_id=round_id,
                            sender_id=agent.agent_id,
                            receiver_id=0,
                            conversation_round=self.round_number,
                            message=f"Opened container {container_id}: {message}",
                            message_type="container_opened",
                            message_category="agent_conversation"
                        )
                    else:
                        opening_msg += f"- {container_id}: ERROR - Container not found\n"
                
                # Send summary to agent
                await self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=0,
                    receiver_id=agent.agent_id,
                    conversation_round=self.round_number,
                    message=opening_msg,
                    message_type="opening_results",
                    message_category="system_instruction"
                )
                
            agent.containers_opening_message = opening_msg
        
        return codes_obtained
    
    async def _collect_sharing_actions(self, round_id, codes_obtained, container_manager):
        """Collect code sharing actions after containers have been opened."""
        sharing_actions = {}  # agent_id -> list of (recipient_name, container_id, code)
        
        for agent in self.agents:
            # Build list of codes this agent knows
            share_prompt = ""

            if agent.containers_opening_message == "No":
                share_prompt = "You didn't open any containers"
            else:
                share_prompt = agent.containers_opening_message

            share_prompt += SHARING_ACTIONS_PROMPT
                
            # Log the sharing request
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=share_prompt,
                message_type="sharing_request",
                message_category="system_instruction"
            )
            
            response = await agent.send_message(share_prompt)
            
            # Log the response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,
                conversation_round=self.round_number,
                message=response,
                message_type="sharing_response",
                message_category="agent_conversation"
            )
            
            # Parse sharing actions
            lines = response.strip().split('\n')
            in_share_section = False
            
            for line in lines:
                line = line.strip()
                if line == "SHARE:":
                    in_share_section = True
                    continue
                
                if in_share_section and line and '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        recipient_name, container_id, provided_code = parts
                        
                        # Check the nature of the share
                        share_type = "unknown"
                        warning_msg = None
                        
                        # Check if container exists
                        if container_id in container_manager.container_registry:
                            actual_code = container_manager.container_registry[container_id].code
                            
                            if provided_code == actual_code:
                                # Correct code
                                if container_id in codes_obtained.get(agent.agent_id, {}):
                                    share_type = "correct_just_opened"
                                else:
                                    share_type = "correct_from_memory"
                                    warning_msg = f"WARNING: Agent {agent.get_name()} shared correct code for {container_id} from memory (not just opened)"
                            else:
                                # Wrong code - deception
                                share_type = "deception"
                                warning_msg = f"WARNING: Agent {agent.get_name()} shared WRONG code for {container_id}: {provided_code} (actual: {actual_code})"
                        else:
                            share_type = "invalid_container"
                            warning_msg = f"WARNING: Agent {agent.get_name()} tried to share code for non-existent container {container_id}"
                        
                        # Log the share attempt with its type
                        await self.db_manager.log_message(
                            conversation_id=round_id,
                            sender_id=agent.agent_id,
                            receiver_id=0,
                            conversation_round=self.round_number,
                            message=f"Share attempt - Type: {share_type}, To: {recipient_name}, Container: {container_id}, Code: {provided_code}",
                            message_type="share_analysis",
                            message_category="system_feedback"
                        )
                        
                        # Print warning if needed
                        if warning_msg:
                            print(warning_msg)
                        
                        # Add to sharing actions (allow all shares, even deceptive ones)
                        sharing_actions[agent.agent_id].append((recipient_name, container_id, provided_code))
        
        return sharing_actions
        
    
    async def _verify_actions(self, round_id, sharing_actions, container_manager):
        """Process code shares and send verification results to all agents."""
        # Process all shares and build verification results
        all_shares = []
        
        for sender_agent in self.agents:

            sender_agent.end_round_msg = ""

            sender_shares = sharing_actions.get(sender_agent.agent_id, [])

            for recipient_name, container_id, provided_code in sender_shares:
                # Find recipient agent
                recipient = None
                for a in self.agents:
                    if a.get_name().lower() == recipient_name.lower():
                        recipient = a
                        break
                
                if not recipient:
                    sender_agent.sharing_warnings+=f"WARNING: You tried to share with non-existent agent {recipient_name}\n"
                    print(f"WARNING: Agent {sender_agent.get_name()} tried to share with non-existent agent {recipient_name}")
                    continue
                
                # Verify the code
                is_valid = False
                if container_id in container_manager.container_registry:
                    container = container_manager.container_registry[container_id]
                    actual_code = container.code
                    is_valid = (provided_code == actual_code)
                    
                    if is_valid:
                        # Add valid code to recipient's obtained codes
                        recipient.obtained_codes[container_id] = provided_code
                
                # Record the share
                all_shares.append({
                    'sender': sender_agent.get_name(),
                    'recipient': recipient.get_name(),
                    'recipient_agent': recipient,
                    'container': container_id,
                    'code': provided_code,
                    'valid': is_valid
                })
        
        # Send verification results to each agent
        for agent in self.agents:
            # Find all codes shared with this agent
            received_shares = [share for share in all_shares 
                            if share['recipient'].lower() == agent.get_name().lower()]
            
            if received_shares:
                # Build the verification message
                verification_lines = ["You have received the following codes:\n"]
                
                for share in received_shares:
                    if share['valid']:
                        status = "valid"
                    else:
                        status = "WARNING: invalid code"
                    
                    line = f"{share['sender']} | {share['container']} | {share['code']} | {status}"
                    verification_lines.append(line)
                
                verification_message = "\n".join(verification_lines)
            else:
                verification_message = "You have not received any codes from other agents."
            
            # Log the verification message
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=verification_message,
                message_type="verification_results",
                message_category="system_instruction"
            )
            
            # Pospone sending to agent
            agent.end_round_msg += verification_message +"\n"
        
        # Return summary for logging
        return {
            'total_shares': len(all_shares),
            'valid_shares': sum(1 for share in all_shares if share['valid']),
            'invalid_shares': sum(1 for share in all_shares if not share['valid']),
            'shares_by_agent': {
                agent.get_name(): len([s for s in all_shares if s['sender'] == agent.get_name()])
                for agent in self.agents
            }
        }
    async def _auto_resolve_containers(self, round_id):
        """Automatically open any remaining containers."""
        for agent in self.agents:
            containers_to_open = []
            
            for container in agent.containers:
                container_id = container.get_identifier()
                _, cost = container.get_cost(agent)
                if container_id not in agent.obtained_codes:
                    containers_to_open.append(container)
            
            if containers_to_open:
                auto_open_msg = f"Auto-opening {len(containers_to_open)} remaining containers:\n"
                total_cost_to_open = 0
                
                for container in containers_to_open:
                    # Get cost before opening
                    _, cost = container.get_cost(agent)
                    # Now open (which deducts the cost)
                    _, code = container.open(agent)
                    total_cost_to_open += cost
                    auto_open_msg += f"- {container.get_identifier()}: {code} (cost: {cost} credits)\n"
                
                auto_open_msg += f"\nTotal auto-open cost: {total_cost_to_open} credits"
                auto_open_msg += f"\nFinal credit balance: {agent.credits} credits\n"
                
                await self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=0,
                    receiver_id=agent.agent_id,
                    conversation_round=self.round_number,
                    message=auto_open_msg,
                    message_type="auto_resolution",
                    message_category="system_instruction"
                )
                
                agent.end_round_msg.join(auto_open_msg)

            else:
                # Agent successfully obtained all codes without auto-opening
                success_msg = (f"Auto-opening ... you already have all the codes! Nothing to open\n"
                            f"Final credit balance: {agent.credits} credits\n")
                
                await self.db_manager.log_message(
                    conversation_id=round_id,
                    sender_id=0,
                    receiver_id=agent.agent_id,
                    conversation_round=self.round_number,
                    message=success_msg,
                    message_type="auto_resolution",
                    message_category="system_instruction"
                )
                
                agent.end_round_msg.join(auto_open_msg)
    
    async def _collect_feedback_and_memory(self, round_id):

        """Collect feedback and memories from agents."""
        
        for agent in self.agents:

            ### Analysis of results

            # how much would be saved if all containters are opened my itself
            # minus 
            # how much was saved
            net_gain = agent.total_cost * (agents_credits_multiplier - 1) - agent.credits

            msg = (f"\nAt the start you had {agent.total_cost * agents_credits_multiplier} credits\n"
                   f"If you open all the containers yourself you would spend {agent.total_cost} credits\n"
                   f"Therefore our net gain is {net_gain} credits\n\n")
            
            if net_gain > 0: 
                msg+=("Congratulations! It is positive, during negotiations you manage to save credits!"
                    "Please, tell me about your strategy")
            elif net_gain == 0: 
                msg+=("Actualy, you didn't manage to save any credits, why do you think it has happend?"
                    "How you can improve in the future?")
            else:
                msg+=("You spent more than it would be needed to open everything yourself."
                      "Why has it happend? How would you improve in the future?")
                
            # Send resulting message and feedback request
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=msg,
                message_type="strategy_feedback_request",
                message_category="system_instruction"
            )         
     
            strategy_feedback = await agent.send_message(agent.end_message + msg)
            
            # Log response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,
                conversation_round=self.round_number,
                message=strategy_feedback,
                message_type="strategy_feedback_response",
                message_category="agent_conversation"
            )

            ### Memory request 

            memory = await self.db_manager.get_agent_memory(agent.agent_id)
            recent_experience = ""
            if memory and memory.get("experiences"):
                recent_experience = "\nHere is a reminder how your previous memory note looks like. Please update it. \n \n" + memory["experiences"][-1]


            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=MEMORY_REQUEST_PROMPT + recent_experience,
                message_type="memory_request",
                message_category="system_instruction"
            )

            new_experience = await agent.send_message(MEMORY_REQUEST_PROMPT + recent_experience)

            # Log response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,
                conversation_round=self.round_number,
                message=new_experience,
                message_type="memory_response",
                message_category="agent_conversation"
            )


            ### Questions

            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=self.round_number,
                message=GENERAL_FEEDBACK_REQUEST_PROMPT,
                message_type="general_feedback_request",
                message_category="system_instruction"
            )

            general_feedback = await agent.send_message(GENERAL_FEEDBACK_REQUEST_PROMPT)

            # Log response
            await self.db_manager.log_message(
                conversation_id=round_id,
                sender_id=agent.agent_id,
                receiver_id=0,
                conversation_round=self.round_number,
                message=general_feedback,
                message_type="general_feedback_responce",
                message_category="agent_conversation"
            )
            
            await self.db_manager.update_agent_memory(agent.agent_id, strategy_feedback, new_experience)

# =============================================================================
# Conteiners: economical system 
# =============================================================================

class Container:
    # Shared code generator instance for all containers
    _code_generator = None
    
    @classmethod
    def initialize_code_generator(cls, seed=None):
        """Initialize the shared code generator"""
        cls._code_generator = ContainerCodeGenerator(seed)
    
    def __init__(self, color, number, base_credit_cost):
        """Initialize a container with specified attributes"""
        self.color = color
        self.number = number
        self.base_credit_cost = base_credit_cost
        self.is_opened = False
        
        # Ensure code generator exists
        if not Container._code_generator:
            Container.initialize_code_generator()
        
        # Generate code deterministically
        self._code = None  # Will be generated on demand
        
    @property
    def code(self):
        """Get the container's code, generating it if needed"""
        if self._code is None:
            self._code = Container._code_generator.generate_code(self.color, self.number)
        return self._code
        
    def get_identifier(self):
        """Returns a unique identifier for this container"""
        return f"{self.color}-{self.number}"
    
    def get_cost(self, agent):
        """Calculate the cost for a specific agent to open this container"""
            
        has_specialization = agent.check_specialization(self.color)
        
        if has_specialization:
            discount = agent.specialization_discount
            discounted_cost = self.base_credit_cost * (1 - discount)
            return (f"The price to open this {self.color} container #{self.number} is {discounted_cost} credits "
                    f"for you because you have a specialization in {self.color} containers, "
                    f"which reduces the price by {int(discount * 100)}%.", discounted_cost)
        else:
            return (f"The price to open this {self.color} container #{self.number} is {self.base_credit_cost} credits "
                   f"for you as you have no specialization for {self.color} containers.", self.base_credit_cost)

    def open(self, agent):
        """Attempt to open the container using the agent's credits"""
        _, cost = self.get_cost(agent)
        
        # Deduct credits (allow negative balance)
        agent.credits -= cost
        self.is_opened = True
        
        # Record the code in agent's obtained codes
        agent.obtained_codes[self.get_identifier()] = self.code
        
        return f"Container opened successfully! Cost: {cost} credits. Remaining credits: {agent.credits}. Code revealed: {self.code}", self.code

    def verify_code(self, provided_code):
        """Verify if a provided code matches this container's code"""
        if provided_code == self.code:
            return True, "The code is correct for this container."
        else:
            return False, "The code does not match this container."    
    
    def __str__(self):
        """String representation of the container"""
        return f"{self.color.capitalize()} Container #{self.number} - Base cost: {self.base_credit_cost} credits"


class ContainerCodeGenerator:
    def __init__(self, seed=None):
        """Initialize with an optional seed for reproducibility"""
        self.seed = seed if seed is not None else random.randint(10000, 99999)
        
    def generate_code(self, color, number):
        """Generate a deterministic code based on color and number"""
        # Create a base string that combines all inputs
        base_string = f"{color}#{number}#{self.seed}"
        
        # Use a cryptographic hash function (SHA-256)
        import hashlib
        hash_obj = hashlib.sha256(base_string.encode())
        digest = hash_obj.hexdigest()
        
        # Extract a portion and convert to an integer
        hex_segment = digest[:6]  # First 6 hex chars (24 bits)
        value = int(hex_segment, 16)
        
        # Limit to a 5-digit number
        code_number = value % 90000 + 10000  # Range: 10000-99999
        
        # Format the final code
        return str(code_number)


# =============================================================================
# Container Manager: handles distribution of containers to agents
# =============================================================================

class ContainerManager:
    def __init__(self, config=None):
        """
        Initialize the container manager with configuration.
        
        :param config: Dictionary with container distribution settings
        Example config:
        {
            'colors': ['red', 'blue', 'green'],
            'numbers': [1, 2, 3],
            'base_costs': {'red': 10, 'blue': 15, 'green': 20},  # per color
            'distribution_mode': 'fixed',  # 'fixed' or 'random'
            'containers_per_agent': 2,  # for fixed mode
            'containers_per_agent_range': (1, 4),  # for random mode
            'overlap_mode': 'fixed',  # 'fixed', 'random', or 'controlled'
            'copies_per_container': 2,  # for fixed overlap
            'copies_range': (1, 3),  # for random overlap
            'controlled_distribution': {  # for controlled mode
                'red-1': ['agent1', 'agent2'],
                'blue-2': ['agent1'],
                # etc.
            }
        }
        """
        self.config = config or self._get_default_config()
        self.container_registry = {}  # Maps container_id to Container instance
        self.agent_containers = {}  # Maps agent_id to list of container_ids
        self.container_assignments = {}  # Maps container_id to list of agent_ids
        
    def _get_default_config(self):
        """Returns default configuration for container distribution"""
        return {
            'colors': ['red', 'blue', 'green'],
            'numbers': [1, 2, 3],
            'base_costs': {'red': 10, 'blue': 10, 'green': 10},
            'distribution_mode': 'fixed',
            'containers_per_agent': 2,
            'containers_per_agent_range': (1, 4),
            'overlap_mode': 'fixed',
            'copies_per_container': 2,
            'copies_range': (1, 3)
        }
    
    def generate_all_containers(self):
        """Generate all possible containers based on colors and numbers"""
        containers = []
        for color in self.config['colors']:
            base_cost = self.config['base_costs'].get(color, 10)
            for number in self.config['numbers']:
                container = Container(color, number, base_cost)
                container_id = container.get_identifier()
                self.container_registry[container_id] = container
                self.container_assignments[container_id] = []
                containers.append(container)
        return containers
    
    async def distribute_containers(self, agents, db_manager, round_number):
        """
        Distribute containers to agents based on configuration.
        
        :param agents: List of Agent instances
        :param db_manager: DatabaseManager instance for logging (optional)
        :param round_number: Current round number for logging
        :return: Dictionary mapping agent_id to list of assigned containers
        """
        # Initialize agent container lists
        for agent in agents:
            self.agent_containers[agent.agent_id] = []
        
        if self.config.get('controlled_distribution'):
            # Use predefined distribution
            self._controlled_distribution(agents)
        else:
            # Use automatic distribution
            if self.config['overlap_mode'] == 'fixed':
                self._fixed_overlap_distribution(agents)
            elif self.config['overlap_mode'] == 'random':
                self._random_overlap_distribution(agents)
            else:
                raise ValueError(f"Unknown overlap_mode: {self.config['overlap_mode']}")
        
        # Assign containers to agents and calculate initial credits
        for agent in agents:
            agent.containers = []
            total_cost = 0
            
            for container_id in self.agent_containers[agent.agent_id]:
                container = self.container_registry[container_id]
                agent.containers.append(container)
                # Calculate cost considering specialization
                _, cost = container.get_cost(agent)
                total_cost += cost
                
                # Log the container assignment if db_manager provided

                await db_manager.log_container_assignment(
                    agent_id=agent.agent_id,
                    container_id=container_id,
                    container_color=container.color,
                    container_number=container.number,
                    base_cost=container.base_credit_cost,
                    actual_cost=cost,
                    round_number=round_number)

            
            # Set credits to total cost * agents_credits_multiplier
            agent.credits = int(total_cost * agents_credits_multiplier)
            agent.total_cost = total_cost
            
            # Log the initial credit assignment

            await db_manager.log_message(
                conversation_id=None,
                sender_id=0,
                receiver_id=agent.agent_id,
                conversation_round=round_number,
                message=f"Initial credits assigned: {agent.credits} (based on container costs + 20%)",
                message_type="credit_assignment",
                message_category="system_instruction"
            )
            
        return self.agent_containers
    
    def _controlled_distribution(self, agents):
        """Distribute containers based on explicit configuration"""
        # Create agent name to agent mapping
        agent_map = {agent.get_name(): agent for agent in agents}
        
        for container_id, agent_names in self.config['controlled_distribution'].items():
            if container_id not in self.container_registry:
                print(f"Warning: Container {container_id} not in registry, skipping")
                continue
                
            for agent_name in agent_names:
                if agent_name in agent_map:
                    agent = agent_map[agent_name]
                    self.agent_containers[agent.agent_id].append(container_id)
                    self.container_assignments[container_id].append(agent.agent_id)
                else:
                    print(f"Warning: Agent {agent_name} not found, skipping")
    
    def _fixed_overlap_distribution(self, agents):
        """Each container appears in exactly 'copies_per_container' agents"""
        all_containers = list(self.container_registry.keys())
        copies_per_container = self.config['copies_per_container']
        
        # Determine containers per agent
        if self.config['distribution_mode'] == 'fixed':
            containers_per_agent = self.config['containers_per_agent']
        else:
            # Random mode: each agent gets random number of containers
            containers_per_agent = None
        
        # Create assignment slots
        assignment_slots = []
        for container_id in all_containers:
            assignment_slots.extend([container_id] * copies_per_container)
        
        # Shuffle for randomness
        random.shuffle(assignment_slots)
        
        # Distribute to agents
        agent_list = list(agents)
        agent_index = 0
        
        for container_id in assignment_slots:
            # Find an agent that doesn't have this container yet
            attempts = 0
            while attempts < len(agent_list):
                agent = agent_list[agent_index % len(agent_list)]
                
                # Check if agent needs more containers
                if self.config['distribution_mode'] == 'fixed':
                    can_add = len(self.agent_containers[agent.agent_id]) < containers_per_agent
                else:
                    min_cont, max_cont = self.config['containers_per_agent_range']
                    can_add = len(self.agent_containers[agent.agent_id]) < max_cont
                
                # Check if agent doesn't have this container
                if can_add and container_id not in self.agent_containers[agent.agent_id]:
                    self.agent_containers[agent.agent_id].append(container_id)
                    self.container_assignments[container_id].append(agent.agent_id)
                    break
                
                agent_index += 1
                attempts += 1
            
            if attempts == len(agent_list):
                print(f"Warning: Could not assign container {container_id} to any agent")
    
    def _random_overlap_distribution(self, agents):
        """Each container appears in random number of agents"""
        all_containers = list(self.container_registry.keys())
        min_copies, max_copies = self.config['copies_range']
        
        # Assign each container to random number of agents
        for container_id in all_containers:
            num_copies = random.randint(min_copies, max_copies)
            
            # Randomly select agents for this container
            available_agents = list(agents)
            random.shuffle(available_agents)
            
            assigned = 0
            for agent in available_agents:
                if assigned >= num_copies:
                    break
                
                # Check if agent can receive more containers
                if self.config['distribution_mode'] == 'fixed':
                    max_containers = self.config['containers_per_agent']
                else:
                    _, max_containers = self.config['containers_per_agent_range']
                
                if len(self.agent_containers[agent.agent_id]) < max_containers:
                    self.agent_containers[agent.agent_id].append(container_id)
                    self.container_assignments[container_id].append(agent.agent_id)
                    assigned += 1
        
        # Ensure minimum containers per agent in random mode
        if self.config['distribution_mode'] == 'random':
            min_containers, _ = self.config['containers_per_agent_range']
            for agent in agents:
                while len(self.agent_containers[agent.agent_id]) < min_containers:
                    # Find a container to add
                    available = [c for c in all_containers 
                               if c not in self.agent_containers[agent.agent_id]]
                    if available:
                        container_id = random.choice(available)
                        self.agent_containers[agent.agent_id].append(container_id)
                        self.container_assignments[container_id].append(agent.agent_id)
                    else:
                        break
    
    def get_container_distribution_summary(self):
        """Get a summary of the container distribution"""
        summary = {
            'total_containers': len(self.container_registry),
            'agent_container_counts': {},
            'container_copy_counts': {},
            'overlap_matrix': {}
        }
        
        # Count containers per agent
        for agent_id, containers in self.agent_containers.items():
            summary['agent_container_counts'][agent_id] = len(containers)
        
        # Count copies per container
        for container_id, agents in self.container_assignments.items():
            summary['container_copy_counts'][container_id] = len(agents)
        
        # Calculate overlap between agents
        agent_ids = list(self.agent_containers.keys())
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i+1:]:
                containers1 = set(self.agent_containers[agent1])
                containers2 = set(self.agent_containers[agent2])
                overlap = len(containers1.intersection(containers2))
                summary['overlap_matrix'][f"{agent1}-{agent2}"] = overlap
        
        return summary

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
    
    print("\nAll conversations completed. Logs have been stored in 'conversation_logs.db'.")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
