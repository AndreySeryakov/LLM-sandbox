import sqlite3
import json
import os
import argparse
from tabulate import tabulate
from datetime import datetime


class DatabaseExplorer:
    def __init__(self, db_path):
        """Initialize the DatabaseExplorer with path to the database file."""
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Connect to the database
        self.conn = sqlite3.connect(db_path)
        # Enable row factory to get dictionaries
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        print(f"Connected to database: {db_path}")

    def __del__(self):
        """Close connection when object is deleted."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def list_agents(self):
        """List all agents in the database."""
        self.cursor.execute("""
        SELECT * FROM agents ORDER BY agent_id
        """)
        
        agents = [dict(row) for row in self.cursor.fetchall()]
        
        if not agents:
            print("No agents found in the database.")
            return
        
        # Extract headers and data for tabulate
        headers = ["ID", "Name", "Model", "Created", "Type"]
        data = []
        
        for agent in agents:
            # Parse parameters if available
            params = None
            if agent['parameters']:
                try:
                    params = json.loads(agent['parameters'])
                except:
                    params = str(agent['parameters'])
            
            created = agent['creation_timestamp']
            if created:
                # Try to format the timestamp if it's an ISO format
                try:
                    dt = datetime.fromisoformat(created)
                    created = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            data.append([
                agent['agent_id'], 
                agent['name'], 
                agent['model_name'], 
                created,
                agent['agent_type']
            ])
        
        print("\n=== AGENTS ===")
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
        return agents

    def list_conversations(self):
        """List all conversations in the database."""
        self.cursor.execute("""
        SELECT * FROM conversations ORDER BY conversation_id
        """)
        
        conversations = [dict(row) for row in self.cursor.fetchall()]
        
        if not conversations:
            print("No conversations found in the database.")
            return
        
        # Extract headers and data for tabulate
        headers = ["ID", "Round", "Start Time", "End Time", "Status", "Type", "Turns"]
        data = []
        
        for conv in conversations:
            # Format timestamps
            start_time = conv['start_timestamp']
            if start_time:
                try:
                    dt = datetime.fromisoformat(start_time)
                    start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            end_time = conv['end_timestamp']
            if end_time:
                try:
                    dt = datetime.fromisoformat(end_time)
                    end_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            # Parse parameters if available
            params = None
            if conv['task_parameters']:
                try:
                    params = json.loads(conv['task_parameters'])
                except:
                    params = str(conv['task_parameters'])
            
            data.append([
                conv['conversation_id'],
                conv['round_number'],
                start_time,
                end_time,
                conv['status'],
                conv['task_type'],
                conv['turn_count']
            ])
        
        print("\n=== CONVERSATIONS ===")
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
        return conversations

    def view_conversation(self, conversation_id):
        """View details of a specific conversation including all messages."""
        # Get conversation details
        self.cursor.execute("""
        SELECT * FROM conversations WHERE conversation_id = ?
        """, (conversation_id,))
        
        conversation = self.cursor.fetchone()
        if not conversation:
            print(f"Conversation with ID {conversation_id} not found.")
            return
        
        conversation = dict(conversation)
        
        # Format conversation details
        print("\n=== CONVERSATION DETAILS ===")
        print(f"ID: {conversation['conversation_id']}")
        print(f"Round: {conversation['round_number']}")
        print(f"Status: {conversation['status']}")
        print(f"Task Type: {conversation['task_type']}")
        print(f"Turn Count: {conversation['turn_count']}")
        
        if conversation['task_parameters']:
            try:
                params = json.loads(conversation['task_parameters'])
                print("\nTask Parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            except:
                print(f"Task Parameters: {conversation['task_parameters']}")
        
        if conversation['notes']:
            print(f"\nNotes: {conversation['notes']}")
        
        # Get messages for this conversation
        self.cursor.execute("""
        SELECT m.*, 
               a_sender.name as sender_name,
               a_receiver.name as receiver_name
        FROM messages m
        JOIN agents a_sender ON m.sender_id = a_sender.agent_id
        LEFT JOIN agents a_receiver ON m.receiver_id = a_receiver.agent_id
        WHERE m.conversation_id = ?
        ORDER BY m.message_sequence, m.timestamp
        """, (conversation_id,))
        
        messages = [dict(row) for row in self.cursor.fetchall()]
        
        if not messages:
            print("\nNo messages found for this conversation.")
            return
        
        print("\n=== MESSAGES ===")
        for message in messages:
            # Format timestamp
            timestamp = message['timestamp']
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            # Print message details
            print(f"\n[{timestamp}] ", end="")
            
            if message['message_sequence']:
                print(f"Seq {message['message_sequence']} | ", end="")
            
            print(f"{message['sender_name']} → ", end="")
            print(f"{message['receiver_name'] if message['receiver_name'] else 'All'}")
            
            print(f"Type: {message['message_type']} | Category: {message['message_category']}")
            print(f"Message: {message['message']}")
        
        # Get conversation summaries
        self.cursor.execute("""
        SELECT cs.*, a.name as agent_name
        FROM conversation_summaries cs
        JOIN agents a ON cs.agent_id = a.agent_id
        WHERE cs.conversation_id = ?
        """, (conversation_id,))
        
        summaries = [dict(row) for row in self.cursor.fetchall()]
        
        if summaries:
            print("\n=== CONVERSATION SUMMARIES ===")
            for summary in summaries:
                print(f"\nSummary by {summary['agent_name']}:")
                print(f"{summary['summary']}")
                
                if summary['learnings']:
                    print(f"\nLearnings:")
                    print(f"{summary['learnings']}")

    def list_messages_by_agent(self, agent_id):
        """List all messages sent by and received by a specific agent, in chronological order."""
        # First verify the agent exists
        self.cursor.execute("SELECT name FROM agents WHERE agent_id = ?", (agent_id,))
        agent = self.cursor.fetchone()
        
        if not agent:
            print(f"Agent with ID {agent_id} not found.")
            return
        
        agent_name = agent['name']
        
        # Get all messages sent by OR received by this agent
        self.cursor.execute("""
        SELECT m.*, 
               a_sender.name as sender_name,
               a_receiver.name as receiver_name,
               c.round_number,
               CASE
                  WHEN m.sender_id = ? THEN 'sent'
                  WHEN m.receiver_id = ? THEN 'received'
               END as direction
        FROM messages m
        JOIN agents a_sender ON m.sender_id = a_sender.agent_id
        LEFT JOIN agents a_receiver ON m.receiver_id = a_receiver.agent_id
        LEFT JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE m.sender_id = ? OR m.receiver_id = ?
        ORDER BY m.timestamp
        """, (agent_id, agent_id, agent_id, agent_id))
        
        messages = [dict(row) for row in self.cursor.fetchall()]
        
        if not messages:
            print(f"No messages found for agent {agent_name} (ID: {agent_id}).")
            return
        
        print(f"\n=== MESSAGES INVOLVING {agent_name.upper()} (ID: {agent_id}) ===")
        
        # Group messages by conversation for better context
        conversation_messages = {}
        for message in messages:
            conv_id = message['conversation_id'] or 'system'
            if conv_id not in conversation_messages:
                conversation_messages[conv_id] = []
            conversation_messages[conv_id].append(message)
        
        # Print messages by conversation
        for conv_id, msgs in conversation_messages.items():
            if conv_id == 'system':
                print(f"\n--- SYSTEM MESSAGES ---")
            else:
                print(f"\n--- CONVERSATION {conv_id} ---")
            
            for message in msgs:
                # Format timestamp
                timestamp = message['timestamp']
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                # Print message details with direction indicator
                direction = message['direction']
                direction_symbol = "→" if direction == 'sent' else "←"
                direction_label = "SENT" if direction == 'sent' else "RECEIVED"
                
                print(f"\n[{timestamp}] [{direction_label}] ", end="")
                
                if message['message_sequence']:
                    print(f"Seq {message['message_sequence']} | ", end="")
                
                if direction == 'sent':
                    other_party = message['receiver_name'] if message['receiver_name'] else 'All'
                    print(f"{agent_name} {direction_symbol} {other_party}")
                else:
                    print(f"{message['sender_name']} {direction_symbol} {agent_name}")
                
                print(f"Type: {message['message_type']} | Category: {message['message_category']}")
                print(f"Message: {message['message']}")


    def get_conversation_stats(self):
        """Get statistics about all conversations."""
        # Count conversations by status
        self.cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM conversations
        GROUP BY status
        """)
        
        status_counts = [dict(row) for row in self.cursor.fetchall()]
        
        # Count messages by category
        self.cursor.execute("""
        SELECT message_category, COUNT(*) as count
        FROM messages
        GROUP BY message_category
        """)
        
        category_counts = [dict(row) for row in self.cursor.fetchall()]
        
        # Count messages by type
        self.cursor.execute("""
        SELECT message_type, COUNT(*) as count
        FROM messages
        GROUP BY message_type
        """)
        
        type_counts = [dict(row) for row in self.cursor.fetchall()]
        
        # Average messages per conversation
        self.cursor.execute("""
        SELECT AVG(message_count) as avg_messages_per_conversation
        FROM (
            SELECT conversation_id, COUNT(*) as message_count
            FROM messages
            WHERE conversation_id IS NOT NULL
            GROUP BY conversation_id
        )
        """)
        
        avg_messages = self.cursor.fetchone()
        
        # Agent participation
        self.cursor.execute("""
        SELECT a.name, COUNT(DISTINCT m.conversation_id) as conversation_count
        FROM agents a
        JOIN messages m ON a.agent_id = m.sender_id
        WHERE m.conversation_id IS NOT NULL
        GROUP BY a.agent_id
        ORDER BY conversation_count DESC
        """)
        
        agent_participation = [dict(row) for row in self.cursor.fetchall()]
        
        # Print results
        print("\n=== DATABASE STATISTICS ===")
        
        print("\nConversation Status Counts:")
        if status_counts:
            for status in status_counts:
                print(f"  {status['status']}: {status['count']}")
        else:
            print("  No conversations found.")
        
        print("\nMessage Category Counts:")
        if category_counts:
            for category in category_counts:
                print(f"  {category['message_category']}: {category['count']}")
        else:
            print("  No messages found.")
        
        print("\nMessage Type Counts:")
        if type_counts:
            for type_count in type_counts:
                print(f"  {type_count['message_type']}: {type_count['count']}")
        else:
            print("  No messages found.")
        
        if avg_messages and avg_messages['avg_messages_per_conversation'] is not None:
            print(f"\nAverage Messages Per Conversation: {avg_messages['avg_messages_per_conversation']:.2f}")
        
        print("\nAgent Participation:")
        if agent_participation:
            for agent in agent_participation:
                print(f"  {agent['name']}: {agent['conversation_count']} conversations")
        else:
            print("  No agent participation data found.")

    def search_messages(self, search_term):
        """Search for messages containing a specific term."""
        self.cursor.execute("""
        SELECT m.*, 
               a_sender.name as sender_name,
               a_receiver.name as receiver_name,
               c.round_number
        FROM messages m
        JOIN agents a_sender ON m.sender_id = a_sender.agent_id
        LEFT JOIN agents a_receiver ON m.receiver_id = a_receiver.agent_id
        LEFT JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE m.message LIKE ?
        ORDER BY m.timestamp
        """, (f"%{search_term}%",))
        
        messages = [dict(row) for row in self.cursor.fetchall()]
        
        if not messages:
            print(f"No messages found containing '{search_term}'.")
            return
        
        print(f"\n=== MESSAGES CONTAINING '{search_term}' ===")
        for message in messages:
            # Format timestamp
            timestamp = message['timestamp']
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            # Print message details
            print(f"\n[{timestamp}] ", end="")
            
            if message['conversation_id']:
                print(f"Conv {message['conversation_id']} | ", end="")
            
            if message['round_number']:
                print(f"Round {message['round_number']} | ", end="")
            
            print(f"{message['sender_name']} → {message['receiver_name'] if message['receiver_name'] else 'All'}")
            
            print(f"Type: {message['message_type']} | Category: {message['message_category']}")
            print(f"Message: {message['message']}")


def main():
    parser = argparse.ArgumentParser(description='Explore and analyze a conversation database.')
    parser.add_argument('database', help='Path to the SQLite database file')
    parser.add_argument('--action', choices=['agents', 'conversations', 'conversation', 'agent_messages', 'stats', 'search'], 
                        default='agents', help='Action to perform (default: list agents)')
    parser.add_argument('--id', type=int, help='ID for conversation or agent (required for some actions)')
    parser.add_argument('--search', help='Search term for message search')
    
    args = parser.parse_args()
    
    try:
        explorer = DatabaseExplorer(args.database)
        
        if args.action == 'agents':
            explorer.list_agents()
        
        elif args.action == 'conversations':
            explorer.list_conversations()
        
        elif args.action == 'conversation':
            if args.id is None:
                print("Error: --id is required for viewing a specific conversation")
                return
            explorer.view_conversation(args.id)
        
        elif args.action == 'agent_messages':
            if args.id is None:
                print("Error: --id is required for viewing agent messages")
                return
            explorer.list_messages_by_agent(args.id)
        
        elif args.action == 'stats':
            explorer.get_conversation_stats()
        
        elif args.action == 'search':
            if args.search is None:
                print("Error: --search is required for searching messages")
                return
            explorer.search_messages(args.search)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()