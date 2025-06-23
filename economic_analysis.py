import sqlite3
import json
import argparse
import os
from contextlib import redirect_stdout
from datetime import datetime
from tabulate import tabulate
from collections import defaultdict
from colorama import init, Fore, Style, Back
init()


class EconomicAnalyzer:
    def __init__(self, db_path):
        """Initialize the analyzer with database path."""
        self.db_path = db_path
        print(f"Connecting to database: {db_path}")
        
        import os
        if not os.path.exists(db_path):
            print(f"WARNING: Database file does not exist at {db_path}")
        else:
            print(f"Database file size: {os.path.getsize(db_path)} bytes")
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Quick check of tables
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.cursor.fetchall()]
        print(f"Tables found: {', '.join(tables)}")


        # Color palette for agents (excluding red for target agent, blue for system)
        self.agent_colors = [
            Fore.GREEN,
            Fore.YELLOW,
            Fore.MAGENTA,
            Fore.CYAN,
            Fore.WHITE,
            Fore.LIGHTGREEN_EX,
            Fore.LIGHTYELLOW_EX,
            Fore.LIGHTMAGENTA_EX,
            Fore.LIGHTCYAN_EX,
            Fore.LIGHTBLACK_EX,  # Gray
        ]
        self.agent_color_map = {}
        self.next_color_index = 0
        
    def __del__(self):
        """Close connection when object is deleted."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def agent_participation_summary(self):
        """
        Function 1: Create a table showing each agent and all rounds they participated in.
        Format: Agent ID | Agent Name | Rounds Participated
        """
        print("\n=== AGENT PARTICIPATION SUMMARY ===\n")
        
        # Get all agents from agents_library
        self.cursor.execute("""
        SELECT agent_name, rounds_participated 
        FROM agents_library 
        ORDER BY agent_name
        """)
        
        # Group by agent name to handle multiple library entries
        agent_rounds = defaultdict(set)
        
        for row in self.cursor.fetchall():
            agent_name = row['agent_name']
            rounds_json = row['rounds_participated']
            
            if rounds_json:
                rounds = json.loads(rounds_json)
                for round_id in rounds:
                    agent_rounds[agent_name].add(round_id)
        
        # Also get agents from current agents table that might not be in library yet
        self.cursor.execute("""
        SELECT DISTINCT a.agent_id, a.name 
        FROM agents a 
        WHERE a.agent_type != 'system' AND a.agent_type != 'human'
        ORDER BY a.agent_id  -- Order by actual agent ID
        """)
        
        current_agents = {row['name']: row['agent_id'] for row in self.cursor.fetchall()}
        
        # Build the table data - now ordered by agent_id
        table_data = []
        
        # Create a combined dict with agent_id as key
        all_agents = {}
        
        # Add agents from current table
        for name, agent_id in current_agents.items():
            all_agents[agent_id] = {
                'name': name,
                'rounds': agent_rounds.get(name, set())
            }
        
        # Sort by agent_id and build table
        for agent_id in sorted(all_agents.keys()):
            try:
                agent_data = all_agents[agent_id]
                rounds = sorted(agent_data['rounds'])
                
                # Convert rounds to display format
                rounds_display = []
                for r in rounds:
                    r_str = str(r)
                    rounds_display.append(r_str.replace('.', ''))
                rounds_str = ', '.join(rounds_display) if rounds_display else 'No rounds yet'
                
                table_data.append([agent_id, agent_data['name'], rounds_str])
            except Exception as e:
                print(f"Error processing agent_id {agent_id}: {e}")
                print(f"Agent data: {all_agents[agent_id]}")
                raise
        
        # Display the table
        headers = ["Agent ID", "Agent Name", "Rounds Participated"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return table_data


    def round_participation_summary(self):
        """
        Function 2: Create a table showing each round with timestamp and participating agents.
        Format: Round ID | Timestamp | Participating Agents (ID, Name pairs)
        """
        print("\n=== ROUND PARTICIPATION SUMMARY ===\n")
        
        # Get round metadata
        self.cursor.execute("""
        SELECT round_id, timestamp 
        FROM round_metadata 
        ORDER BY round_number, branch_path
        """)
        
        rounds = list(self.cursor.fetchall())
        
        if not rounds:
            print("No rounds found in round_metadata.")
            return []
        
        table_data = []
        
        for round_row in rounds:
            round_id = round_row['round_id']
            timestamp = round_row['timestamp']
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp_display = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_display = timestamp
            
            # Get agents who participated in this round from agents_library
            self.cursor.execute("""
            SELECT agent_name, rounds_participated 
            FROM agents_library 
            WHERE rounds_participated LIKE ?
            """, (f'%{round_id}%',))
            
            # Get ALL agents from agents_library and check their rounds
            self.cursor.execute("""
            SELECT al.agent_name, al.rounds_participated, a.agent_id
            FROM agents_library al
            LEFT JOIN agents a ON al.agent_name = a.name
            """)
            
            participating_agents = []
            
            for agent_row in self.cursor.fetchall():
                agent_name = agent_row['agent_name']
                actual_agent_id = agent_row['agent_id'] if agent_row['agent_id'] else '?'
                rounds_json = agent_row['rounds_participated'] if agent_row['rounds_participated'] else '[]'
                
                # Parse JSON and check if this round is in the list
                if rounds_json:
                    rounds = json.loads(rounds_json)
                    # Convert both to strings for comparison
                    if str(round_id) in [str(r) for r in rounds] or int(round_id) in rounds:
                        participating_agents.append(f"({actual_agent_id}, {agent_name})")
            
            # Join agent pairs
            agents_str = ', '.join(participating_agents) if participating_agents else 'No agents'
            
            # Clean round_id for display (remove dots)
            round_display = round_id.replace('.', '')
            
            table_data.append([round_display, timestamp_display, agents_str])
        
        # Display the table
        headers = ["Round ID", "Timestamp", "Participating Agents (ID, Name)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return table_data
    
    
    # def get_round_details(self, round_id):
    #     """
    #     Get detailed information about a specific round.
    #     """
    #     print(f"\n=== DETAILS FOR ROUND {round_id} ===\n")
        
    #     # Fetch metadata
    #     self.cursor.execute("SELECT round_id AS round, timestamp FROM round_metadata WHERE round_id = ?", (round_id,))
    #     meta = self.cursor.fetchone()

    #     # Get round metadata
    #     self.cursor.execute("""
    #     SELECT * FROM round_metadata 
    #     WHERE round_id = ?
    #     """, (round_id,))
        
    #     round_data = self.cursor.fetchone()
    #     if not round_data:
    #         print(f"Round {round_id} not found.")
    #         return
        
    #     print(f"Round: {round_id}")
    #     print(f"Timestamp: {round_data['timestamp']}")
    #     print(f"Database: {round_data['database_path']}")
        
    #     if round_data['branch_path']:
    #         print(f"Branch Path: {round_data['branch_path']}")
    #     if round_data['parent_round']:
    #         print(f"Parent Round: {round_data['parent_round']}")
        
    #     # Get participating agents
    #     self.cursor.execute("""
    #     SELECT agent_name, memory_note, specializations, rounds_participated
    #     FROM agents_library 
    #     WHERE rounds_participated LIKE ?
    #     """, (f'%{round_id}%',))
        
    #     agents = []
    #     for row in self.cursor.fetchall():
    #         # Verify this agent actually participated
    #         rounds = json.loads(row['rounds_participated'])
    #         if round_id in rounds:
    #             agents.append({
    #                 'name': row['agent_name'],
    #                 'memory_note': row['memory_note'][:100] + '...' if row['memory_note'] and len(row['memory_note']) > 100 else (row['memory_note'] or ''), 
    #                'specializations': json.loads(row['specializations']) if row['specializations'] else []
    #             })
        
    #     if agents:
    #         print(f"\nParticipating Agents ({len(agents)}):")
    #         for i, agent in enumerate(agents, 1):
    #             print(f"\n{i}. {agent['name']}")
    #             if agent['specializations']:
    #                 print(f"   Specializations: {', '.join(agent['specializations'])}")
    #             print(f"   Memory Note: {agent['memory_note']}")

    #     return {
    #         "round": meta["round"],
    #         "timestamp": meta["timestamp"],
    #         "database": self.db_path,
    #         "agents": agents
    #     }

    def get_round_details(self, round_id):
        """
        Returns a dict:
          {
            "round": <round_id>,
            "timestamp": <ISO timestamp string>,
            "database": <path to your SQLite file>,
            "agents": [
               {"name": ..., "memory_note": ..., "specializations": [...]},
               ...
            ]
          }
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # 1) Fetch the round metadata
        cur.execute("""
            SELECT round_id AS round,
                   timestamp
              FROM round_metadata
             WHERE round_id = ?
        """, (round_id,))
        meta = cur.fetchone()
        if not meta:
            raise ValueError(f"Round '{round_id}' not found.")

        # 2) Scan every agent row and filter by participation
        cur.execute("""
            SELECT agent_name,
                   memory_note,
                   specializations,
                   rounds_participated
              FROM agents_library
        """)
        agents = []
        for row in cur.fetchall():
            # Load the JSON array (fallback to empty list on bad/missing data)
            try:
                raw = row["rounds_participated"] or "[]"
                participated_raw = json.loads(raw)
            except json.JSONDecodeError:
                participated_raw = []

            # Normalize everything to strings so "2" matches 2
            participated = [str(x) for x in participated_raw]

            if str(round_id) in participated:
                # Parse specializations JSON
                try:
                    specs = json.loads(row["specializations"] or "[]")
                except json.JSONDecodeError:
                    specs = []

                # Truncate memory note for display
                note = row["memory_note"] or ""
                if len(note) > 100:
                    note = note[:100] + "…"

                agents.append({
                    "name": row["agent_name"],
                    "memory_note": note,
                    "specializations": specs
                })

        conn.close()

        return {
            "round": meta["round"],
            "timestamp": meta["timestamp"],
            # Use the actual DB path passed into your analyzer, not a column
            "database": self.db_path,
            "agents": agents
        }

    def get_agent_round_messages(self, agent_id, round_id, save_to_file=True):
        """
        Get all messages sent and received by an agent during a specific round.
        
        :param agent_id: The agent's database ID
        :param round_id: The round ID (e.g., "1" or "5.a")
        :param save_to_file: Whether to save output to a file
        """
        print(f"\n=== MESSAGE HISTORY FOR AGENT {agent_id} IN ROUND {round_id} ===\n")
        
        # First, get the agent's name
        try:
            self.cursor.execute("SELECT name FROM agents WHERE agent_id = ?", (agent_id,))
            agent_result = self.cursor.fetchone()
            
            if not agent_result:
                print(f"Agent with ID {agent_id} not found.")
                # Let's check what agents exist
                self.cursor.execute("SELECT agent_id, name FROM agents WHERE agent_type != 'system'")
                available = self.cursor.fetchall()
                if available:
                    print("\nAvailable agents:")
                    for a in available:
                        print(f"  - ID {a['agent_id']}: {a['name']}")
                return
            
            agent_name = agent_result['name']
            self.target_agent_name = agent_name  # Store for use in colorizing
            print(f"Agent: {Fore.RED}{agent_name}{Style.RESET_ALL} (ID: {agent_id})\n")
        except Exception as e:
            print(f"Error looking up agent: {e}")
            # Try a simpler query
            self.cursor.execute("SELECT COUNT(*) as count FROM agents")
            count = self.cursor.fetchone()
            print(f"Total agents in database: {count['count']}")
            raise

        # Set up file output if requested
        output_content = []
            
        def print_and_collect(text=""):
            """Print to console and collect for file output"""
            print(text)
            if save_to_file:
                output_content.append(text)

        # Create output directory
        if save_to_file:
            output_dir = "agent_message_logs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename: agentId_agentName_roundId.txt
            round_str = round_id.replace('.', '')  # Clean round ID for filename
            filename = f"{agent_id}_{agent_name}_{round_str}.txt"
            filepath = os.path.join(output_dir, filename)
        
        # Now use print_and_collect instead of print for all output
        print_and_collect(f"\n=== MESSAGE HISTORY FOR AGENT {agent_id} IN ROUND {round_id} ===\n")
        print_and_collect(f"Agent: {agent_name} (ID: {agent_id})\n")
        
        # Convert round_id to round_number for the query
        round_number = int(round_id.split('.')[0])
        
        # Get all messages where this agent was sender or receiver
        # First, get the timestamp of the first round message for this agent
        self.cursor.execute("""
        SELECT MIN(timestamp) as first_round_timestamp
        FROM messages 
        WHERE conversation_round = ?
        AND (sender_id = ? OR receiver_id = ?)
        """, (round_number, agent_id, agent_id))
        
        first_round_time = self.cursor.fetchone()['first_round_timestamp']
        
        # First, get all messages for this round to find which agents are actually mentioned
        self.cursor.execute("""
        SELECT DISTINCT sender_id, receiver_id
        FROM messages m
        WHERE (
            (m.conversation_round = ? AND (m.sender_id = ? OR m.receiver_id = ?))
            OR (m.conversation_round = 0 AND m.receiver_id = ? AND m.timestamp < ?)
        )
        """, (round_number, agent_id, agent_id, agent_id, first_round_time))

        mentioned_agent_ids = set()
        for row in self.cursor.fetchall():
            if row['sender_id'] and row['sender_id'] != 0:
                mentioned_agent_ids.add(row['sender_id'])
            if row['receiver_id'] and row['receiver_id'] != 0:
                mentioned_agent_ids.add(row['receiver_id'])

        # Now pre-populate colors only for agents who actually participate in this round
        if mentioned_agent_ids:
            placeholders = ','.join(['?' for _ in mentioned_agent_ids])
            self.cursor.execute(f"""
            SELECT agent_id, name 
            FROM agents 
            WHERE agent_id IN ({placeholders})
            """, list(mentioned_agent_ids))
            
            for row in self.cursor.fetchall():
                if row['agent_id'] != agent_id:  # Don't assign color to target agent
                    _ = self.get_agent_color(row['name'], row['agent_id'], agent_id)

        # Now get messages, limiting round 0 messages to those before the round started
        self.cursor.execute("""
        SELECT 
            m.message_id,
            m.sender_id,
            m.receiver_id,
            m.message,
            m.message_type,
            m.timestamp,
            m.message_sequence,
            sender.name as sender_name,
            receiver.name as receiver_name
        FROM messages m
        JOIN agents sender ON m.sender_id = sender.agent_id
        LEFT JOIN agents receiver ON m.receiver_id = receiver.agent_id
        WHERE (
            (m.conversation_round = ? AND (m.sender_id = ? OR m.receiver_id = ?))
            OR (m.conversation_round = 0 AND m.receiver_id = ? AND m.timestamp < ?)
        )
        AND m.message_type NOT IN ('container_assignment', 'credit_assignment', 'memory_reset')
        ORDER BY m.timestamp, m.message_sequence
        """, (round_number, agent_id, agent_id, agent_id, first_round_time))
        
        messages = self.cursor.fetchall()
        
        if not messages:
            print(f"No messages found for agent {agent_name} in round {round_id}")
            return
        
        # Format and display messages
        for msg in messages:
            sender_name = msg['sender_name']
            sender_id = msg['sender_id']
            receiver_name = msg['receiver_name'] if msg['receiver_name'] else "All"
            receiver_id = msg['receiver_id'] if msg['receiver_id'] else "broadcast"
            message_text = msg['message']
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(msg['timestamp'])
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = msg['timestamp']
            
            # Get colors
            sender_color = self.get_agent_color(sender_name, sender_id, agent_id)
            receiver_color = self.get_agent_color(receiver_name, receiver_id, agent_id) if receiver_name else ""
            
            # Format based on sender
            if sender_id == 0:
                # System message - colorize agent names in the message
                colored_message = self.colorize_agent_names_in_text(message_text)
                print(f"[{time_str}] {sender_color}System:{Style.RESET_ALL} {colored_message}")
            elif sender_id == agent_id:
                # Message from our agent
                colored_message = self.colorize_agent_names_in_text(message_text)
                if receiver_id == 0:
                    print(f"[{time_str}] {sender_color}{agent_name}, {agent_id}:{Style.RESET_ALL} {colored_message}")
                else:
                    print(f"[{time_str}] {sender_color}{agent_name}, {agent_id}{Style.RESET_ALL} -> {receiver_color}{receiver_name}:{Style.RESET_ALL} {colored_message}")
            else:
                # Message to our agent
                colored_message = self.colorize_agent_names_in_text(message_text)
                print(f"[{time_str}] {sender_color}{sender_name}, {sender_id}:{Style.RESET_ALL} {colored_message}")
            
            #print()  # Empty line between messages
                
        if save_to_file:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_content))
            print(f"\n[Output saved to: {filepath}]")
        
        print(f"\nTotal messages: {len(messages)}")


    def colorize_agent_names_in_text(self, text):
        """Colorize all agent names found in the text"""
        # First, get all agent names and their colors
        colored_text = text
        
        # Color system mentions (case-insensitive)
        import re
        colored_text = re.sub(r'\b(system)\b', f'{Fore.BLUE}\\1{Style.RESET_ALL}', colored_text, flags=re.IGNORECASE)
        
        # Get all agents and color their names
        for name, color in self.agent_color_map.items():
            if name in colored_text:
                colored_text = colored_text.replace(name, f'{color}{name}{Style.RESET_ALL}')
        
        # Also color the target agent name if it appears
        if hasattr(self, 'target_agent_name') and self.target_agent_name in colored_text:
            colored_text = colored_text.replace(self.target_agent_name, f'{Fore.RED}{self.target_agent_name}{Style.RESET_ALL}')
        
        return colored_text

    def debug_agent_init(self, agent_id):
        """Debug: Show initialization messages for an agent"""
        print(f"\n=== DEBUG: INITIALIZATION MESSAGES FOR AGENT {agent_id} ===\n")
        
        self.cursor.execute("""
        SELECT 
            m.conversation_round,
            m.message_type,
            m.sender_id,
            m.message
        FROM messages m
        WHERE m.receiver_id = ?
        AND m.conversation_round IN (0, 1)
        AND m.sender_id = 0
        ORDER BY m.timestamp
        LIMIT 10
        """, (agent_id,))
        
        for row in self.cursor.fetchall():
            print(f"Round: {row['conversation_round']}")
            print(f"Type: {row['message_type']}")
            print(f"Message: {row['message'][:200]}...")
            print("-" * 50)

    def get_agent_color(self, agent_name, agent_id, target_agent_id):
        """Get consistent color for an agent"""
        if agent_id == 0 or (isinstance(agent_name, str) and agent_name.lower() == "system"):            
            return Fore.BLUE  # System
        elif agent_id == target_agent_id:
            return Fore.RED  # Target agent we're querying
        else:
            # Assign colors consistently to other agents
            if agent_name not in self.agent_color_map:
                self.agent_color_map[agent_name] = self.agent_colors[self.next_color_index % len(self.agent_colors)]
                self.next_color_index += 1
            return self.agent_color_map[agent_name]
        

    def get_round_strategy_feedback(self, round_id):
        """
        Get strategy feedback from all agents for a specific round.
        
        :param round_id: The round ID (e.g., "1" or "5.a")
        """
        print(f"\n=== STRATEGY FEEDBACK FOR ROUND {round_id} ===\n")
        
        # Create output directory
        output_dir = "memory_notes"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert round_id to round_number
        round_number = int(round_id.split('.')[0])
        
        # Get all strategy feedback messages for this round
        self.cursor.execute("""
        SELECT 
            m.message,
            m.sender_id,
            a.name as agent_name
        FROM messages m
        JOIN agents a ON m.sender_id = a.agent_id
        WHERE m.conversation_round = ?
        AND m.message_type = 'strategy_feedback_response'
        ORDER BY m.sender_id
        """, (round_number,))
        
        feedback_data = []
        output_lines = []
        
        for row in self.cursor.fetchall():
            agent_name = row['agent_name']
            agent_id = row['sender_id']
            feedback = row['message']
            
            feedback_data.append({
                'agent_id': agent_id,
                'agent_name': agent_name,
                'feedback': feedback
            })
            
            # Format for output
            output_lines.append(f"=== Agent: {agent_name} (ID: {agent_id}) ===")
            output_lines.append(feedback)
            output_lines.append("")  # Empty line between agents
            
            # Also print to console
            print(f"Agent: {agent_name} (ID: {agent_id})")
            print(feedback)
            print()
        
        # Save to file
        filename = f"{round_id}_strategy.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"[Strategy feedback saved to: {filepath}]")
        return feedback_data

    def get_round_memory_notes(self, round_id):
        """
        Get memory notes from all agents for a specific round.
        
        :param round_id: The round ID (e.g., "1" or "5.a")
        """
        print(f"\n=== MEMORY NOTES FOR ROUND {round_id} ===\n")
        
        # Create output directory
        output_dir = "memory_notes"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert round_id to round_number
        round_number = int(round_id.split('.')[0])
        
        # Get all memory response messages for this round
        self.cursor.execute("""
        SELECT 
            m.message,
            m.sender_id,
            a.name as agent_name
        FROM messages m
        JOIN agents a ON m.sender_id = a.agent_id
        WHERE m.conversation_round = ?
        AND m.message_type = 'memory_response'
        ORDER BY m.sender_id
        """, (round_number,))
        
        memory_data = []
        output_lines = []
        
        for row in self.cursor.fetchall():
            agent_name = row['agent_name']
            agent_id = row['sender_id']
            memory_note = row['message']
            
            memory_data.append({
                'agent_id': agent_id,
                'agent_name': agent_name,
                'memory_note': memory_note
            })
            
            # Format for output
            output_lines.append(f"=== Agent: {agent_name} (ID: {agent_id}) ===")
            output_lines.append(memory_note)
            output_lines.append("")  # Empty line between agents
            
            # Also print to console
            print(f"Agent: {agent_name} (ID: {agent_id})")
            print(memory_note)
            print()
        
        # Save to file
        filename = f"{round_id}_memory_notes.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"[Memory notes saved to: {filepath}]")
        return memory_data

    def get_round_general_feedback(self, round_id):
        """
        Get general feedback from all agents for a specific round.
        
        :param round_id: The round ID (e.g., "1" or "5.a")
        """
        print(f"\n=== GENERAL FEEDBACK FOR ROUND {round_id} ===\n")
        
        # Create output directory
        output_dir = "memory_notes"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert round_id to round_number
        round_number = int(round_id.split('.')[0])
        
        # Get all general feedback response messages for this round
        self.cursor.execute("""
        SELECT 
            m.message,
            m.sender_id,
            a.name as agent_name
        FROM messages m
        JOIN agents a ON m.sender_id = a.agent_id
        WHERE m.conversation_round = ?
        AND m.message_type = 'general_feedback_response'
        ORDER BY m.sender_id
        """, (round_number,))
        
        feedback_data = []
        output_lines = []
        
        for row in self.cursor.fetchall():
            agent_name = row['agent_name']
            agent_id = row['sender_id']
            feedback = row['message']
            
            feedback_data.append({
                'agent_id': agent_id,
                'agent_name': agent_name,
                'feedback': feedback
            })
            
            # Format for output
            output_lines.append(f"=== Agent: {agent_name} (ID: {agent_id}) ===")
            output_lines.append(feedback)
            output_lines.append("")  # Empty line between agents
            
            # Also print to console
            print(f"Agent: {agent_name} (ID: {agent_id})")
            print(feedback)
            print()
        
        # Save to file
        filename = f"{round_id}_feedback.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"[General feedback saved to: {filepath}]")
        return feedback_data
    
    def search_agent_messages(self, search_term):
        """
        Search for a specific word/phrase in agent messages (excluding system messages).
        
        :param search_term: The word or phrase to search for
        """
        print(f"\n=== SEARCHING FOR '{search_term}' IN AGENT MESSAGES ===\n")
        
        # Search for the term in messages (case-insensitive)
        self.cursor.execute("""
        SELECT 
            m.sender_id as agent_id,
            a.name as agent_name,
            m.conversation_round,
            m.message,
            m.timestamp,
            m.message_type
        FROM messages m
        JOIN agents a ON m.sender_id = a.agent_id
        WHERE m.sender_id != 0  -- Exclude system messages
        AND m.message LIKE ?
        ORDER BY m.conversation_round, m.timestamp
        """, (f'%{search_term}%',))
        
        results = self.cursor.fetchall()
        
        if not results:
            print(f"No messages found containing '{search_term}'")
            return []
        
        print(f"Found {len(results)} messages containing '{search_term}':\n")
        
        # Group by round for better readability
        current_round = None
        search_results = []
        
        for row in results:
            agent_id = row['agent_id']
            agent_name = row['agent_name']
            round_number = row['conversation_round']
            message = row['message']
            message_type = row['message_type']
            
            # Convert round_number to round_id (for now just using the number)
            # In the future, you might want to look up the actual round_id from round_metadata
            round_id = str(round_number)
            
            # Print with grouping by round
            if round_number != current_round:
                current_round = round_number
                print(f"\n--- Round {round_id} ---")
            
            print(f"\nAgent: {agent_name} (ID: {agent_id})")
            print(f"Type: {message_type}")
            
            # Highlight the search term in the message
            highlighted_message = message.replace(
                search_term, 
                f"{Fore.YELLOW}{search_term}{Style.RESET_ALL}"
            )
            # Also highlight case variations
            import re
            highlighted_message = re.sub(
                re.escape(search_term), 
                f"{Fore.YELLOW}\\g<0>{Style.RESET_ALL}", 
                highlighted_message, 
                flags=re.IGNORECASE
            )
            
            print(f"Message: {highlighted_message}")
            print("-" * 80)
            
            search_results.append({
                'agent_id': agent_id,
                'agent_name': agent_name,
                'round_id': round_id,
                'message': message,
                'message_type': message_type
            })
        
        return search_results
    
    def get_rounds(self):
        "Return a list of all round IDs in the database."
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT round_id FROM round_metadata")
        return [row['round_id'] for row in cursor.fetchall()]
    
    def plot_round_messages(self, round_id, out_path=None):
        """Plot message counts over time for a given round. 
        Implement using Matplotlib and save to out_path if provided."""
        # TODO: implement plotting logic here
        pass


def main():
    parser = argparse.ArgumentParser(description='Analyze economic rounds and agent participation.')
    parser.add_argument('database', help='Path to the SQLite database file')
    parser.add_argument('--action', choices=['agents', 'rounds', 'round_detail', 'agent_messages', 'debug', 'strategy', 'memory', 'feedback', 'search'], 
                default='agents', help='Analysis action to perform')
    parser.add_argument('--round', help='Round ID for detailed view (e.g., "5" or "5.a")')
    parser.add_argument('--agent', type=int, help='Agent ID for agent-specific queries')
    parser.add_argument('--search', help='Search term for message search')
    
    args = parser.parse_args()
    
    try:
        analyzer = EconomicAnalyzer(args.database)
        
        if args.action == 'agents':
            analyzer.agent_participation_summary()
        
        elif args.action == 'rounds':
            analyzer.round_participation_summary()
        
        elif args.action == 'round_detail':
            if not args.round:
                print("Error: --round required for round_detail action")
                return
            analyzer.get_round_details(args.round)
        elif args.action == 'agent_messages':
            if not args.agent or not args.round:
                print("Error: --agent and --round required for agent_messages action")
                return
            analyzer.get_agent_round_messages(args.agent, args.round)
        elif args.action == 'debug':
            if not args.agent:
                print("Error: --agent required for debug action")
                return
            analyzer.debug_agent_init(args.agent)
        elif args.action == 'strategy':
            if not args.round:
                print("Error: --round required for strategy action")
                return
            analyzer.get_round_strategy_feedback(args.round)
        elif args.action == 'memory':
            if not args.round:
                print("Error: --round required for memory action")
                return
            analyzer.get_round_memory_notes(args.round)
        elif args.action == 'feedback':
            if not args.round:
                print("Error: --round required for feedback action")
                return
            analyzer.get_round_general_feedback(args.round)
        elif args.action == 'search':
            if not args.search:
                print("Error: --search term required for search action")
                return
            analyzer.search_agent_messages(args.search)
        
        # If no specific action, show both summaries
        if args.action == 'agents':
            analyzer.round_participation_summary()
    
    except Exception as e:
        print(f"Error: {e}")


#if __name__ == "__main__":
#    main()
