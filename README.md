# Multi-Agent Conversation Framework

> **Note:** This README was automatically generated by Claude on March 30, 2025.
> **Note:** Andrey: I modified it a bit.

## Overview

This project implements a multi-agent conversation framework that allows LLM-based agents to interact with each other autonomously. The system facilitates agent-to-agent communication with comprehensive logging and analysis capabilities. LLMs can engage in conversations with one another, discuss topics, create summaries, and build on each other's ideas.

> **Note:** The current topic chosen for LLMs to discuss (AGI) is just for testing purposes. The framework is designed to be topic-agnostic and can be adapted for various conversational tasks.

## Features

- Creation of multiple LLM-based agents with unique identities
- Agent matchmaking system for establishing conversations
- Structured conversation sessions with configurable turn limits
- Comprehensive logging of all interactions
- Conversation summaries and learning extraction
- SQLite database for persistent storage of all interactions
- Database exploration tools for analysis

## Components

### 1. Database Schema

The system uses SQLite with the following tables:
- `agents`: Stores agent information and metadata
- `conversations`: Tracks conversation sessions and their status
- `messages`: Records all messages exchanged between agents
- `conversation_summaries`: Stores agent-generated summaries of conversations

### 2. Agent Class

The `Agent` class encapsulates an LLM instance and provides methods for:
- Initializing with a unique identity
- Sending and receiving messages
- Generating conversation summaries
- Updating its knowledge based on interactions

### 3. Conversation Manager

The `ConversationManager` handles:
- Matchmaking between agents
- Managing conversation rounds
- Enforcing turn limits
- Requesting conversation summaries
- Broadcasting system messages

### 4. Database Explorer

The `DatabaseExplorer` provides tools for:
- Listing all agents and conversations
- Viewing detailed conversation records
- Analyzing message patterns
- Searching for specific content
- Generating statistics

## Setup and Usage

### Prerequisites

- Python 3.8+
- Check the code required libraries 

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your API keys in a .env file:
```
API_KEY=your_deepseek_api_key_here
```

### Running the System

To start a conversation between agents:

```bash
python LLMs_interactions_1.py
```

This will:
1. Create a timestamped SQLite database
2. Initialize multiple agents
3. Allow agents to choose unique names
4. Run conversation rounds where agents connect with each other
5. Log all interactions to the database

### Analyzing Results

To explore the conversation data:

```bash
python db_explorer.py conversation_logs/conversation_logs_YYYYMMDD_HHMMSS.db --action [action]
```

Available actions:
- `agents`: List all agents
- `conversations`: List all conversations
- `conversation --id [ID]`: View details of a specific conversation
- `agent_messages --id [ID]`: List all messages sent/received by an agent
- `stats`: Display statistics about conversations
- `search --search [term]`: Search for messages containing a specific term

## Customization

The system can be customized by modifying:

- `SYSTEM_PROMPT`: Initial prompt for agent initialization
- `TASK_INTRODUCTION_PROMPT`: Task description for agents
- Conversation parameters (max turns, termination signals)
- The LLM model (currently using DeepSeek)
