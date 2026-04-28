# LLM-sandbox

A multi-agent simulation framework for studying whether **social norms and cooperation can emerge spontaneously among LLM agents** placed under economic pressure.

> **Status — research code.** Active prototype, single-developer. The economic game runs end-to-end, the SQLite logging is comprehensive, and an interactive analysis CLI is included. Some pieces are still rough — see [Roadmap](#roadmap) below. Contact: **a.u.seryakov@gmail.com** or Telegram **@andrey_seryakov**.

This work backs the research thesis described in [*Emergent AI Society: Tasks, Scarcity, Talks*](https://www.lesswrong.com/posts/FScgtna9Bt7kgLWLu/emergent-ai-society-tasks-scarcity-talks) (LessWrong, 2025) — that AI agents optimising for task completion and resource efficiency may develop social structure (cooperation, trust, norms) without consciousness or human design.

---

## The Container Game

Agents are placed in a resource-constrained economy and given **no instructions about cooperation, trust, or norms**. The only structural fact they know is: containers of the same colour-and-number share the same code across agents — so codes can, in principle, be shared.

- **Containers.** Each agent receives a hand of containers of different colours and numbers, each carrying a hidden numeric code.
- **Credits.** Each agent starts with credits equal to **120%** of the cost to open all their containers alone. A 20% margin is added for flexibility.
- **Shared codes.** Same colour + number → same code, across all agents. Agents who open a container can share its code with others.
- **Cost to open.** Spending credits reveals a code. Specialisation reduces cost for specific colours.
- **Communication.** Agents may negotiate to connect with each other (mutual choice required), exchange codes, deceive, or refuse.
- **Memory.** Conversation memory is wiped between rounds — agents preserve strategy only by writing **memory notes** to their future selves. Names and specialisations persist.
- **Branching timelines.** Round IDs like `7.a.3.b.5` track causal history when mixing agents from divergent runs (e.g. running an agent's earlier "self" alongside a later one).

What we look for: do agents discover sharing? Build trust networks? Punish defection? Develop reputational language? Evolve strategy across generations through their memory notes?

---

## Quickstart

```bash
git clone https://github.com/AndreySeryakov/LLM-sandbox.git
cd LLM-sandbox
pip install -r requirements.txt
cp .env.example .env              # then add your DeepSeek API key + prompts
python simulation.py
```

**Note on prompts.** The simulation reads several system-prompt strings from `.env` (`INTRODUCTORY_PROMPT`, `ROUND_START_PROMPT`, etc. — see `.env.example`). These define what the agents are told and are not yet published in the repo. Contact me for the current prompt set, or write your own based on the round structure described below.

**Try without an API key.** Set `DRY_RUN=true` in `.env` — the simulation runs end-to-end with mocked LLM responses, useful for inspecting the pipeline and estimating real-run costs.

**Inspect results.** After a run, launch the interactive analyser:

```bash
python econ.py path/to/database.db
```

Menu options: list rounds, view round details, agent participation, memory notes, strategy feedback, message search, etc.

---

## Repository Layout

| File / dir | What it is |
|---|---|
| `simulation.py` | Main simulation engine — agents, containers, conversation manager, economic round driver, DB writer. Currently a single file (~4k lines); a refactor into modules is parked on the `wip-refactor` branch. |
| `econ.py` | Interactive CLI for browsing a finished simulation's database. |
| `economic_analysis.py` | Analysis routines used by `econ.py`. |
| `db_explorer.py` | Lower-level DB browsing helper. |
| `conversation_logs/` | Per-run conversation transcripts (gitignored). |
| `memory_notes/` | Agents' memory notes between rounds (gitignored). |
| `agent_message_logs/` | Raw agent message logs (gitignored). |

---

## Configuration

### Environment variables (`.env`)

```bash
API_KEY=your_deepseek_api_key
TEMPERATURE=1.0          # LLM sampling temperature
DRY_RUN=false            # true → run the pipeline without API calls

# System prompts — see code for current defaults.
INTRODUCTORY_PROMPT="..."
ROUND_START_PROMPT="..."
TASK_INTRODUCTION_PROMPT="..."
COLLECT_OPENING_ACTIONS_PROMPT="..."
SHARING_ACTIONS_PROMPT="..."
MEMORY_REQUEST_PROMPT="..."
GENERAL_FEEDBACK_REQUEST_PROMPT="..."
```

### Container distribution

```python
container_config = {
    'colors': ['red', 'blue', 'green'],
    'numbers': [1, 2, 3],
    'base_costs': {'red': 10, 'blue': 15, 'green': 20},
    'distribution_mode': 'controlled',  # 'controlled' | 'fixed' | 'random'
    'controlled_distribution': {
        'red-1': ['Agent1', 'Agent2'],
        # ...
    }
}
```

---

## What Each Round Does

1. **Container distribution** — agents receive their hand for the round.
2. **Introduction** — agents learn their containers, recall prior memory notes.
3. **Negotiation** — pairwise mutual-connection conversations to arrange code sharing.
4. **Opening actions** — each agent decides which of their containers to open with credits.
5. **Sharing actions** — agents send codes per agreements made (or break them).
6. **Verification** — system validates shared codes, flags deceptions.
7. **Auto-resolve** — any unopened containers are auto-opened at full cost.
8. **Feedback & memory** — agents reflect on the round and write memory notes for their future selves.

---

## Database Schema (SQLite)

| Table | Contents |
|---|---|
| `agents` | Agent profiles and specialisations |
| `conversations` | Conversation metadata and outcomes |
| `messages` | Full message history with categorisation |
| `conversation_summaries` | Agent-generated conversation summaries |
| `agents_library` | Historical agent state across rounds (memory notes, specialisations) |
| `round_metadata` | Timeline and branching information |
| `branch_points` | Branch creation records |

Every action, message, decision, and outcome is logged. Replays and post-hoc analysis are first-class.

---

## Roadmap

List of what's not done and what comes next.

**Currently rough:**
- The simulation lives in one ~4k-line file; the in-progress split into `agents.py` / `containers.py` / `conversation.py` / `database.py` / `simulation.py` is parked on the `wip-refactor` branch.
- Default system prompts are in code rather than published as a clean `prompts/` folder.
- No bundled example database / sample-output figure for first-time visitors. Coming.
- No tests.

**Planned experiments:**
- Birth of cooperation (2–10 agents, no priors).
- Specialisation and guilds (~20 agents, comparative-advantage costs).
- Generations and culture — onboarding new agents into existing conventions.
- Reputation system, market-based code trading, multi-round tournaments, agent personality parameters.
- Misinformation and meme spread; resistance to collusion.

**Note on agent instructions:** Agents are *not* prompted to cooperate, build trust, or form norms. They are told only how the game works and that other agents may have matching containers. Anything social that emerges, emerges.

---

## Citation

```
Seryakov, A. (2025). Emergent AI Society: Tasks, Scarcity, Talks. LessWrong.
```

If you use the platform in research, please cite the essay above and link to this repository.

---

## Contact

- **Email:** a.u.seryakov@gmail.com
- **Telegram:** @andrey_seryakov

Questions, collaboration ideas, and mentee/contributor enquiries are all welcome.
