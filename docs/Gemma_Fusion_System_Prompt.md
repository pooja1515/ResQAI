# Gemma Fusion System Prompt Architecture

ResQAI uses a modular System-of-Agents orchestration framework powered by Gemma 4.

Instead of relying on a single generic prompt, ResQAI separates reasoning into specialized intelligence agents designed for humanitarian disaster coordination.

---

# 1. Temporal Crisis Intelligence Agent

Source:
`src/resqai/memory/memory_reasoner.py`

```python
SYSTEM = """You are ResQAI's temporal crisis intelligence analyst.
```

### Responsibilities

- track evolving disaster escalation
- reason across timeline events
- identify crisis progression
- maintain operational memory consistency

---

# 2. Crisis Fusion Coordinator

Source:
`src/resqai/agents/fusion_coordinator.py`

```python
"ROLE: You are ResQAI's crisis fusion coordinator for emergency response operations."
```

### Responsibilities

- coordinate multimodal reasoning
- fuse vision + voice + weather + memory signals
- generate operational intelligence
- prioritize emergency actions

---

# 3. Voice Intelligence Semantic Agent

Source:
`src/resqai/pipelines/voice_intelligence/prompts.py`

```python
SYSTEM_INSTRUCTIONS = """You are ResQAI's emergency semantic reasoner.
```

### Responsibilities

- analyze multilingual distress speech
- extract urgency semantics
- identify trapped civilian indicators
- classify emergency severity

---

# 4. Multimodal Fusion Agent

Source:
`src/resqai/pipelines/multimodal/multimodal_prompts.py`

```python
SYSTEM_INSTRUCTIONS = """You are ResQAI's multimodal crisis fusion agent.
```

### Responsibilities

- synthesize multimodal disaster intelligence
- coordinate operational reasoning
- merge structured disaster signals
- support grounded emergency coordination

---

# 5. Grounded Disaster Intelligence Agent

Source:
`src/resqai/rag/prompts.py`

```python
SYSTEM_INSTRUCTIONS = """You are ResQAI's grounded disaster intelligence assistant.
```

### Responsibilities

- ground recommendations using RAG
- reduce hallucination risks
- prioritize verified humanitarian guidance
- enforce operational safety constraints

---

# 6. Weather Intelligence Analyst

Source:
`src/resqai/weather/weather_prompts.py`

```python
SYSTEM = """You are ResQAI's weather intelligence analyst for disaster response.
```

### Responsibilities

- analyze environmental escalation
- evaluate flood risk severity
- reason over weather-driven crisis progression
- support operational hazard forecasting

---

# Safety & Trust Design Philosophy

ResQAI uses:

- grounded retrieval
- modular reasoning agents
- temporal intelligence
- explainable AI
- operational safety prompting

to ensure reliable humanitarian intelligence generation during disaster-response scenarios.

The orchestration system prioritizes:

1. Human safety
2. Grounded retrieval evidence
3. Conservative operational guidance
4. Explainable reasoning
5. Human-in-the-loop coordination