# ResQAI  
### Multimodal Disaster Intelligence Powered by Gemma

> Transforming fragmented disaster signals into coordinated operational intelligence using multimodal AI orchestration.

---

![ResQAI Banner](assets/media/resqai_hero_banner.png)

---

# Overview

Disasters generate fragmented and chaotic information streams:

- distress calls
- flood imagery
- environmental hazards
- evolving emergency timelines
- weather alerts
- evacuation requests

Emergency responders often struggle to synthesize these signals quickly during high-pressure situations.

ResQAI is a multimodal humanitarian intelligence system designed to transform disaster chaos into coordinated operational awareness using:

- Vision Intelligence
- Voice Intelligence
- Weather Intelligence
- Retrieval-Augmented Generation (RAG)
- Temporal Crisis Memory
- Geospatial Intelligence
- Explainable AI
- Gemma-powered Fusion Reasoning

---

# The Problem

During real-world disasters, critical information arrives fragmented across multiple modalities:

| Information Source | Challenge |
|---|---|
| Distress Calls | Language barriers and urgency detection |
| Flood Imagery | Difficult rapid assessment |
| Weather Alerts | Escalation uncertainty |
| Emergency Protocols | Information overload |
| Crisis Timelines | Tracking escalation over time |

This fragmentation slows decision-making during life-critical emergencies.

---

# The Solution — ResQAI

ResQAI acts as an AI-powered disaster intelligence coordination system.

It combines:
- flood image analysis
- multilingual emergency speech understanding
- grounded humanitarian guidance
- temporal memory reasoning
- weather-aware escalation prediction
- geospatial intelligence

into a unified operational reasoning framework powered by **Gemma**.

---

# Core Features

## Multimodal AI Orchestration

ResQAI combines:

- disaster imagery
- multilingual distress audio
- weather intelligence
- temporal crisis memory
- grounded emergency guidance

to generate coordinated operational intelligence.

---

## Vision Intelligence

### Capabilities
- Flood detection
- Disaster scene classification
- Explainable AI
- Grad-CAM visualization

### Model
- EfficientNetB0

---

## Voice Intelligence

### Capabilities
- Multilingual distress transcription
- Emergency urgency analysis
- Semantic distress understanding

### Supported Languages
- Hindi
- English
- French
- Spanish

### Models
- Whisper
- Gemma semantic reasoning

---

## Retrieval-Augmented Generation (RAG)

Grounded emergency intelligence using:

- Red Cross manuals
- evacuation protocols
- humanitarian response documents
- disaster preparedness guides

### Stack
- ChromaDB
- Sentence Transformers
- Semantic Retrieval

---

## Temporal Crisis Memory

Tracks:

- escalation trends
- evolving severity
- crisis progression
- recurring emergency patterns

This allows ResQAI to reason over time instead of isolated snapshots.

---

## Weather Intelligence

Integrates environmental intelligence:

- rainfall analysis
- flood escalation prediction
- storm risk reasoning
- operational hazard assessment

---

## Geospatial Intelligence

### Features
- hotspot visualization
- disaster mapping
- operational overlays
- crisis location intelligence

Built using:
- Folium
- OpenStreetMap
- Leaflet

---

## Streaming Operational Intelligence

ResQAI streams reasoning progressively:

- analyzing signals
- retrieving grounded guidance
- updating temporal memory
- synthesizing operational intelligence

This creates transparent AI coordination for responders.

---

# Hero Media Gallery

## ResQAI Hero Banner

![Hero Banner](assets/media/resqai_hero_banner.png)

---

# Multimodal Fusion Intelligence

ResQAI combines multiple intelligence streams into a unified humanitarian reasoning pipeline.

| Intelligence Source | Model/System |
|---|---|
| Vision Intelligence | EfficientNetB0 + Grad-CAM |
| Voice Intelligence | Whisper |
| Grounded Retrieval | ChromaDB + Semantic Search |
| Weather Intelligence | Environmental APIs |
| Temporal Memory | Crisis Event Tracking |
| Fusion Reasoning | Gemma |

Gemma acts as the central multimodal fusion layer, synthesizing fragmented disaster signals into coordinated operational intelligence.

---

# System Architecture

## Orchestration Flow

![Architecture Diagram](assets/architecture/resqai_architecture.png)

---

## Operational Pipeline

```text
Disaster Inputs
(Image + Audio + Text + Weather)
                ↓
        ResQAI Orchestrator
                ↓
 ┌───────────────┬───────────────┬───────────────┐
 │ Vision Agent │ Voice Agent  │ Weather Agent │
 └───────────────┴───────────────┴───────────────┘
                ↓
RAG + Temporal Memory + Geospatial Intelligence
                ↓
         Gemma Fusion Layer
                ↓
Operational Disaster Intelligence
                ↓
Responder Recommendations + Crisis Mapping
```

---

# Explainable AI — Grad-CAM

ResQAI integrates Grad-CAM explainability to visualize why the flood detection model flagged dangerous regions.

| Original Flood Image | Grad-CAM Heatmap |
|---|---|
| ![](assets/media/flood_original.png) | ![](assets/media/flood_gradcam.png) |

This improves:
- trust
- transparency
- operational interpretability

during AI-assisted disaster response.

---

# Streaming Intelligence

## Real-Time Thought Stream

![Streaming Intelligence](assets/media/streaming_reasoning.png)

ResQAI streams operational reasoning progressively:

```text
Analyzing crisis signals...

Retrieving grounded evacuation guidance...

Updating temporal memory...

Synthesizing operational intelligence...

Critical flood conditions detected.
Immediate evacuation recommended.
```

This creates a transparent AI coordination experience for emergency responders.

---

# Offline Humanitarian Intelligence

ResQAI is designed for edge-friendly deployment using local Gemma inference through Ollama.

![Local AI](assets/media/local_ai.png)

### Key Capabilities

- Local inference
- Offline reasoning
- MPS acceleration
- Low-connectivity resilience
- Humanitarian deployment readiness

This enables disaster intelligence workflows even in communication-disrupted environments.

---

# Gemma Integration

Gemma serves as the central reasoning engine of ResQAI.

## Gemma Responsibilities

- Multimodal fusion reasoning
- Operational intelligence synthesis
- Crisis prioritization
- Grounded response generation
- Semantic coordination
- Escalation analysis

---

## Gemma Deployment

ResQAI uses:

- Ollama
- local inference
- streaming orchestration
- system-prompted reasoning
- edge-friendly deployment workflows

### Gemma Context Optimization

ResQAI is designed to leverage:
- long-context reasoning
- structured operational prompts
- multimodal intelligence fusion
- chain-of-thought orchestration

for disaster-response scenarios.

---

# Repository Structure

```bash
ResQAI/
│
├── assets/
│   ├── architecture/
│   └── media/
│
├── components/
├── dataset/
├── events/
├── frontend/
├── outputs/
├── src/resqai/
├── tests/
├── ui/
│
├── Home.py
├── prepare_dataset.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Tech Stack

## AI/ML

- Gemma
- Ollama
- Whisper
- EfficientNetB0
- PyTorch
- TensorFlow
- Grad-CAM

---

## Retrieval

- ChromaDB
- Sentence Transformers

---

## Backend

- FastAPI
- Python

---

## Frontend

- Next.js
- React
- TailwindCSS

---

## Geospatial

- Folium
- Leaflet
- OpenStreetMap

---

# Installation

## 1. Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ResQAI
```

---

## 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

Install Ollama:

```bash
https://ollama.com/download
```

Pull Gemma:

```bash
ollama pull gemma4
```

Verify:

```bash
ollama list
```

---

# Run Backend

```bash
PYTHONPATH=src uvicorn resqai.main:app --host 127.0.0.1 --port 8000 --reload
```

---

# Run Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Example Usage

## Text-Based Crisis Reasoning

```text
Heavy flooding near Mumbai. Elderly civilians trapped inside homes.
```

---

## Multimodal Inputs

ResQAI supports:
- disaster images
- multilingual distress audio
- text reports
- location-aware intelligence

---

# Simulation Mode

ResQAI includes reproducible demo workflows using:

- sample flood imagery
- multilingual emergency audio
- orchestrator outputs
- geospatial overlays

This ensures:
- stable demonstrations
- reproducible evaluations
- presentation reliability

---

# Demo Assets

Included sample assets:

- `critical_hindi.wav`
- `sample.wav`
- flood imagery
- Grad-CAM overlays
- orchestrator outputs

---

# Future Work

- satellite intelligence integration
- drone-assisted disaster assessment
- responder coordination systems
- predictive escalation forecasting
- edge-device deployment
- offline humanitarian AI systems
- multimodal responder copilots

---

# Research Vision

ResQAI explores how multimodal AI systems can:

- reduce disaster response latency
- improve humanitarian coordination
- assist emergency responders
- increase operational transparency
- transform fragmented signals into actionable intelligence

---

# License

MIT License

---

# Acknowledgements

- Google Gemma
- Ollama
- OpenAI Whisper
- ChromaDB
- FastAPI
- Next.js
- OpenWeatherMap API
- OpenStreetMap

---

# ResQAI

### “Transforming disaster chaos into coordinated intelligence.”