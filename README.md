## Agentic AI Behavioral Profiling System for HR Interviews

### Scope

- **Inputs**: Real-time video and audio streams from a custom meeting application.  
- **Analysis**: Emotion detection, tonal analysis, behavioral profiling (e.g., confidence, honesty).  
- **Interaction**: Dynamic question generation, natural responses to candidate queries, calibration through small talk.  
- **Features**: Lie detection, personalized calibration, context retention, post-interview reporting.  
- **Integration**: Streaming API interface with the meeting application.  
- **APIs**: Groq (STT, TTS, LLM tiers) and Gemini (LLM) free tiers.  
- **Constraints**: No local file I/O or hardware dependencies; optimized for free-tier API usage.

---

## System Overview

An Agentic AI framework where autonomous agents collaborate to perceive, decide, and act, aiming to conduct effective interviews. Designed for low latency, scalability, modularity, and privacy compliance.

### Key Principles

- **Autonomy**: Agents independently handle tasks and adapt in real time.  
- **Goal-Driven**: Maximizes interview insights and profiling accuracy.  
- **Real-Time**: Analysis and responses occur within milliseconds.  
- **Modularity**: Decoupled components allow easy feature updates.  
- **Privacy**: Data encryption and GDPR/CCPA compliance.

---

## Architecture

Microservices-based architecture with each agent responsible for a specific role. Agents communicate via message queues or WebSockets for real-time performance.

### Core Agents

- **Input Agent**: Ingests and tags video/audio streams.  
- **Video Analysis Agent**: Detects facial expressions and emotions.  
- **Audio Analysis Agent**: Transcribes speech and analyzes tone.  
- **Profiling Agent**: Builds and updates behavioral profiles, flags potential dishonesty.  
- **Question Generation Agent**: Crafts context-relevant, dynamic questions.  
- **Interaction Agent**: Manages conversational flow and candidate queries.  
- **Calibration Agent**: Establishes baseline behavior through initial interaction.  
- **Knowledge Base Agent**: Manages context and embeddings for continuity.  
- **Report Agent**: Compiles profiles and insights into final reports.  
- **Central Orchestrator Agent**: Coordinates all agents to ensure low-latency, adaptive interviewing.

---

## Technical Specifications

### Tech Stack

- **Backend**: Python with asyncio for asynchronous processing and agent orchestration.  
- **Frontend & Dashboard**: Appwright for building the candidate and HR-facing UI components.  
- **Video Analysis**: OpenCV and TensorFlow Lite for real-time emotion detection.  
- **Audio Analysis**: Librosa for tonal feature extraction and Groq STT for transcription.  
- **NLP & Agent Orchestration**:  
  - Groq LLM (lightweight, normal, thinking) for profiling, calibration, and quick responses.  
  - Gemini LLM for dynamic question generation.  
  - LangChain to coordinate multi-agent workflows and prompt management.  
- **Vector Database**: Qdrant as the primary vector store for embeddings; Pinecone or FAISS as fallback options.  
- **Real-Time Media**: LiveKit for low-latency video/audio streaming and session management.  
- **TTS**: Groq TTS for natural speech synthesis.  
- **Knowledge Base**: Qdrant (vector embeddings) managed via LangChain retrieval chains.  
- **Communication**: WebSockets for real-time data streams; RabbitMQ for decoupled messaging.  
- **Deployment**: Docker containers deployed on AWS or GCP free-tier infrastructure.  
- **Monitoring & Metrics**: Prometheus for metrics collection; Grafana for visualization.

### 

### API Integration

- **Groq**  
  - STT endpoint for audio transcription.  
  - TTS endpoint for speech generation.  
  - Chat endpoints for tonal analysis, profiling, and report analysis.  
- **Gemini**  
  - Content generation endpoint for dynamic question creation.

### Rate Limits & Optimization

- Batch lightweight Groq calls.  
- Limit Gemini calls per interview cycle.  
- Implement caching to reduce redundant requests.

## Testing Strategy

- **Unit Tests**: Individual agent functionality and model accuracy.  
- **Integration Tests**: End-to-end interview workflows.  
- **Stress Tests**: Concurrent session handling.  
- **User Feedback**: HR review and validation of report usefulness.

---

## Challenges & Mitigations

- **API Limits**: Batch requests and cache common queries.  
- **Latency**: Sample video frames, use lightweight models, parallel processing.  
- **Accuracy**: Model fine-tuning and confidence score reporting.  
- **Natural Interaction**: Continuous LLM training and conversational dataset testing.  
- **Privacy**: TLS encryption and secure data storage.  
- **Scalability**: Containerization and autoscaling on cloud platforms.

---

## API Endpoints

- **WebSocket** `/stream`  
    
  - Streams video/audio data and returns questions/responses.  
  - **Input**: Tagged video and audio streams with metadata.  
  - **Output**: Question and response messages.


- **GET** `/report`  
    
  - Retrieves post-interview report.  
  - **Query**: Session identifier.  
  - **Response**: JSON report including summary, profile scores, flags, and trends.


- **GET** `/health`  
    
  - Checks system health.  
  - **Response**: Status and agent availability.

---

## Error Handling

- Standardized error codes for stream failures, API limits, and other exceptions.  
- Descriptive messages to aid debugging and recovery.

---

