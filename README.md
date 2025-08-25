Jarvis - A Local AI Voice Assistant
This is a personal project to create a voice-activated AI assistant that runs entirely on your local machine. The assistant listens for a wake word ("Hey Jarvis"), transcribes your command, determines your intent, and responds either by performing a task or engaging in conversation.

This project was built as a learning exercise. The code is a work-in-progress and serves as an example of integrating various local AI/ML tools.

Features
Wake Word Detection: Uses Picovoice Porcupine to listen for "Hey Jarvis."

Local Transcription: Employs faster-whisper for fast and accurate speech-to-text.

Local LLM: Connects to a local large language model (like Gemma 3) via Ollama for intent classification and chat.

Agentic Tasks: Integrates smol-agents to execute tasks like running code or searching.

Local Text-to-Speech: Uses Piper for clear, natural-sounding voice responses.

