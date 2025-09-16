# University Test Centre Chatbot

An AI-powered chatbot designed to assist Ontario Tech University students with Test Centre-related questions, including bookings, accommodations, locations, and contact information. The chatbot combines semantic FAQ search with a language model to provide accurate, context-aware responses.

## Features
- **Query-Type Detection:** Automatically identifies questions about location, booking, contact, and accommodations.  
- **Semantic FAQ Matching:** Uses `SentenceTransformers (MiniLM)` and `cosine similarity` for precise answers.  
- **LLM Fallback:** Generates conversational responses with `Ollama (Qwen)` when FAQ match is insufficient.  
- **Frontend:** React-based interface with real-time typing indicators and session-based chat history.  
- **Backend:** Flask API with CORS, parallel processing using `ThreadPoolExecutor`, and RESTful endpoints.  
- **Robust UX:** Handles errors gracefully and maintains context for ongoing conversations.  

## Tech Stack
- **Frontend:** React, HTML, CSS, Lucide icons  
- **Backend:** Python, Flask, Flask-CORS  
- **AI / NLP:** SentenceTransformers, SciPy, Ollama (Qwen)  
- **Concurrency:** ThreadPoolExecutor for parallel response handling  

## Getting Started
1. Clone the repo:  
```bash
git clone https://github.com/yourusername/university-test-centre-chatbot.git
```
2. Install backend dependencies:
```
pip install -r requirements.txt
```
3. Start Flask server:
```
python app.py
```
4. Start React frontend:
```
npm install
npm start
```
5. Open http://localhost:3000 in your browser.

## Usage

- Type your questions about Test Centre services (e.g., bookings, accommodations, locations).
- The chatbot responds using FAQ matching and AI-generated answers for clarity and context.
