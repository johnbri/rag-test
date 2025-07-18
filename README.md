**Setup:**

## RAG chatbot with local Ollama and Elevenlabs voice conversation 

#### **Required APIs**

1. **Create `.env` file** 

2. **ElevenLabs Voice API** (Required for voice features):
   - Sign up at [elevenlabs.io](https://elevenlabs.io)
   - Get your API key from profile settings
   - Add to `.env` file:
     ```bash
     ELEVENLABS_API_KEY=sk_your_elevenlabs_api_key_here
     ```

**File:** `voice_rag_streamlit.py`

1. Install Ollama: [ollama.com](https://ollama.com)
2. Pull required models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run voice_rag_streamlit.py`

## Usage

1. Upload documents using the sidebar
2. Click "Process Documents" to create embeddings/graphs
3. Ask questions about your documents in the chat interface