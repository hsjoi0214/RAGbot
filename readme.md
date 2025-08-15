# RAGbot — Conversations with *Crime and Punishment*

## Objective

The goal of this project is to provide an **interactive Retrieval-Augmented Generation (RAG) chatbot** that allows users to explore the novel *Crime and Punishment* by Fyodor Dostoevsky in a conversational manner.  
By combining **document retrieval** with **large language model generation**, RAGbot delivers contextually accurate, memory-aware responses to literary and philosophical questions about the text.

---

## Project Strategy

This app follows a **retrieval + generation** architecture using [LlamaIndex](https://www.llamaindex.ai/), **HuggingFace embeddings**, and **Groq’s LLaMA 3 model**.

### Workflow Overview
1. **Document Loading**  
   The full text of *Crime and Punishment* (plaintext file) is ingested using `SimpleDirectoryReader`.
   
2. **Embedding & Indexing**  
   - Uses **sentence-transformers/all-MiniLM-L6-v2** for text embeddings.  
   - Indexed into a vector store for fast semantic search.
   
3. **Context Retrieval**  
   - Retrieves the top-*k* most relevant passages for each query.  
   - `top_k` is configurable in the UI.

4. **Generation with Context**  
   - Groq's **LLaMA 3.3 70B Versatile** model is used for answer generation.  
   - Responses are grounded in retrieved context to reduce hallucination.

5. **Memory-Aware Conversations**  
   - Maintains a buffer of conversation history so the chatbot can respond coherently over multiple turns.

---

## Why This Approach Works

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **HuggingFace Embeddings** | Encode text into vector space | Enables accurate semantic search |
| **VectorStoreIndex** | Store embeddings for fast retrieval | Low-latency, scalable context retrieval |
| **Groq LLaMA 3.3 70B** | Generate answers from context | High-quality, human-like responses |
| **ChatMemoryBuffer** | Store chat history | Provides conversational continuity |
| **Streamlit UI** | Easy web interface | Quick deployment & interaction |

---

## Tools & Libraries

- **Languages**: Python
- **Frameworks**: Streamlit, LlamaIndex
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Provider**: Groq (LLaMA 3.3 70B Versatile)
- **Others**: python-dotenv for secrets, pathlib for file handling

---

## Key Features

- **Semantic Search** — Retrieves the most relevant text excerpts from *Crime and Punishment*.
- **Memory-Aware Chat** — Keeps track of past exchanges for contextually coherent conversations.
- **Adjustable Context Depth** — `top_k` slider to control how many passages to retrieve.
- **Streamlit UI** — Simple, elegant web app interface.
- **Configurable API Keys** — Supports `.env` or `.streamlit/secrets.toml`.

---

## Live App

You can access the deployed RAGbot here:  
**[RAGbot Live on Streamlit](https://hsjoi0214-ragbot-appstreamlit-app-xxc6lx.streamlit.app/)**

---

## Architecture Diagram

![RAGbot Architecture](assets/ragbot_architecture.png)

**Architecture Steps:**
1. **User Query** → Enters prompt in Streamlit chat UI.  
2. **Retriever** → Queries vector store for top-*k* relevant passages.  
3. **LLM** → Groq LLaMA 3.3 70B processes query + retrieved context.  
4. **Response** → Sent back to Streamlit UI and added to memory buffer.  
5. **Conversation History** → Maintains context for multi-turn dialogue.  

---

## Project Structure

```text
ragbot_crime_and_punishment/
│
├── data/
│   └── crime_and_punishment.txt   # Novel text file
│
├── storage/
│   └── vector_index/              # Persistent vector index data
│
├── .streamlit/
│   └── secrets.toml               # (Optional) API keys for deployment
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .env                           # Local development secrets
├── README.md
```

## Installation & Usage

### Clone this Repository
```bash
git clone https://github.com/hsjoi1402/ragbot-crime-and-punishment.git
cd ragbot-crime-and-punishment
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure API Key
Set your Groq API key in .env:
```bash
GROQ_API_KEY=your_api_key_here
```

### Run the App
```bash
streamlit run app.py
```
The app will open in your browser at:
http://localhost:8xxx

---

## How It Works (Step-by-Step)
1. Load the novel text from /data/crime_and_punishment.txt.
2. Embed & Index: Create a vector index using HuggingFace embeddings.
3. Persist Index: Store it in /storage/vector_index for reuse.
4. Retrieve Context: On user queries, fetch top-k relevant passages.
5. Generate Answer: Send the context to Groq's LLaMA 3.3 model.
6. Display & Store: Show answer in chat UI and add to conversation history.

---

## Example Queries
1. What is Raskolnikov’s moral struggle?
2. Summarize the conversation between Raskolnikov and Sonia.
3. How does Dostoevsky portray guilt in the novel?

---

## Deployment
The app is Streamlit-ready and can be deployed:
1. Locally (via streamlit run)
2. On Streamlit Cloud with .streamlit/secrets.toml
3. In a Docker container for production

---

## Contribution Guidelines
Pull requests are welcome!
Future improvements:
1. Add multi-document support.
2. Enhance UI with richer formatting.
3. Integrate summarization features.
4. Retrieve links and sources with answers.

---

## Author
Prakash Joshi

---

## Acknowledgements
Fyodor Dostoevsky — For writing Crime and Punishment.
LlamaIndex — For simplifying RAG pipelines.
Groq — For making LLaMA 3 accessible.