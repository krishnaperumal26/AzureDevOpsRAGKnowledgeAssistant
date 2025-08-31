# Azure DevOps RAG Knowledge Tool

This is a knowledge management and guidance tool for Scrum teams, built with FastAPI and a modern HTML/CSS/JS frontend. It uses Retrieval-Augmented Generation (RAG) with OpenAI and integrates with Azure DevOps to provide context-aware answers during refinement sessions.

## Features
- **OpenAI & Azure DevOps Integration:** Enter your OpenAI API key and Azure DevOps Personal Access Token (PAT) in the UI.
- **Custom Document Upload:** Upload multiple requirement documents (PDF, DOCX, TXT) for use as RAG context.
- **Azure DevOps Wiki & PBIs:** Fetches and uses Wiki pages, Product Backlog Items, and Features from your Azure DevOps project.
- **Vector Database:** All content (documents + Azure DevOps) is embedded and stored in a vector DB (FAISS) for fast retrieval.
- **Interactive Chat:** Ask questions and get context-aware answers from the bot, using all available knowledge sources.
- **Modern, Colorful UI:** User-friendly, interactive, and visually appealing interface.

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```
3. **Open the app:**
   Go to [http://127.0.0.1:8000] in your browser.

## Usage
1. **Enter your OpenAI API key and Azure DevOps PAT, Organization, and Project.**
2. **Upload one or more requirement documents** (PDF, DOCX, TXT). (Optional)
3. **Click 'Update Vector DB'** to embed all available documents and/or Azure DevOps content.
4. **Ask questions in the chat** and get answers based on your project knowledge.

- You can use only documents, only Azure DevOps, or both—neither is mandatory.
- All settings are saved in your browser for convenience.

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python dependencies.

## Notes
- Your API keys and tokens are stored in memory only for the session and are not persisted.
- Uploaded documents are stored in the `uploads/` folder.
- Vector DB is stored in the `vector_db/` folder.

## License
MIT License
