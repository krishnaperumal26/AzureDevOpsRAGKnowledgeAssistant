from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import requests
import faiss
import numpy as np
import openai
import pickle
from typing import List
from docx import Document as DocxDocument
import PyPDF2

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


user_settings = {"openai_key": "", "azure_pat": "", "azure_org": "", "azure_project": ""}

VECTOR_DB_PATH = "vector_db/faiss.index"
VECTOR_META_PATH = "vector_db/meta.pkl"
os.makedirs("vector_db", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

def get_openai_embedding(text: str, api_key: str) -> np.ndarray:
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def save_faiss_index(index, meta):
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(VECTOR_META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_faiss_index():
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(VECTOR_META_PATH):
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(VECTOR_META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    else:
        return None, []

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def parse_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def get_file_text(file_path):
    if file_path.endswith(".txt"):
        return parse_txt(file_path)
    elif file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".docx"):
        return parse_docx(file_path)
    else:
        return ""

def fetch_azure_wiki_and_workitems(pat, org, project):
    # Fetch wiki pages
    wiki_url = f"https://dev.azure.com/{org}/{project}/_apis/wiki/wikis?api-version=7.0"
    headers = {"Authorization": f"Basic {pat}"}
    wikis = requests.get(wiki_url, headers=headers).json()
    wiki_content = []
    for wiki in wikis.get("value", []):
        wiki_id = wiki["id"]
        pages_url = f"https://dev.azure.com/{org}/{project}/_apis/wiki/wikis/{wiki_id}/pages?api-version=7.0"
        pages = requests.get(pages_url, headers=headers).json()
        for page in pages.get("value", []):
            page_id = page["id"]
            page_url = f"https://dev.azure.com/{org}/{project}/_apis/wiki/wikis/{wiki_id}/pages/{page_id}?api-version=7.0&includeContent=true"
            page_data = requests.get(page_url, headers=headers).json()
            wiki_content.append(page_data.get("content", ""))
    # Fetch PBIs and Features
    wiql_url = f"https://dev.azure.com/{org}/{project}/_apis/wit/wiql?api-version=7.0"
    wiql_query = {
        "query": "SELECT [System.Id], [System.Title], [System.WorkItemType], [System.Description] FROM WorkItems WHERE [System.WorkItemType] IN ('Product Backlog Item', 'Feature')"
    }
    workitems = requests.post(wiql_url, headers={**headers, "Content-Type": "application/json"}, json=wiql_query).json()
    ids = [str(wi["id"]) for wi in workitems.get("workItems", [])]
    items_content = []
    if ids:
        ids_str = ",".join(ids)
        details_url = f"https://dev.azure.com/{org}/{project}/_apis/wit/workitems?ids={ids_str}&api-version=7.0"
        details = requests.get(details_url, headers=headers).json()
        for item in details.get("value", []):
            title = item["fields"].get("System.Title", "")
            desc = item["fields"].get("System.Description", "")
            items_content.append(f"{title}\n{desc}")
    return wiki_content + items_content

def embed_and_store_texts(texts: List[str], api_key: str):
    vectors = [get_openai_embedding(t, api_key) for t in texts]
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    save_faiss_index(index, texts)

def search_rag(query: str, api_key: str):
    index, meta = load_faiss_index()
    if not index or not meta:
        return "Vector DB is empty. Please update it first."
    q_vec = get_openai_embedding(query, api_key)
    D, I = index.search(np.array([q_vec]), k=3)
    context = "\n---\n".join([meta[i] for i in I[0] if i < len(meta)])
    prompt = f"Answer the question using the following context from Azure DevOps Wiki and PBIs.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    openai.api_key = api_key
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=256
    )
    return response.choices[0].text.strip()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, **user_settings})

@app.post("/set_keys")
def set_keys(openai_key: str = Form(...), azure_pat: str = Form(...), azure_org: str = Form(...), azure_project: str = Form(...)):
    user_settings["openai_key"] = openai_key
    user_settings["azure_pat"] = azure_pat
    user_settings["azure_org"] = azure_org
    user_settings["azure_project"] = azure_project
    return RedirectResponse("/", status_code=303)

@app.post("/upload_doc")
def upload_doc(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return RedirectResponse("/", status_code=303)

@app.post("/update_vector_db")
def update_vector_db():
    openai_key = user_settings["openai_key"]
    azure_pat = user_settings["azure_pat"]
    # Get org and project from user_settings (set via UI)
    org = user_settings.get("azure_org", "")
    project = user_settings.get("azure_project", "")
    # Fetch Azure DevOps content
    azure_texts = fetch_azure_wiki_and_workitems(azure_pat, org, project) if org and project and azure_pat else []
    # Fetch uploaded docs
    doc_texts = []
    for fname in os.listdir("uploads"):
        doc_texts.append(get_file_text(f"uploads/{fname}"))
    all_texts = azure_texts + doc_texts
    if all_texts and openai_key:
        embed_and_store_texts(all_texts, openai_key)
    return RedirectResponse("/", status_code=303)

@app.post("/ask")
def ask(request: Request, question: str = Form(...)):
    openai_key = user_settings["openai_key"]
    answer = search_rag(question, openai_key) if openai_key else "OpenAI key not set."
    return templates.TemplateResponse("index.html", {"request": request, **user_settings, "answer": answer})
