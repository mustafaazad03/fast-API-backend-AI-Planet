from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # From PyMuPDF
import supabase
import os
from dotenv import load_dotenv
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_text_from_pdf(contents):
    pdf_document = fitz.open(stream=contents, filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks to embed.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    contents = await file.read()
    try:
        text = extract_text_from_pdf(contents=contents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no text.")
        
        text_chunks = split_text_into_chunks(text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Failed to split the text into valid chunks.")
        
        try:
            vectorstore = get_vector_store(text_chunks)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        response = supabase_client.table("pdfs").insert({"filename": file.filename, "content": text, "history": json.dumps([])}).execute()
        if response.data is None:
            raise HTTPException(status_code=500, detail="Failed to insert PDF into database.")

        pdf_id = response.data[0]['id']
        filename = file.filename
        return JSONResponse(content={"pdf_id": pdf_id, "filename": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
