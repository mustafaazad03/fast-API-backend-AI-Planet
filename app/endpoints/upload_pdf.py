from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import SUPABASE_URL, SUPABASE_KEY
import supabase
import json
import fitz
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()

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

@router.post("/")
async def upload_pdf(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    # Save the uploaded PDF file in the project directory with a folder named "uploads"
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    if os.path.exists(f"uploads/{file.filename}"):
        os.remove(f"uploads/{file.filename}")
    
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(file.file.read())

    # Read the contents of the uploaded PDF file
    with open(f"uploads/{file.filename}", "rb") as f:
        contents = f.read()

    try:
        text = extract_text_from_pdf(contents=contents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no text.")
        
        text_chunks = split_text_into_chunks(text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Failed to split the text into valid chunks.")
        
            vectorstore = get_vector_store(text_chunks)
        
        response = supabase_client.table("pdfs").insert({"filename": file.filename, "content": text, "history": json.dumps([])}).execute()
        if response.data is None:
            raise HTTPException(status_code=500, detail="Failed to insert PDF into database.")

        pdf_id = response.data[0]['id']
        filename = file.filename
        return JSONResponse(content={"pdf_id": pdf_id, "filename": filename})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))