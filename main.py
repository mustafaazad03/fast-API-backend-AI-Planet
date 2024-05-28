from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz # From PyMuPDF
import supabase
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
import json

app = FastAPI()

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

class QuestionRequest(BaseModel):
    pdf_id: str
    question: str

def extract_text_from_pdf(file: UploadFile):
    pdf_reader = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    
    text = extract_text_from_pdf(file)
    text_chunks = split_text_into_chunks(text)
    vectorstore = get_vector_store(text_chunks)
    
    response = supabase_client.table("pdfs").insert({"filename": file.filename, "content": text, "history": json.dumps([])}).execute()
    if response.error:
        raise HTTPException(status_code=500, detail="Failed to store PDF metadata.")

    return JSONResponse(content={"message": "PDF uploaded successfully.", "pdf_id": response.data[0]['id']})

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    pdf_data = supabase_client.table("pdfs").select("content", "history").eq("id", request.pdf_id).single().execute()
    if pdf_data.error:
        raise HTTPException(status_code=404, detail="PDF not found.")
    
    context = pdf_data.data['content']
    history = json.loads(pdf_data.data['history'])

    # Check for follow-up questions
    if len(history) >= 2:
        prev_questions = [entry['question'] for entry in history[-2:]]
        prev_responses = [entry['response'] for entry in history[-2:]]
        follow_up_keywords = ['above', 'previous', 'before', 'mentioned', 'following up', 'related to']
        if any(keyword in request.question.lower() for keyword in follow_up_keywords):
            context += "\n".join(prev_questions + prev_responses)

    model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    response = model.start_chat(history=[]).send_message(
        f"Context: {context}\nQuestion: {request.question}\n"
    )

    answer = response.text

    # Update chat history
    new_entry = {"question": request.question, "response": answer}
    history.append(new_entry)
    supabase_client.table("pdfs").update({"history": json.dumps(history)}).eq("id", request.pdf_id).execute()

    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)