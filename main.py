from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  # From PyMuPDF
import supabase
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import List
import json
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

class QuestionRequest(BaseModel):
    pdf_id: str
    question: str

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
    # Filter out any empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks to embed.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

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

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    pdf_data = supabase_client.table("pdfs").select("content", "history").eq("id", request.pdf_id).single().execute()
    if pdf_data.error:
        raise HTTPException(status_code=404, detail="PDF not found.")
    
    context = pdf_data.data['content']
    history = json.loads(pdf_data.data['history'])
    if len(history) >= 2:
        prev_context = "\n".join([f"Q: {entry['question']} A: {entry['response']}" for entry in history[-2:]])
        context = f"{prev_context}\n{context}"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    docs = vectorstore.similarity_search(request.question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": request.question}, return_only_outputs=True)
    answer = response["output_text"]

    new_entry = {"question": request.question, "response": answer}
    history.append(new_entry)
    supabase_client.table("pdfs").update({"history": json.dumps(history)}).eq("id", request.pdf_id).execute()

    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
