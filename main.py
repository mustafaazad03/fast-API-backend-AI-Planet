from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
import re

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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "aiplanet"

genai.configure(api_key=GOOGLE_API_KEY)

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

class QuestionRequest(BaseModel):
    pdf_id: str
    question: str

class HistoryResponse(BaseModel):
    history: List[dict]


def convert_to_html(text):
    # Convert headers
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert newlines and list items
    text = re.sub(r'\n', r'<br>', text)
    text = re.sub(r'\* (.*?)<br>', r'<li>\1</li>', text)
    
    # Handle nested lists
    text = re.sub(r'\*\* (\w)\. (.*?)<br>', r'<strong>\1.</strong> \2<br>', text)
    text = re.sub(r'\*\* (\d)\. (.*?)<br>', r'<strong>\1.</strong> \2<br>', text)
    text = re.sub(r'\*\* (\w\.) (.*?)<br>', r'<strong>\1</strong> \2<br>', text)
    
    # Handle <ul> and <ol> tags for list items
    text = re.sub(r'(<br>)(<li>)', r'\1<ul>\2', text)
    text = re.sub(r'(<li>.*?</li>)(<br>)(?=<li>)', r'\1\2', text)
    text = re.sub(r'(<li>.*?</li>)(<br>)(?!<li>)', r'\1</ul>\2', text)
    
    # Add <ol> for ordered lists
    text = re.sub(r'<strong>(\d\.)</strong> (.*?)<br>', r'<ol><li>\2</li></ol>', text)
    
    # Convert tables
    text = re.sub(r'\n\| (.*?) \| (.*?) \|\n', r'<tr><th>\1</th><th>\2</th></tr>', text)
    text = re.sub(r'\| (.*?) \| (.*?) \|\n', r'<tr><td>\1</td><td>\2</td></tr>', text)
    
    # Add <table> tags
    text = re.sub(r'<tr><th>', r'<table><thead><tr><th>', text)
    text = re.sub(r'</th><th>', r'</th><th>', text)
    text = re.sub(r'</th></tr><tr><td>', r'</th></tr></thead><tbody><tr><td>', text)
    text = re.sub(r'</td><td>', r'</td><td>', text)
    text = re.sub(r'</td></tr>', r'</td></tr>', text)
    text = re.sub(r'</tr>', r'</tr></tbody></table>', text)

    return text


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

def form_history_obj(history):
    new_history = []
    for entry in history:
        if "question" in entry:
            ai_message = AIMessage(content=entry["question"])
            new_history.append(ai_message)
        if "response" in entry:
            human_message = HumanMessage(content=entry["response"])
            new_history.append(human_message)
    return new_history

async def qa_chain(callback=StreamingStdOutCallbackHandler()):
    contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

    {context}
    
    Always provide your answers in markdown format.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    streaming_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        streaming=True,
        callbacks=[callback],
        verbose=True,
    )
    question_answer_chain = create_stuff_documents_chain(streaming_llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

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
    try:
        callback = StreamingStdOutCallbackHandler()
        chain = await qa_chain(callback)
        pdf_data = supabase_client.table("pdfs").select("content", "history").eq("id", request.pdf_id).single().execute()
        if pdf_data.data is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        context = pdf_data.data['content']
        history = json.loads(pdf_data.data['history'])
        history_obj = form_history_obj(history)
        input_data = {"input": request.question, "chat_history": history_obj}
        
        response = chain.invoke(input_data) 
        answer = convert_to_html(response["answer"])
        new_entry = {"question": request.question, "response": answer}
        history.append(new_entry)
        supabase_client.table("pdfs").update({"history": json.dumps(history)}).eq("id", request.pdf_id).execute()
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
