from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import supabase
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.models.request_models import QuestionRequest
from app.core.config import SUPABASE_URL, SUPABASE_KEY
from app.utils.text_conversion import convert_to_html

router = APIRouter()

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


@router.post("/")
async def ask_question(request: QuestionRequest):
    try:
        callback = StreamingStdOutCallbackHandler()
        chain = await qa_chain(callback)
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
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