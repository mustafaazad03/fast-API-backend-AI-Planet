from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import supabase
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.models.request_models import QuestionRequest
from app.core.config import SUPABASE_URL, SUPABASE_KEY
from app.utils.text_conversion import convert_to_html
from fastapi_utils.tasks import repeat_every

# Initialize API router
router = APIRouter()

# Pre-load models and vectorstore embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
retriever = vectorstore.as_retriever()

# Cache for chains
chains_cache = {}

# Function to convert history from database format to LangChain message objects
def form_history_obj(history):
    new_history = []
    for entry in history:
        if "question" in entry:
            human_message = HumanMessage(content=entry["question"])
            new_history.append(human_message)
        if "response" in entry:
            ai_message = AIMessage(content=entry["response"])
            new_history.append(ai_message)
    return new_history

# Function to create the QA chain, uses caching to avoid reloading models
async def qa_chain(callback=StreamingStdOutCallbackHandler()):
    if 'qa_chain' not in chains_cache:
        # System prompt for contextualizing questions
        contextualize_q_system_prompt = """
        Given a chat history and the latest user question which might reference context in the chat history or the provided PDF content,
        formulate a standalone question which can be understood without the chat history or the full PDF content.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # System prompt for the QA task
        qa_system_prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        {context}
        Always provide your answers in markdown format. As this is a PDF-based question-answering system, the answers should be based on the content of the PDF.
        If the answer is not in the PDF, you can say that the answer is not in the PDF. Also, you can use the chat history to answer the question in a better way.
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Initialize the streaming LLM with the defined prompt and callbacks
        streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            streaming=True,
            callbacks=[callback],
            verbose=True,
        )
        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(streaming_llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Cache the chain
        chains_cache['qa_chain'] = rag_chain
        
    return chains_cache['qa_chain']

@router.post("/")
async def ask_question(request: QuestionRequest):
    """
    Process a question related to a specific PDF document, retrieve PDF content and interaction history from a Supabase database,
    generate an answer using a question-answering chain, update the interaction history, and return the answer in HTML format.

    Args:
        request (QuestionRequest): An instance of QuestionRequest containing pdf_id and question.

    Returns:
        JSONResponse: A JSON response containing the answer in HTML format.
    """
    try:
        callback = StreamingStdOutCallbackHandler()
        # Retrieve the QA chain, loading models if necessary
        chain = await qa_chain(callback)
        
        # Connect to Supabase and retrieve PDF content and history
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        pdf_data = supabase_client.table("pdfs").select("content", "history").eq("id", request.pdf_id).single().execute()
        
        if pdf_data.data is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        
        context = pdf_data.data['content']
        history = json.loads(pdf_data.data['history'])
        history_obj = form_history_obj(history)
        
        # Prepare input data for the QA chain
        input_data = {"input": request.question, "chat_history": history_obj}
        
        # Invoke the QA chain to get the response
        response = chain.invoke(input_data)
        answer = convert_to_html(response["answer"])
        
        # Update the interaction history with the new question and answer
        new_entry = {"question": request.question, "response": answer}
        history.append(new_entry)
        
        # Save the updated history back to the database
        supabase_client.table("pdfs").update({"history": json.dumps(history)}).eq("id", request.pdf_id).execute()
        
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI()

# Include the router
app.include_router(router, prefix="/api")

# Load models and chains on startup
@app.on_event("startup")
@repeat_every(seconds=3600) 
def load_models_and_chains():
    # Pre-load and cache the QA chain during startup
    qa_chain()
