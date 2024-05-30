from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload_pdf, ask_question, delete_pdf, get_history
from dotenv import load_dotenv


app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_pdf.router, prefix="/upload-pdf", tags=["upload-pdf"])
app.include_router(ask_question.router, prefix="/ask-question", tags=["ask-question"])
app.include_router(delete_pdf.router, prefix="/delete-pdf", tags=["delete-pdf"])
app.include_router(get_history.router, prefix="/get-history", tags=["get-history"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
