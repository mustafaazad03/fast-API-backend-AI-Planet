from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import fitz # From PyMuPDF
import supabase
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    
    pdf_reader = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text()
    
    response = supabase_client.table("pdfs").insert({"filename": file.filename, "content": text}).execute()
    if response.error:
        raise HTTPException(status_code=500, detail="Failed to store PDF metadata.")
    
    return JSONResponse(content={"message": "PDF uploaded successfully.", "pdf_id": response.data[0]['id']})


@app.get("/")
def read_root():
    return {"Hello": "World"}