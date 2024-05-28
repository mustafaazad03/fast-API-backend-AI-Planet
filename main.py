from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz # From PyMuPDF

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")
    
    pdf_reader = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text()
    return {"filename": file.filename, "content": text}

@app.get("/")
def read_root():
    return {"Hello": "World"}