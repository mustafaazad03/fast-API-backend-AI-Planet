from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import SUPABASE_URL, SUPABASE_KEY
import supabase
import os

router = APIRouter()

@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    try:
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        pdf_data = supabase_client.table("pdfs").select("filename").eq("id", pdf_id).single().execute()
        if pdf_data.data is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        
        filename = pdf_data.data['filename']
        file_path = os.path.join("uploads", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        supabase_client.table("pdfs").delete().eq("id", pdf_id).execute()
        return JSONResponse(content={"message": "PDF deleted successfully."})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
