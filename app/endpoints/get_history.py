from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import SUPABASE_URL, SUPABASE_KEY
import supabase
import json

router = APIRouter()

@router.get("/{pdf_id}")
async def get_history(pdf_id: str):
    try:
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        pdf_data = supabase_client.table("pdfs").select("history").eq("id", pdf_id).single().execute()
        if pdf_data.data is None:
            raise HTTPException(status_code=404, detail="PDF not found.")
        
        if pdf_data.data['history'] is None:
            raise HTTPException(status_code=404, detail="No history found for the PDF.")
        
        history = json.loads(pdf_data.data['history'])
        return JSONResponse(content={"history": history})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
