from app.core.config import SUPABASE_URL, SUPABASE_KEY
import supabase

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_file_to_storage(file_path, file_contents):
    response = supabase_client.storage.from_('pdfs').upload(file_path, file_contents)
    if response.get('error'):
        raise ValueError("Failed to upload file to storage.")
    return f"{SUPABASE_URL}/storage/v1/object/public/{file_path}"

def download_file_from_storage(file_path):
    response = supabase_client.storage.from_('pdfs').download(file_path)
    if response.get('error'):
        raise ValueError("Failed to download file from storage.")
    return response.get('data')
