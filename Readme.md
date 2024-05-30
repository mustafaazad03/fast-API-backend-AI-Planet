# PDF Chat Application - Assignment for AI Planet

This project provides a FastAPI-based backend for a PDF chat application that allows users to upload PDFs, ask questions about the content, retrieve chat history, and delete PDFs. The backend leverages various services, including Supabase for storage and Langchain for natural language processing.

# Tech Stack

- FastAPI
- Supabase
- Langchain
- Pinecone
- Python
- Gemini API

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running the Application](#running-the-application)
4. [API Endpoints](#api-endpoints)
   - [Upload PDF](#upload-pdf)
   - [Ask Question](#ask-question)
   - [Delete PDF](#delete-pdf)
   - [Get History](#get-history)
5. [File Structure](#file-structure)

## Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/mustafaazad03/fast-API-backend-AI-Planet.git
   cd fast-API-backend-AI-Planet
   ```

2. **Create a virtual environment and activate it**:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up Supabase**:
   - Create a project on Supabase.
   - Note the `SUPABASE_URL` and `SUPABASE_KEY` for your project.
   - Create a table named `pdfs` with columns for `id`, `filename`, `content`, and `history`.

## Configuration

Create a `.env` file in the root directory and add the following environment variables:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=aiplanet
```

## Running the Application

Start the FastAPI server by running:

```sh
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## API Endpoints

### Upload PDF

- **Endpoint**: `/upload-pdf/`
- **Method**: `POST`
- **Description**: Uploads a PDF file, extracts its text, splits the text into chunks, and stores the content and metadata in Supabase.
- **Request Body**:

  - `file`: PDF file (Content-Type: application/pdf)

- **Response**:
  ```json
  {
  	"pdf_id": "string",
  	"filename": "string"
  }
  ```

### Ask Question

- **Endpoint**: `/ask-question/`
- **Method**: `POST`
- **Description**: Asks a question about the content of a specific PDF and retrieves an answer based on the content and chat history.
- **Request Body**:

  ```json
  {
  	"pdf_id": "string",
  	"question": "string"
  }
  ```

- **Response**:
  ```json
  {
  	"answer": "string"
  }
  ```

### Delete PDF

- **Endpoint**: `/delete-pdf/{pdf_id}`
- **Method**: `DELETE`
- **Description**: Deletes a specific PDF and its metadata from Supabase.
- **Response**:
  ```json
  {
  	"message": "PDF deleted successfully."
  }
  ```

### Get History

- **Endpoint**: `/get-history/{pdf_id}`
- **Method**: `GET`
- **Description**: Retrieves the chat history for a specific PDF.
- **Response**:
  ```json
  {
      "history": [
          {
              "question": "string",
              "response": "string"
          },
          ...
      ]
  }
  ```

## File Structure

```
pdf-chat-app/
├── app/
│   ├── core/
│   │   └── config.py
│   ├── endpoints/
│   │   ├── upload_pdf.py
│   │   ├── ask_question.py
│   │   ├── delete_pdf.py
│   │   └── get_history.py
│   ├── models/
│   │   └── request_models.py
│   ├── utils/
│   │   └── text_conversion.py
├── main.py
├── requirements.txt
├── .env
└── README.md
```

## Usability Functionality

- **Text Extraction**: The application extracts text from uploaded PDF files using the `PyMuPDF` library.
- **Text Chunking**: The extracted text is split into chunks of 500 characters to facilitate natural language processing.
- **Natural Language Processing**: The application uses the `Langchain` API to generate responses to user questions based on the content of the PDF and chat history.
- **Chat History**: The application stores the chat history for each PDF in Supabase and retrieves it when requested.
- **PDF Deletion**: Users can delete PDF files and their metadata from the database.
- **Text To HTML**: The extracted text is converted to HTML format for better readability.
- **QA Chain**: The application uses the `Pinecone` API to store and retrieve question-answer pairs for each PDF.

## Future Improvements

- **User Authentication**: Implement user authentication to secure the API endpoints.
- **Testing**: Write unit tests for the API endpoints and utility functions.
- **Pagination**: Implement pagination for chat history to handle large datasets.
- **Multiple File Upload**: Allow users to upload multiple PDF files at once.
- **Real-Time Chat**: Implement real-time chat functionality using WebSockets.
- **Dockerization**: Dockerize the application for easier deployment and scaling.

## Contributors

- [Mustafa Azad](https://github.com/mustafaazad03)
