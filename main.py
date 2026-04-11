import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from supabase import create_client, Client

# Load API
load_dotenv()

# Initialize Supabase memory bank
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Start FastAPI server
app = FastAPI()

print("Waking up optimus...")

# 1 connect to brian (ChromaDB)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(embedding_function=embeddings,
                   persist_directory="./chroma_db")
# 2. Set up the conversational AI model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# This defines what an incoming message looks like


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"

# 3. Create the endpoint that receives questions

# Endpoint 1: srve the website


@app.get("/")
async def get_ui():
    return FileResponse("index.html")

# Endpoint 2: answer the chat


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"User asked: {request.message}")

    # A. Fetch previous chat history from Supabase
    history = supabase.table("chat_history").select(
        "*").eq("session_id", request.session_id).order("created_at", desc=False).limit(6).execute()

    formatted_history = ""
    if history.data:
        for row in history.data:
            formatted_history += f"{row['role'].capitalize()}: {row['content']}\n"

    # B. Search the database for the 3 most relevant paragraphs from your PDF
    docs = vector_db.similarity_search(request.message, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # C. Build a strict prompt for the AI
    prompt = f"""You are a helpful customer support bot for a life insurance company. 
    Use ONLY the following FAQ information to answer the user's question. 
    If the answer is not in the information provided, politely say that you do not have that information.
    
    Recent Chat History:
    {formatted_history}

    FAQ Information:
    {context}
    
    User Question: {request.message}
    """

    # D. Get the answer from Gemini
    response = llm.invoke(prompt)
    bot_reply = response.content

    # E. Save both the user's question and bot's answer to Supabase
    supabase.table("chat_history").insert({
        "session_id": request.session_id,
        "role": "user",
        "content": request.message
    }).execute()

    supabase.table("chat_history").insert({
        "session_id": request.session_id,
        "role": "optimus",
        "content": bot_reply
    }).execute()

    return {"reply": response.content}

print("Bot is ready to chat!")
