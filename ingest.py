import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load your secret API key from the .env file
load_dotenv()

# 2. Tell the script which PDF to read (using your exact file name!)
pdf_filename = "life_insurance_faq.pdf"
print(f"Loading {pdf_filename}...")

# 3. Read the PDF
loader = PyPDFLoader(pdf_filename)
documents = loader.load()

# 4. Chop the text into smaller, digestible chunks
print("Chopping text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 5. Convert text to "embeddings" and save to the Chroma database
print("Saving to database. This might take a moment...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# This creates a folder called 'chroma_db' to hold the data
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Success! Your PDF knowledge is now saved in the database.")
