from fastapi import FastAPI,UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os,time
from pinecone import Pinecone,SchemaBuilder
from pydantic import BaseModel






load_dotenv()
app = FastAPI()
client  = genai.Client()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


schema = SchemaBuilder().add_dense_vector_field(name="embedding",dimension=1024,metric="cosine").build()

if not pc.indexes.exists(name="byo-vectors") : 
    pc.indexes.create(name="byo-vectors",schema=schema)

while not pc.indexes.describe(name="byo-vectors").status.ready:
    time.sleep(2)

index = pc.Index(name="byo-vectors")

@app.get("/")
def hello():
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life?"
)

    print(result.embeddings)
    return {"message" :"hello fastapi",
            "embedding" : result.embeddings}

@app.post("/uploadFile")
def uploadPdfFile(file : UploadFile) :

    pdf_bytes = file.file.read()

    pdf_document = fitz.open(stream=pdf_bytes ,filetype="pdf")

    chunks= []
    splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for page_number , page in enumerate(pdf_document,start=1):
        page_text = page.get_text()
        page_chunks = splitter.split_text(page_text)

        for chunk_index , chunk in enumerate(page_chunks):
            chunks.append({
                "text" : chunk,
                "page_number": page_number,
                "chunk_index": chunk_index
            })

    chunk_texts = []
    for chunk in chunks :
        chunk_texts.append(chunk["text"])


    embeddings_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk_texts,  
        config=genai.types.EmbedContentConfig(output_dimensionality=1024)
        # instead of passing a single string like in the hello route,
        # we pass the entire list of chunks
        # gemini will return one embedding vector per chunk
    )

    # with this
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_response.embeddings)):
       documents.append({
        "_id": f"{file.filename}-chunk-{i}",
        "embedding": embedding.values,
        "text": chunk["text"],
        "filename": file.filename,
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"]
    })

    index.documents.upsert(
    namespace="__default__",
    documents=documents
    )

    time.sleep(5)  # pinecone indexes documents asynchronously
               # meaning it doesn't happen instantly
               # if you query immediately after upsert, documents may not be there yet
               # so we wait 5 seconds

    return {
        "filename": file.filename,
        "total_pages": len(pdf_document),
        "total_chunks": len(chunks),
        "message": "PDF uploaded and embeddings stored successfully"
    }

# Chat endpoint

class ChatRequest(BaseModel):
    question:str
    filename:str


@app.post("/chat")
async def chat(req : ChatRequest):
    question_embedding = client.models.embed_content(model="gemini-embedding-2",
            contents=req.question,config=genai.types.EmbedContentConfig(output_dimensionality=1024))

    search_results = index.documents.search(     namespace="__default__",
    top_k=5,
    score_by=[{
        "type": "dense_vector",
        "field": "embedding",  # ← "field" not "fields", and it's a string not a list
        "values": question_embedding.embeddings[0].values
    }],
    include_fields=["*"],
    filter={"filename": {"$eq": req.filename}})

    context = ""

    for match in search_results.matches : 
        context += match.text + "\n\n"

    prompt = f"""
    You are a helpful assistant. Answer the question based only on the context provided below.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context}

    Question: {req.question}
    """


    def stream_response():
     for chunk in client.models.generate_content_stream(
        model="gemini-3.1-flash-lite",
        contents=prompt
     ):
        if chunk.text:
            print("CHUNK:", repr(chunk.text))
            yield chunk.text
            time.sleep(1)

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )