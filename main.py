from fastapi import FastAPI, HTTPException,UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os,time
from pinecone import Pinecone,SchemaBuilder
from pydantic import BaseModel
from supabase_client import supabase
from utils import count_tokens , trim_history




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
    conversation_id : str | None = None

ROLE_MAP = {"User": "user", "LLM": "model"}

SYSTEM_INSTRUCTION = """
Answer only from the provided context.
The earlier conversation is only there to help you understand what the user refers to.
If the answer is not in the context, say "I don't know based on the provided document."
"""

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. get or create the conversation
    if not req.conversation_id:
        res = supabase.table("conversation").insert({}).execute()
        conversation_id = res.data[0]["id"]
        history = []
    else:
        conversation_id = req.conversation_id
        res = (
            supabase.table("conversation")
            .select("id, created_at, message(id, created_at, role, message)")
            .eq("id", conversation_id)
            .order("created_at", desc=False, foreign_table="message")
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = res.data[0]["message"]

    # 2. trim history to the token budget
    recent_history = trim_history(history)

    # 3. rewrite the question if there is history
    search_query = req.question
    if recent_history:
        convo_text = "\n".join(f'{m["role"]}: {m["message"]}' for m in recent_history)
        rewrite_prompt = (
            "Given the conversation and the follow-up question, rewrite the "
            "follow-up as a standalone question. Return only the question.\n\n"
            f"Conversation:\n{convo_text}\n\nFollow-up: {req.question}"
        )
        rewrite = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=rewrite_prompt,
        )
        search_query = rewrite.text.strip()

    # 4. retrieve with the rewritten question
    question_embedding = client.models.embed_content(
        model="gemini-embedding-2",
        contents=search_query,
        config=types.EmbedContentConfig(output_dimensionality=1024),
    )
    search_results = index.documents.search(
        namespace="__default__",
        top_k=5,
        score_by=[{
            "type": "dense_vector",
            "field": "embedding",
            "values": question_embedding.embeddings[0].values,
        }],
        include_fields=["*"],
    )
    context = "\n\n".join(match.text for match in search_results.matches)

    # 5. build messages: history + (context + new question)
    llm_messages = [
        types.Content(role=ROLE_MAP[m["role"]], parts=[types.Part(text=m["message"])])
        for m in recent_history
    ]
    llm_messages.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"Context:\n{context}\n\nQuestion: {req.question}")],
        )
    )

    # 6. stream the answer, then save both messages
    def stream_response():
        full_answer = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-3.1-flash-lite",
            contents=llm_messages,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        ):
            if chunk.text:
                full_answer += chunk.text
                yield chunk.text

        supabase.table("message").insert([
            {"conversation_id": conversation_id, "role": "User", "message": req.question},
            {"conversation_id": conversation_id, "role": "LLM", "message": full_answer},
        ]).execute()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
        headers={"X-Conversation-Id": str(conversation_id)},
    )

# @app.get("/test-db")
# async def test():
#     conversation_id = None
#     # if not conversation_id :
#     #      conversation_id =  supabase.table("conversation").insert({}).select("id").execute()
#     resposne =   supabase.table("conversation").select("id, created_at, message(id, created_at, role, message)").eq("id", 1).order("created_at", desc=False, foreign_table="message").execute()
#     print(resposne)
#     return {"data" : resposne.data,}
    