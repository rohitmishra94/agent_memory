from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import logging
import asyncio
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and clients
model = None
client = None
user_collections = {}  # Cache for user collections

# Pydantic models for API
class Memory(BaseModel):
    memory: str = Field(..., description="First-person memory statement")
    category: Literal["Identity", "Technical", "Style", "Goals", "Challenges", "Collaboration", "Patterns", "Workflow"]
    context: str = Field(..., description="Why this memory is worth remembering")
    emotional_tone: Optional[Literal["confident", "excited", "frustrated", "curious", "concerned", "satisfied", "conflicted", "neutral", "determined", "overwhelmed"]] = None
    tags: List[str] = Field(..., min_items=2, max_items=5)

class WorkflowMemory(BaseModel):
    memory: str = Field(..., description="First-person workflow description")
    category: Literal["Workflow"] = "Workflow"
    context: str = Field(..., description="What problem this workflow solved")
    process_pattern: str = Field(..., description="Workflow pattern like 'web_search → analysis → visualization'")
    tools_used: List[str] = Field(..., description="Tools used in this workflow")
    emotional_tone: Optional[Literal["confident", "excited", "frustrated", "curious", "concerned", "satisfied", "conflicted", "neutral", "determined", "overwhelmed"]] = None
    tags: List[str] = Field(..., min_items=2, max_items=5)

class MemoryAgentOutput(BaseModel):
    session_assessment: Literal["processed", "discarded"]
    discard_reason: Optional[str] = None
    memories: List[Memory] = []
    workflow_memories: List[WorkflowMemory] = []
    session_summary: Optional[str] = None

class StoreMemoriesRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    memory_output: MemoryAgentOutput = Field(..., description="Output from memory agent")

class QueryMemoriesRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Search query")
    n_results: int = Field(10, description="Number of initial results", ge=1, le=50)
    top: int = Field(3, description="Number of top contexts to return", ge=1, le=10)
    category_filter: Optional[str] = None
    memory_type_filter: Optional[Literal["regular", "workflow"]] = None
    recent_days: Optional[int] = Field(None, description="Only memories from last N days", ge=1, le=365)

class RecentContextRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    days: int = Field(7, description="Number of days to look back", ge=1, le=90)
    limit: int = Field(10, description="Maximum memories to return", ge=1, le=50)

# Embedding function for ChromaDB
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        global model
        return model.encode(input, return_dense=True)['dense_vecs']

def get_unique_text_indices(text_list):
    """Get indices of unique texts."""
    unique_texts = {}
    unique_indices = []
    for i, text in enumerate(text_list):
        if text not in unique_texts:
            unique_texts[text] = i
            unique_indices.append(i)
    return unique_indices

async def get_full_context(query, collection, n_results=5, top=2):
    """Your custom context retrieval function - now async."""
    logger.info(f'Querying collection for context retrieval')
    
    result = await collection.query(query_texts=query, n_results=n_results)
    texts = result['documents'][0]
    ids = result['ids'][0]
    unique_indices = get_unique_text_indices(texts)
    unique_docs = [texts[x] for x in unique_indices]
    unique_ids = [ids[x] for x in unique_indices]
    
    # ColBERT scoring
    query_col = model.encode([query[0]], return_colbert_vecs=True)
    docs_col = model.encode(unique_docs, return_colbert_vecs=True)
    colbert_scores = []
    for vectors in docs_col['colbert_vecs']:
        colbert_scores.append(model.colbert_score(query_col['colbert_vecs'][0], vectors).numpy())

    # Full context ColBERT
    full_context_scores = []
    full_context_ids = []
    for id in unique_ids:
        try:
            pre_id, post_id = str(int(id)-1), str(int(id)+1)
            full_context_ids.append([pre_id, id, post_id])
            full_context = (await collection.get(ids=[pre_id, id, post_id]))['documents']
            full_context = ''.join(full_context)
            full_context_colbert_vec = model.encode([full_context], return_colbert_vecs=True)
            full_context_colbert_score = model.colbert_score(
                query_col['colbert_vecs'][0], 
                full_context_colbert_vec['colbert_vecs'][0]
            ).numpy()
            full_context_scores.append(full_context_colbert_score)
        except Exception as e:
            logger.warning(f"Error processing context for id {id}: {e}")
            full_context_scores.append(0.0)
            full_context_ids.append([id, id, id])

    all_scores = [2*full_context_scores[i] + 0.9*colbert_scores[i] for i in range(len(colbert_scores))]
    sorted_indices = [index for index, _ in sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)]
    top_context_ids_list = [full_context_ids[index] for index in sorted_indices][:top]
    flattened_list = np.array(top_context_ids_list).flatten().tolist()
    top_ids = list(set(flattened_list))
    top_context = (await collection.get(ids=top_ids))['documents']

    logger.info(f'Context retrieved successfully')
    return top_context, top_ids

# Import your VectorMemoryStorage class (assuming it's in the same file or imported)
from vector_db_storage import VectorMemoryStorage

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, client
    logger.info("Starting up Memory Storage Server...")
    
    try:
        # Initialize BGE-M3 model
        logger.info("Loading BGE-M3 model...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info("Model loaded successfully")
        
        # Initialize ChromaDB client
        logger.info("Connecting to ChromaDB...")
        client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
        heartbeat = await client.heartbeat()
        logger.info(f"ChromaDB connected: {heartbeat}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Memory Storage Server...")

# Create FastAPI app
app = FastAPI(
    title="Memory Storage API",
    description="API for storing and retrieving user memories with semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_user_collection(user_id: str):
    """Get or create collection for user."""
    global user_collections, client
    
    if user_id not in user_collections:
        collection = await client.get_or_create_collection(
            name=user_id,
            embedding_function=MyEmbeddingFunction()
        )
        storage = VectorMemoryStorage(collection, user_id)
        await storage.initialize_counter()
        user_collections[user_id] = storage
        logger.info(f"Created new collection for user {user_id}")
    
    return user_collections[user_id]

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        heartbeat = await client.heartbeat()
        return {
            "status": "healthy",
            "chromadb": "connected" if heartbeat else "disconnected",
            "model": "loaded" if model else "not loaded",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/memories/store")
async def store_memories(request: StoreMemoriesRequest):
    """Store memories from memory agent output."""
    try:
        storage = await get_user_collection(request.user_id)
        stored_ids = await storage.store_memories(
            request.memory_output, 
            request.session_id
        )
        
        return {
            "success": True,
            "stored_ids": stored_ids,
            "count": len(stored_ids),
            "session_id": request.session_id,
            "assessment": request.memory_output.session_assessment
        }
        
    except Exception as e:
        logger.error(f"Error storing memories for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/query")
async def query_memories(request: QueryMemoriesRequest):
    """Query memories using semantic search with ColBERT scoring."""
    try:
        storage = await get_user_collection(request.user_id)
        
        results = await storage.get_relevant_memories(
            query=request.query,
            get_full_context_func=get_full_context,
            model=model,
            n_results=request.n_results,
            top=request.top,
            category_filter=request.category_filter,
            memory_type_filter=request.memory_type_filter,
            recent_days=request.recent_days
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "user_id": request.user_id
        }
        
    except Exception as e:
        logger.error(f"Error querying memories for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/recent")
async def get_recent_memories(request: RecentContextRequest):
    """Get recent memories for conversation context."""
    try:
        storage = await get_user_collection(request.user_id)
        recent_memories = await storage.get_recent_context(
            days=request.days,
            limit=request.limit
        )
        
        return {
            "success": True,
            "memories": recent_memories,
            "count": len(recent_memories),
            "days": request.days,
            "user_id": request.user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting recent memories for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/stats/{user_id}")
async def get_memory_stats(user_id: str):
    """Get statistics about user's stored memories."""
    try:
        storage = await get_user_collection(user_id)
        stats = await storage.get_memory_stats()
        
        return {
            "success": True,
            "stats": stats,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting stats for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/user/{user_id}")
async def delete_user_memories(user_id: str):
    """Delete all memories for a user."""
    try:
        if user_id in user_collections:
            # Delete the collection
            await client.delete_collection(name=user_id)
            del user_collections[user_id]
            logger.info(f"Deleted all memories for user {user_id}")
        
        return {
            "success": True,
            "message": f"All memories deleted for user {user_id}",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting memories for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all user collections."""
    try:
        collections = await client.list_collections()
        return {
            "success": True,
            "collections": [col.name for col in collections],
            "count": len(collections)
        }
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for cleanup (optional)
@app.post("/admin/cleanup")
async def cleanup_old_memories(
    days_old: int = 365,
    dry_run: bool = True
):
    """Admin endpoint to cleanup old memories."""
    try:
        # Implementation for cleaning up old memories
        # This is a placeholder - implement based on your needs
        
        return {
            "success": True,
            "message": f"Cleanup {'simulated' if dry_run else 'completed'}",
            "days_old": days_old,
            "dry_run": dry_run
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Adjust module name as needed
        host="0.0.0.0",
        port=8001,  # Different from ChromaDB port
        reload=True,
        log_level="info"
    )