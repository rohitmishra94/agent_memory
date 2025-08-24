import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class VectorMemoryStorage:
    """Handles storing memories in ChromaDB with BGE-M3 embeddings and custom retrieval."""
    
    def __init__(self, collection, user_id: str):
        self.collection = collection
        self.user_id = user_id
        self._memory_counter = 0  # For sequential IDs
    
    async def store_memories(self, memory_output, session_id: str) -> List[str]:
        """
        Store memories from MemoryAgentOutput in ChromaDB collection.
        Returns list of memory IDs that were stored.
        """
        if memory_output.session_assessment == "discarded":
            logger.info(f"Session {session_id} discarded: {memory_output.discard_reason}")
            return []
        
        ids = []
        documents = []
        metadatas = []
        
        # Store regular memories
        for memory in memory_output.memories:
            memory_id = self._get_next_memory_id()
            
            # DOCUMENT: The searchable text content
            document = self._create_memory_document(memory)
            
            # METADATA: All structured information for filtering
            metadata = self._create_memory_metadata(memory, session_id, "regular")
            
            ids.append(memory_id)
            documents.append(document)
            metadatas.append(metadata)
            
            logger.debug(f"Prepared regular memory {memory_id}: {memory.category}")
        
        # Store workflow memories
        for workflow in memory_output.workflow_memories:
            memory_id = self._get_next_memory_id()
            
            # DOCUMENT: Include workflow pattern in searchable text
            document = self._create_workflow_document(workflow)
            
            # METADATA: Include workflow-specific fields
            metadata = self._create_workflow_metadata(workflow, session_id)
            
            ids.append(memory_id)
            documents.append(document)
            metadatas.append(metadata)
            
            logger.debug(f"Prepared workflow memory {memory_id}: {workflow.process_pattern}")
        
        # Add to ChromaDB collection
        if ids:
            await self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(ids)} memories for user {self.user_id} from session {session_id}")
        
        return ids
    
    def _get_next_memory_id(self) -> str:
        """Generate sequential memory ID for proper context retrieval."""
        self._memory_counter += 1
        return str(self._memory_counter)
    
    async def initialize_counter(self):
        """Initialize counter based on existing memories in collection."""
        try:
            # Get all existing memories to find the highest ID
            existing = await self.collection.get()
            if existing['ids']:
                max_id = max([int(id_str) for id_str in existing['ids'] if id_str.isdigit()])
                self._memory_counter = max_id
            logger.info(f"Initialized memory counter to {self._memory_counter}")
        except Exception as e:
            logger.warning(f"Could not initialize counter: {e}. Starting from 0.")
            self._memory_counter = 0
    
    def _create_memory_document(self, memory) -> str:
        """Create searchable document text for regular memory."""
        return f"{memory.memory} {memory.context}"
    
    def _create_workflow_document(self, workflow) -> str:
        """Create searchable document text for workflow memory."""
        return f"{workflow.memory} {workflow.context} {workflow.process_pattern} Tools: {' '.join(workflow.tools_used)}"
    
    def _create_memory_metadata(self, memory, session_id: str, memory_type: str) -> Dict[str, Any]:
        """Create metadata for regular memory."""
        metadata = {
            "user_id": self.user_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "category": memory.category,
            "emotional_tone": memory.emotional_tone or "neutral",
            "tags": ",".join(memory.tags),
            "context": memory.context,
            "timestamp": datetime.utcnow().isoformat(),
            "created_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "memory_text": memory.memory,  # Store original memory text
        }
        
        # Add individual tag fields for easier filtering
        for i, tag in enumerate(memory.tags[:5]):
            metadata[f"tag_{i}"] = tag
            
        return metadata
    
    def _create_workflow_metadata(self, workflow, session_id: str) -> Dict[str, Any]:
        """Create metadata for workflow memory."""
        metadata = {
            "user_id": self.user_id,
            "session_id": session_id,
            "memory_type": "workflow",
            "category": "Workflow",
            "emotional_tone": workflow.emotional_tone or "neutral",
            "tags": ",".join(workflow.tags),
            "context": workflow.context,
            "process_pattern": workflow.process_pattern,
            "tools_used": ",".join(workflow.tools_used),
            "timestamp": datetime.utcnow().isoformat(),
            "created_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "memory_text": workflow.memory,
        }
        
        # Add individual tag fields
        for i, tag in enumerate(workflow.tags[:5]):
            metadata[f"tag_{i}"] = tag
            
        return metadata
    
    async def get_relevant_memories(
        self, 
        query: str, 
        get_full_context_func,
        model,
        n_results: int = 10,
        top: int = 3,
        category_filter: Optional[str] = None,
        memory_type_filter: Optional[str] = None,
        recent_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get relevant memories using your custom get_full_context function.
        
        Args:
            query: Search query text
            get_full_context_func: Your custom retrieval function
            model: BGE-M3 model instance
            n_results: Number of initial results to retrieve
            top: Number of top contexts to return
            category_filter: Filter by memory category
            memory_type_filter: Filter by memory type (regular/workflow)  
            recent_days: Only include memories from last N days
        """
        
        # Build metadata filters if needed
        where_conditions = {"user_id": self.user_id}
        
        if category_filter:
            where_conditions["category"] = category_filter
            
        if memory_type_filter:
            where_conditions["memory_type"] = memory_type_filter
            
        if recent_days:
            cutoff_date = (datetime.utcnow() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
            where_conditions["created_date"] = {"$gte": cutoff_date}
        
        # Use your custom retrieval function
        try:
            logger.info(f"Retrieving memories for query: '{query}' with filters: {where_conditions}")
            
            # If you have filters, you might need to modify get_full_context to handle them
            # For now, using your function as-is
            top_context, top_ids = await get_full_context_func(
                query=[query], 
                collection=self.collection, 
                n_results=n_results, 
                top=top
            )
            
            # Get full metadata for the retrieved memories
            retrieved_memories = await self.collection.get(ids=top_ids)
            
            # Filter by metadata conditions if needed
            if len(where_conditions) > 1:  # More than just user_id
                filtered_indices = []
                for i, metadata in enumerate(retrieved_memories['metadatas']):
                    if self._matches_filters(metadata, where_conditions):
                        filtered_indices.append(i)
                
                # Filter all results
                top_context = [top_context[i] for i in filtered_indices if i < len(top_context)]
                top_ids = [top_ids[i] for i in filtered_indices if i < len(top_ids)]
                filtered_metadatas = [retrieved_memories['metadatas'][i] for i in filtered_indices]
            else:
                filtered_metadatas = retrieved_memories['metadatas']
            
            logger.info(f"Retrieved {len(top_context)} relevant memories")
            
            return {
                "contexts": top_context,
                "ids": top_ids,
                "metadatas": filtered_metadatas,
                "query": query,
                "total_found": len(top_context)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {
                "contexts": [],
                "ids": [],
                "metadatas": [],
                "query": query,
                "total_found": 0,
                "error": str(e)
            }
    
    def _matches_filters(self, metadata: Dict[str, Any], where_conditions: Dict[str, Any]) -> bool:
        """Check if metadata matches filter conditions."""
        for key, expected_value in where_conditions.items():
            if key not in metadata:
                return False
                
            if isinstance(expected_value, dict):
                # Handle date range queries like {"$gte": "2025-01-01"}
                actual_value = metadata[key]
                for operator, value in expected_value.items():
                    if operator == "$gte" and actual_value < value:
                        return False
                    elif operator == "$lte" and actual_value > value:
                        return False
            else:
                # Direct equality check
                if metadata[key] != expected_value:
                    return False
        return True
    
    async def get_recent_context(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories for conversation context."""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            # Get recent memories
            all_memories = await self.collection.get(
                where={
                    "user_id": self.user_id,
                    "created_date": {"$gte": cutoff_date}
                }
            )
            
            # Sort by timestamp and limit
            if all_memories['metadatas']:
                sorted_memories = sorted(
                    zip(all_memories['documents'], all_memories['metadatas'], all_memories['ids']),
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )[:limit]
                
                return [
                    {
                        "document": doc,
                        "metadata": meta,
                        "id": mem_id
                    }
                    for doc, meta, mem_id in sorted_memories
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent context: {e}")
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            all_memories = await self.collection.get(
                where={"user_id": self.user_id}
            )
            
            if not all_memories['metadatas']:
                return {"total_memories": 0, "categories": {}, "memory_types": {}}
            
            # Count by category and type
            categories = {}
            memory_types = {}
            tags = {}
            
            for metadata in all_memories['metadatas']:
                # Count categories
                cat = metadata.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
                
                # Count memory types
                mem_type = metadata.get('memory_type', 'Unknown')
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                
                # Count tags
                tag_list = metadata.get('tags', '').split(',')
                for tag in tag_list:
                    if tag.strip():
                        tags[tag.strip()] = tags.get(tag.strip(), 0) + 1
            
            return {
                "total_memories": len(all_memories['metadatas']),
                "categories": categories,
                "memory_types": memory_types,
                "top_tags": sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10],
                "user_id": self.user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}

# Example usage with your setup
async def example_usage():
    """Example of how to use VectorMemoryStorage with your ChromaDB setup."""
    
    # Your existing setup
    from FlagEmbedding import BGEM3FlagModel
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
    
    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return embedding_function_bge(input)

    def embedding_function_bge(text_list):
        return model.encode(text_list, return_dense=True)['dense_vecs']

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    default_ef = MyEmbeddingFunction()

    client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
    
    user_id = "user_123"
    collection = await client.get_or_create_collection(
        name=user_id,
        embedding_function=default_ef
    )
    
    # Initialize storage
    storage = VectorMemoryStorage(collection, user_id)
    await storage.initialize_counter()
    
    # Store memories (example)
    # memory_output = ... (from your memory agent)
    # stored_ids = await storage.store_memories(memory_output, "session_456")
    
    # Retrieve relevant memories using your custom function
    relevant = await storage.get_relevant_memories(
        query="React development preferences",
        get_full_context_func=get_full_context,  # Your function
        model=model,
        n_results=10,
        top=3,
        category_filter="Technical"
    )
    
    print(f"Found {relevant['total_found']} relevant memories")
    for i, context in enumerate(relevant['contexts']):
        print(f"{i+1}. {context[:100]}...")
    
    # Get recent context
    recent = await storage.get_recent_context(days=7)
    print(f"Recent context: {len(recent)} memories")
    
    # Get stats
    stats = await storage.get_memory_stats()
    print(f"Memory stats: {stats}")

# Helper functions you'll need
def get_unique_text_indices(text_list):
    """Your existing function."""
    unique_texts = {}
    unique_indices = []

    for i, text in enumerate(text_list):
        if text not in unique_texts:
            unique_texts[text] = i
            unique_indices.append(i)

    return unique_indices