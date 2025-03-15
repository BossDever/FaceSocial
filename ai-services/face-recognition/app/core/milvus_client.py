import time
import uuid
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

try:
    from pymilvus import (
        connections, Collection, CollectionSchema, DataType, 
        FieldSchema, utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("Warning: pymilvus not available, using placeholder implementation.")

class MilvusClient:
    """
    Client for interacting with Milvus vector database.
    """
    
    def __init__(self, host: str = "milvus", port: int = 19530, 
                 collection_name: str = "face_embeddings", dim: int = 128):
        """
        Initialize the Milvus client.
        
        Parameters:
        - host: Milvus server host
        - port: Milvus server port
        - collection_name: Default collection name
        - dim: Dimension of face embeddings
        """
        self.host = host
        self.port = port
        self.default_collection_name = collection_name
        self.dim = dim
        self.connected = False
        
        # Try to connect to Milvus
        if MILVUS_AVAILABLE:
            try:
                connections.connect(host=host, port=port)
                self.connected = True
                print(f"Connected to Milvus at {host}:{port}")
            except Exception as e:
                print(f"Error connecting to Milvus: {str(e)}")
                print("Using placeholder implementation.")
        
        # Initialize placeholder storage if Milvus is not available
        if not self.connected:
            self._init_placeholder_storage()
    
    def _init_placeholder_storage(self):
        """
        Initialize a placeholder in-memory storage when Milvus is not available.
        """
        self.placeholder_storage = {}
        print("Initialized placeholder storage for development.")
    
    def create_collection(self, collection_name: str = None) -> bool:
        """
        Create a collection for storing face embeddings.
        
        Parameters:
        - collection_name: Name of the collection (uses default if None)
        
        Returns:
        - True if successful, False otherwise
        """
        collection_name = collection_name or self.default_collection_name
        
        if not self.connected:
            if collection_name not in self.placeholder_storage:
                self.placeholder_storage[collection_name] = []
            return True
        
        try:
            # Check if collection already exists
            if utility.has_collection(collection_name):
                print(f"Collection {collection_name} already exists.")
                return True
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="user_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="quality", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.INT64)
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Face embeddings collection for {collection_name}")
            
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",  # Use cosine similarity
                "index_type": "IVF_FLAT",  # Simple and effective index
                "params": {"nlist": 1024}  # Number of clusters
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            
            print(f"Created collection {collection_name} with index.")
            return True
            
        except Exception as e:
            print(f"Error creating collection {collection_name}: {str(e)}")
            return False
    
    def insert_embeddings(self, user_id: int, embeddings: List[np.ndarray], 
                           qualities: List[float], collection_name: str = None) -> List[str]:
        """
        Insert face embeddings into Milvus.
        
        Parameters:
        - user_id: User ID associated with the embeddings
        - embeddings: List of face embeddings to insert
        - qualities: Quality scores for each embedding
        - collection_name: Name of the collection (uses default if None)
        
        Returns:
        - List of inserted embedding IDs
        """
        collection_name = collection_name or self.default_collection_name
        
        # Generate unique IDs for each embedding
        embedding_ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        # Current timestamp
        timestamp = int(time.time())
        
        if not self.connected:
            # Use placeholder storage
            if collection_name not in self.placeholder_storage:
                self.placeholder_storage[collection_name] = []
            
            for i, embedding_id in enumerate(embedding_ids):
                self.placeholder_storage[collection_name].append({
                    "id": embedding_id,
                    "user_id": user_id,
                    "embedding": embeddings[i],
                    "quality": qualities[i],
                    "created_at": timestamp
                })
            
            print(f"Inserted {len(embeddings)} embeddings into placeholder storage for {collection_name}.")
            return embedding_ids
        
        try:
            # Ensure collection exists
            if not utility.has_collection(collection_name):
                self.create_collection(collection_name)
            
            # Get collection
            collection = Collection(collection_name)
            
            # Prepare data for insertion
            data = [
                embedding_ids,  # id field
                [user_id] * len(embeddings),  # user_id field
                embeddings,  # embedding field
                qualities,  # quality field
                [timestamp] * len(embeddings)  # created_at field
            ]
            
            # Insert data
            collection.insert(data)
            
            # Flush to ensure data is committed
            collection.flush()
            
            print(f"Inserted {len(embeddings)} embeddings into Milvus for user {user_id}.")
            return embedding_ids
            
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            return []
    
    def search_embeddings(self, query_embedding: np.ndarray, top_k: int = 5, 
                          min_similarity: float = 0.75, collection_name: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar face embeddings in Milvus.
        
        Parameters:
        - query_embedding: Query face embedding
        - top_k: Number of top matches to return
        - min_similarity: Minimum similarity threshold
        - collection_name: Name of the collection (uses default if None)
        
        Returns:
        - List of matches with user_id, similarity, and embedding_id
        """
        collection_name = collection_name or self.default_collection_name
        
        if not self.connected:
            # Use placeholder storage
            if collection_name not in self.placeholder_storage:
                return []
            
            results = []
            for item in self.placeholder_storage[collection_name]:
                # Calculate cosine similarity
                embedding = item["embedding"]
                similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                
                if similarity >= min_similarity:
                    results.append({
                        "user_id": item["user_id"],
                        "similarity": float(similarity),
                        "embedding_id": item["id"]
                    })
            
            # Sort by similarity (highest first) and take top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        
        try:
            # Ensure collection exists
            if not utility.has_collection(collection_name):
                print(f"Collection {collection_name} does not exist.")
                return []
            
            # Get collection
            collection = Collection(collection_name)
            collection.load()
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # Number of clusters to search
            }
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["user_id", "quality"]
            )
            
            # Process results
            matches = []
            for hits in results:
                for hit in hits:
                    if hit.distance >= min_similarity:  # Cosine similarity
                        matches.append({
                            "user_id": hit.entity.get("user_id"),
                            "similarity": float(hit.distance),
                            "embedding_id": hit.id
                        })
            
            return matches
            
        except Exception as e:
            print(f"Error searching embeddings: {str(e)}")
            return []
    
    def delete_embeddings(self, embedding_ids: List[str], collection_name: str = None) -> bool:
        """
        Delete face embeddings from Milvus.
    
        Parameters:
        - embedding_ids: List of embedding IDs to delete
        - collection_name: Name of the collection (uses default if None)
    
        Returns:
        - True if successful, False otherwise
        """
        collection_name = collection_name or self.default_collection_name
    
        if not self.connected:
        # Use placeholder storage
            if collection_name not in self.placeholder_storage:
                return False
        
        before_count = len(self.placeholder_storage[collection_name])
        self.placeholder_storage[collection_name] = [
            item for item in self.placeholder_storage[collection_name]
            if item["id"] not in embedding_ids
        ]
        after_count = len(self.placeholder_storage[collection_name])
        
        print(f"Deleted {before_count - after_count} embeddings from placeholder storage.")
        return True
    
    try:
        # Ensure collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist.")
            return False
        
        # Get collection
        collection = Collection(collection_name)
        
        # Delete by ID - แก้ไขจาก f-string เป็นการสร้าง expression ด้วยวิธีอื่น
        quoted_ids = [f'"{id}"' for id in embedding_ids]
        expr = f"id in [{', '.join(quoted_ids)}]"
        collection.delete(expr)
        
        print(f"Deleted {len(embedding_ids)} embeddings from Milvus.")
        return True
        
    except Exception as e:
        print(f"Error deleting embeddings: {str(e)}")
        return False