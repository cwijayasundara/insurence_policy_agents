import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os
from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

nest_asyncio.apply()

_ = load_dotenv()

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small"
)

# Create New Index from Document

DATA_DIR = "./doc"
PERSIST_DIR = "./insurence_policy_storage"

if os.path.exists(PERSIST_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Creating new index...")
    file_path = "./doc/pb116349-business-health-select-handbook-1024-pdfa.pdf"
    documents = LlamaParse(result_type="markdown").load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# Run a Query Against the Index
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "Whats the cashback amount for optical expenses?"
)
print(response)