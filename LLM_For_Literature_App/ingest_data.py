import os
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
print('Indexes are created successfully')
index.storage_context.persist()
