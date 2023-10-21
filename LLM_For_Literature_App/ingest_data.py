import os
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#print(os.getenv("OPENAI_API_KEY"))
# from llama_index import SimpleDirectoryReader
# from llama_index import VectorStoreIndex

# documents = SimpleDirectoryReader('./data').load_data()
# index = VectorStoreIndex.from_documents(documents)
# print('Indexes are created successfully')
# index.storage_context.persist()

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

############################ Prepare indexes

from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)
# load documents
documents = SimpleDirectoryReader(
    "./data"
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

index.storage_context.persist()