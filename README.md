## **Project Overview**

This project demonstrates how to implement a Retrieval-Augmented Generation (RAG) system using LlamaIndex, LlamaCPP, and HuggingFace embeddings. The system allows querying a vector store index built from documents stored in Google Drive, and it generates responses based on both retrieval and generation techniques.

### **Prerequisites**

`%pip install llama-index-embeddings-huggingface`  
`%pip install llama-index-llms-llama-cpp`  
`!pip install llama-index`  
`!pip -q install sentence-transformers`  
`!pip install llama-index-readers-file`

### **Code Explanation**

#### **1\. Import Libraries**

`from llama_index.core import SimpleDirectoryReader, VectorStoreIndex`  
`from llama_index.llms.llama_cpp import LlamaCPP`  
`from llama_index.llms.llama_cpp.llama_utils import (`  
    `messages_to_prompt,`  
    `completion_to_prompt,`  
`)`  
`from llama_index.embeddings.huggingface import HuggingFaceEmbedding`

#### **2\. Load and Configure the Language Model**

`model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"`  
`llm = LlamaCPP(`  
    `model_url=model_url,`  
    `model_path=None,`  
    `temperature=0.1,`  
    `max_new_tokens=256,`  
    `context_window=3900,`  
    `generate_kwargs={},`  
    `model_kwargs={"n_gpu_layers": 1},`  
    `messages_to_prompt=messages_to_prompt,`  
    `completion_to_prompt=completion_to_prompt,`  
    `verbose=True,`  
`)`

#### **3\. Generate and Stream Responses**

`response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")`  
`print(response.text)`

`response_iter = llm.stream_complete("Can you write me a poem about fast cars?")`  
`for response in response_iter:`  
    `print(response.delta, end="", flush=True)`

#### **4\. Set Up Embedding Model**

`embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")`

#### **5\. Load Documents and Create a Vector Store Index**

`documents = SimpleDirectoryReader("/content/drive/MyDrive/Data").load_data()`  
`index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)`

#### **6\. Query the Vector Store Index**

`query_engine = index.as_query_engine(llm=llm)`

#### **7\. Interactive Query Loop**

`while True:`  
    `query = input()`  
    `if query.lower() in ["thank you", "thanks"]:`  
        `break`  
    `response = query_engine.query(query)`  
    `print(response)`

### **How the RAG System Works**

1. **Retrieval**: The vector store index retrieves relevant documents based on the query.  
2. **Augmented Generation**: The LlamaCPP model generates responses informed by the retrieved documents, creating a powerful RAG system.

### **Conclusion**

This project showcases how to combine retrieval-based methods with generative models to build an effective RAG system, leveraging the strengths of both approaches.

