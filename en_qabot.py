from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read tu VectorDB
# Đọc vector DB
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/caliex-all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

db = read_vectors_db()



# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

#Tao Prompt
template = """<|im_start|>system\n  You are a helpful, respectful and honest assistant. Always answer as helpfully
  as possible, while being safe.  Your answers should not include any harmful,
  unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
  that your responses are socially unbiased and positive in nature. If a
  question does not make any sense, or is not factually coherent, explain why
  instead of answering something not correct. If you don't know the answer to a
  question, please don't share false information.\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

# Chay cai chain
question = " How do we estimate these bigram or n-gram probabilities? ?"
response = llm_chain.invoke({"query": question})
print(response)