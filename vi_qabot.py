from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vi_vectorstores/vi_db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.001
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
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)


prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

# Chay cai chain
question = "  Các chủ đề trong XLNNTN bao gồm? "
response = llm_chain.invoke({"query": question})
print(response)

# # Loại bỏ các ký tự đặc biệt
# clean_response = ''.join(e for e in response if e.isalnum() or e.isspace())
# print("Cleaned response:", clean_response)
# print(response)