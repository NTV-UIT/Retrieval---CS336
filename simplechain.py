from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Cau hinh
model_file = "models/llama-2-7b-chat.Q8_0.gguf"


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
    prompt = PromptTemplate(template = template, input_variables=["question"])
    return prompt


# Tao simple chain
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# Chay thu chain

template = """<|im_start|>system
  You are a helpful, respectful and honest assistant. Always answer as helpfully
  as possible, while being safe.  Your answers should not include any harmful,
  unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure
  that your responses are socially unbiased and positive in nature. If a
  question does not make any sense, or is not factually coherent, explain why
  instead of answering something not correct. If you don't know the answer to a
  question, please don't share false information.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = creat_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt, llm)

question = " How do we estimate these bigram or n-gram probabilities?"

response = llm_chain.invoke({"question":question})
print(response)
