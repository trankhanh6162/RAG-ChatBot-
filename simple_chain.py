from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01  
    )
    return llm

# Tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

# Tạo simple chain
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

# Chạy thử chain
template = """<|im_start|>system
Bạn là một trợ lý AI. Trả lời ngắn gọn và trực tiếp nhất có thể. 
Không cần giải thích hoặc thêm thông tin không cần thiết.

<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simple_chain(prompt, llm)
question = "1+1 bằng mấy?"
response = llm_chain.invoke({"question": question})
print(response)