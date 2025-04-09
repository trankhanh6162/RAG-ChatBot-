from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- Cấu hình ---
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# --- Kiểm tra model ---
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ Không tìm thấy model tại {model_file}")

# --- Load LLM ---
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0
    )
    return llm

# --- Tạo Prompt Template ---
def create_prompt():
    template = """<|im_start|>system
Bạn là một trợ lý AI. Chỉ sử dụng thông tin sau đây để trả lời câu hỏi. 
Nếu bạn không biết câu trả lời, hãy trả lời "Tôi không biết". 
Không được suy đoán hoặc tạo ra câu trả lời từ trí tưởng tượng.

{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Tạo QA Chain ---
def create_qa_chain(prompt, llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Load VectorDB ---
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(vector_db_path):
        raise FileNotFoundError(f"❌ Không tìm thấy vector DB tại {vector_db_path}")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# --- Khởi chạy ---
try:
    print("🔄 Đang tải VectorDB...")
    db = read_vectors_db()
    print("✅ VectorDB đã được tải thành công.")

    print("🔄 Đang tải mô hình...")
    llm = load_llm(model_file)
    print("✅ Mô hình đã được tải thành công.")

    print("🔄 Đang tạo prompt...")
    prompt = create_prompt()

    print("🔄 Đang tạo QA chain...")
    qa_chain = create_qa_chain(prompt, llm, db)

    # --- Đặt câu hỏi ---
    question = "Sự kiện lịch sử năm 1908 ở Việt Nam là gì?"
    response = qa_chain.invoke({"query": question})  # ✅ Sửa lỗi tại đây

    print(f"✅ Câu trả lời: {response}")

except FileNotFoundError as e:
    print(f"❌ Lỗi: {e}")
except Exception as e:
    print(f"❌ Đã xảy ra lỗi: {e}")
