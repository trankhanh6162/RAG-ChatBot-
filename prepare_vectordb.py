from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

pdf_data_path = 'data'
vector_db_path = 'vectorstores/db_faiss'


def create_db_from_text():
    raw_text = """Trí tuệ nhân tạo (AI) là một lĩnh vực trong khoa học máy tính chuyên nghiên cứu và phát triển các hệ thống có khả năng mô phỏng trí tuệ của con người. Những hệ thống này có thể thực hiện các nhiệm vụ như học tập, lập luận, giải quyết vấn đề, nhận diện hình ảnh, hiểu ngôn ngữ tự nhiên và thậm chí đưa ra quyết định. Trí tuệ nhân tạo hiện đang được ứng dụng rộng rãi trong nhiều lĩnh vực như y tế, giáo dục, tài chính, giao thông và sản xuất công nghiệp. Với sự phát triển không ngừng của công nghệ, AI hứa hẹn sẽ đóng vai trò ngày càng quan trọng trong cuộc sống hiện đại, góp phần nâng cao chất lượng cuộc sống và thúc đẩy sự đổi mới sáng tạo."""

    # Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Đưa vào FAISS Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db


def create_db_from_files():
    # Khai báo loader để quét toàn bộ thư mục data
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Lưu vào FAISS Vector DB
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


create_db_from_files()
