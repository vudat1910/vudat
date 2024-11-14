import streamlit as st
import os
import numpy as np
import psycopg2
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_history_aware_retriever
from langgraph.checkpoint.memory import MemorySaver


    

# Tạo memory để lưu trữ các trạng thái của mô hình
memory = MemorySaver()



# Tải dữ liệu từ các nguồn
data_loader = PyPDFLoader(file_path='C:/Users/dat.vuphat/Downloads/Qui_dinh_VAS_FC.pdf')

data = data_loader.load()

# Phân chia văn bản để xử lý tốt hơn
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True)
all_splits = text_splitter.split_documents(data)

# Thiết lập API key cho Together và Langsmith
os.environ["TOGETHER_API_KEY"] = '54c325f0fce2b8c2e19f9148af51ecbcd552fb4d36d48e86215229f379e50c5b'
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ['OPENAI_API_KEY'] = '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'
# Tạo embeddings
embeddings = TogetherEmbeddings(
    model='togethercomputer/m2-bert-80M-8k-retrieval'
)
# Tạo vectorstore để lưu trữ vector hóa của các tài liệu
vectorstore = InMemoryVectorStore.from_documents(
    data,
    embedding=embeddings
)

# Kết nối PostgreSQL
conn = psycopg2.connect(dbname="vb", user="postgres", password="Chelsea@19102002", host="localhost", port="5432")
cursor = conn.cursor()

# Tạo bảng nếu chưa có
cursor.execute("""
CREATE TABLE IF NOT EXISTS vectors (
    document_id TEXT PRIMARY KEY,
    vector FLOAT8[]
)
""")

# Chèn vector vào bảng PostgreSQL (sử dụng UPSERT)
for idx, doc in enumerate(all_splits):
    vector = embeddings.embed_documents([doc.page_content])[0]
    document_id = doc.metadata.get('document_id', f'doc_{idx}')
    cursor.execute("""
        INSERT INTO vectors (document_id, vector)
        VALUES (%s, %s)
        ON CONFLICT (document_id) DO UPDATE SET vector = EXCLUDED.vector;
    """, (document_id, np.array(vector).tolist()))

# Lưu và đóng kết nối
conn.commit()
cursor.close()
conn.close()

# Khởi tạo retriever
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 50}
)

# Khởi tạo LLM từ Together
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)

# Tạo công cụ truy xuất dữ liệu
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post."
)

tools = [tool]

# Tạo prompt trả lời câu hỏi
system_prompt = (
    "Bạn là một trợ lý thông minh, chuyên trả lời câu hỏi. "
    "Hãy trả lời một cách ngắn gọn và chính xác dựa trên thông tin đã được cung cấp."
    "\n\nNếu không tìm thấy thông tin liên quan, hãy nói rằng bạn không biết."
    "\n\n{context}"
)

# Tạo prompt để gợi ý câu hỏi có tham chiếu lịch sử
contextualize_q_system_prompt = "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất, hãy tạo một câu hỏi có liên quan, bao gồm ngữ cảnh và nội dung của câu hỏi trước."
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Khởi tạo history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, 
    retriever, 
    contextualize_q_prompt
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Tạo chain trả lời câu hỏi
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# Kết hợp retriever và question_answer_chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Lưu trữ lịch sử chat
chat_history = []

# Khởi tạo chuỗi trò chuyện có lịch sử
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    get_session_history = get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    runnable = rag_chain
)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{np.random.randint(10000)}"

# Tạo giao diện chatbot
st.title('Chatbot')

# Hiển thị lịch sử trò chuyện
for message in st.session_state['chat_history']:
    if message['role'] == 'human':
        st.chat_message("user").write(message['content'])
    else:
        st.chat_message("assistant").write(message['content'])

# Nhận câu hỏi từ người dùng
user_input = st.chat_input("Bạn muốn hỏi gì?")

if user_input:
    # Lưu tin nhắn người dùng
    st.session_state['chat_history'].append({"role": "human", "content": user_input})
    st.chat_message("user").write(user_input)

    # Gọi hàm trả lời từ mô hình với session_id
    ai_response = conversational_rag_chain.invoke(
        {"input": user_input, "chat_history": st.session_state['chat_history']},
        {"configurable": {"session_id": st.session_state['session_id']}}
    )
    chatbot_response = ai_response["answer"]  # Lấy câu trả lời từ mô hình

    # Lưu phản hồi của chatbot
    st.session_state['chat_history'].append({"role": "assistant", "content": chatbot_response})
    st.chat_message("assistant").write(chatbot_response)
