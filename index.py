import streamlit as st
import numpy as np
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_history_aware_retriever
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
    

# Tạo memory để lưu trữ các trạng thái của mô hình
memory = MemorySaver()

# Tải dữ liệu từ các nguồn
loader = PyPDFLoader(file_path='C:/Users/dat.vuphat/Downloads/mobiedu.pdf')
data = loader.load()

# Phân chia văn bản để xử lý tốt hơn
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100, 
    add_start_index=True)
all_splits = text_splitter.split_documents(data)

# Tạo embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Tạo vectorstore để lưu trữ vector hóa của các tài liệu
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

# Khởi tạo retriever
retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 8}
)

# Khởi tạo LLM từ Together
llm = ChatOllama(
    model="deepseek-r1")

# Tạo prompt trả lời câu hỏi
system_prompt = (
    "Bạn là trợ lý AI, chỉ được trả lời dựa trên dữ liệu trong file PDF được cung cấp, chỉ trả lời thông tin đã xử lý "
    "Trả lời hoàn toàn bằng tiếng việt"
    "Không sử dụng thông tin khác. Nếu không tìm thấy thông tin liên quan, hãy trả lời rằng bạn không biết."
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