from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import GPT4AllEmbeddings

#embedding = GPT4AllEmbeddings()

#vectorstore = InMemoryVectorStore(embedding)

#loader = PyPDFLoader("C:/Users/dat.vuphat/Downloads/1528_QĐ-MOBIFONE_Quy_chế_QLVHKT_mạng_VT_CNTT_2024.pdf")
#data = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#all_splits = text_splitter.split_documents(data)

#_ = vectorstore.add_documents(documents=all_splits)

#question = "Nhiệm vụ phòng kỹ thuật khai thác"
#docs = vectorstore.similarity_search(question, k=3)
 
llm = LlamaCpp(
    model_path="C:/Users/dat.vuphat/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q5_K_M.gguf",# path model của anh vào đây nhénhé
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant who helps answer questions \
results. \n <</SYS>> \n\n [INST] Give the most accurate and concise answer to what you have learned. Please answer the question in Vietnamese \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant who helps answer questions \
results. \n <</SYS>> \n\n [INST] Give the most accurate and concise answer to what you have learned. Please answer the question in Vietnamese \n\n {question} [/INST]""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
prompt

chain = prompt | llm
question = '' # nhập câu hỏi
chain.invoke({"question": question})