
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryBufferMemory

from backend.config import app_settings, validate_environment
from backend.embeddings import initialize_embeddings


MAX_TOKEN_LIMIT = 2000


def _get_llm():
    validate_environment()
    return ChatGoogleGenerativeAI(
        model=app_settings.google_genai.model,
        api_key=app_settings.google_genai.api_key,
        temperature=0,
    )


def _get_retriever():
    validate_environment()
    embeddings = initialize_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=app_settings.pinecone.index_name,
        embedding=embeddings,
        namespace=app_settings.pinecone.chunks_workspace,
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


def build_memory():
    llm = _get_llm()
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=MAX_TOKEN_LIMIT,
        return_messages=True,
        memory_key="chat_history",
    )
    return memory


def build_retrieval_chain(retriever=None, llm=None):
    if retriever is None:
        retriever = _get_retriever()
    if llm is None:
        llm = _get_llm()

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", "Cho đoạn hội thoại và câu hỏi tiếp theo, viết lại câu hỏi thành một câu độc lập (standalone) để tìm trong tài liệu. Chỉ trả lời câu hỏi, không giải thích thêm."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Trả lời dựa trên ngữ cảnh sau. Nếu không đủ thông tin, nói rõ."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}\n\nContext:\n{context}"),
    ])
    stuff_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_chain)

    return retrieval_chain


def chat(question: str, memory: ConversationSummaryBufferMemory) -> dict[str, Any]:

    chain = build_retrieval_chain()
    # Load history từ memory (format list messages hoặc string tùy return_messages)
    memory_vars = memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])

    result = chain.invoke({
        "input": question,
        "chat_history": chat_history,
    })

    memory.save_context(
        {"input": question},
        {"output": result["answer"]},
    )

    return result
