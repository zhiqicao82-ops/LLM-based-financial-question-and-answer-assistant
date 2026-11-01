"""
RAG 核心逻辑：
  用户问题 -> 混合检索 -> 拼接上下文 -> DeepSeek 生成答案
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import langchain_openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import config
from ingest import load_documents, split_documents

try:
    from langchain_openai import ChatOpenAI
    print("使用 langchain_openai")
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        print("使用 langchain_community.chat_models")
    except ImportError:
        from langchain.chat_models import ChatOpenAI
        print("使用 langchain.chat_models")
load_dotenv()

SYSTEM_PROMPT = """你是一名专业、严谨的金融知识助手。请严格遵守以下规则：
1. 只根据下方提供的【参考资料】回答问题，不要使用你自己的知识库之外的信息编造答案。
2. 如果参考资料中没有足够信息回答问题，请直接说"根据现有知识库，暂时无法回答该问题"，不要猜测或编造。
3. 回答末尾用【来源：xxx】的格式标注答案依据的资料来源文件名。
4. 涉及具体费率、收益率等数字时，务必与参考资料完全一致，不要四舍五入或修改。
5. 回答语言简洁、专业，避免不必要的免责声明堆砌。

【参考资料】
{context}
"""

USER_PROMPT = "{question}"


def build_retriever():
    """构建混合检索器（向量 + BM25）"""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embeddings,
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    docs = load_documents()
    chunks = split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.TOP_K

    def hybrid_search(query):
        vec_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        seen_contents = set()
        merged = []
        for doc in vec_docs + bm25_docs:
            content_key = doc.page_content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                merged.append(doc)
        return merged[:config.TOP_K]

    return RunnableLambda(hybrid_search)


def format_docs(docs):
    """格式化检索结果"""
    formatted = []
    for i, doc in enumerate(docs[: config.FINAL_K], start=1):
        source = doc.metadata.get("source", "未知来源")
        if source:
            source = os.path.basename(source)
        section = doc.metadata.get("section_title", "")
        formatted.append(f"[片段{i} | 来源:{source} {section}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain():
    """构建完整的 RAG 链路"""
    retriever = build_retriever()

    # 直接填写你的密钥
    llm = ChatOpenAI(
        openai_api_key="sk-4df4090fd76345199fbfb5b5932a6ed8",
        openai_api_base="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        temperature=config.LLM_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    rag_chain = build_rag_chain()
    test_questions = [
        "夏普比率是什么意思？",
        "成长精选混合基金的管理费率是多少？",
    ]
    for q in test_questions:
        print(f"\n问：{q}")
        print(f"答：{rag_chain.invoke(q)}")