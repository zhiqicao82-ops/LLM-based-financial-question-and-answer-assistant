"""
知识库构建脚本
流程：加载文档 -> 切分 -> 向量化 -> 存入 Chroma

运行方式：
    python ingest.py
"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config


def load_documents():
    """加载 data/ 目录下所有 markdown 文档"""
    loader = DirectoryLoader(
        config.DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"共加载 {len(docs)} 个文档")
    return docs


def split_documents(docs):
    """
    两级切分：
    1. 先按 markdown 标题切分，保留语义边界（每个问题/产品是独立单元）
    2. 对过长的块再按字符数二次切分，避免超出 embedding 模型上下文
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section_title")]
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " "],
    )

    all_chunks = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        header_chunks = header_splitter.split_text(doc.page_content)
        for chunk in header_chunks:
            # 保留来源文件名，便于回答时标注引用来源
            chunk.metadata["source"] = source_name
            sub_chunks = char_splitter.split_documents([chunk])
            all_chunks.extend(sub_chunks)

    print(f"切分后共 {len(all_chunks)} 个知识片段")
    return all_chunks


def build_vector_store(chunks):
    """将切分后的文本片段向量化并持久化存储"""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIR,
    )
    print(f"向量库已构建完成，持久化至: {config.PERSIST_DIR}")
    return vector_store


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
