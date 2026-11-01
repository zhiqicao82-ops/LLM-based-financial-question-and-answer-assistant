"""
Streamlit 演示界面

运行方式：
    streamlit run app.py
"""
import streamlit as st
from rag_chain import build_rag_chain

st.set_page_config(page_title="金融知识问答助手", page_icon="💰")
st.title("💰 基于 LLM 的金融助手")
st.caption("基于 RAG 架构：向量检索 + BM25混合检索 + Claude 生成")

if "chain" not in st.session_state:
    with st.spinner("正在加载知识库与模型..."):
        st.session_state.chain = build_rag_chain()

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("请输入你的金融问题，例如：什么是夏普比率？")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索知识库并生成答案中..."):
            answer = st.session_state.chain.invoke(question)
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))

with st.sidebar:
    st.subheader("关于本项目")
    st.markdown(
        """
        - **检索**：Chroma 向量检索 + BM25 关键词检索 混合
        - **Embedding**：BAAI/bge-small-zh-v1.5（本地运行）
        - **生成**：Claude API
        - **知识库**：金融基础知识 + 投资产品信息（示例数据，可替换为真实语料）
        """
    )
