# 基于大语言模型的金融助手（RAG）

个人项目：基于 LLM 与 RAG 架构的金融问答机器人，集成金融知识库与投资产品信息，
支持语义检索 + 关键词混合检索，提升问答准确率与相关性。

## 技术架构

```
用户问题
   │
   ├─► 向量检索 (Chroma + bge-small-zh)  ─┐
   │                                       ├─► EnsembleRetriever 融合 ─► Top-K 片段
   └─► 关键词检索 (BM25)                  ─┘
                                               │
                                        拼接为 Prompt 上下文
                                               │
                                        Claude API 生成答案
                                               │
                                     答案 + 【来源:xxx】引用标注
```

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

2. 配置 API Key（新建 `.env` 文件）
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

3. 构建知识库（首次运行/更新知识库文档后执行）
   ```bash
   python ingest.py
   ```

4. 命令行测试
   ```bash
   python rag_chain.py
   ```

5. 启动网页演示
   ```bash
   streamlit run app.py
   ```

6. 评估检索效果（量化"问答准确率提升"）
   ```bash
   python eval.py
   ```

## 目录说明

| 文件 | 作用 |
|---|---|
| `config.py` | 模型、路径、切分/检索参数集中配置 |
| `data/` | 知识库原始文档（示例：金融基础知识 + 产品信息，可替换为真实语料） |
| `ingest.py` | 文档加载 → 语义切分 → 向量化 → 存入 Chroma |
| `rag_chain.py` | 混合检索（向量+BM25）→ Prompt拼接 → Claude生成 |
| `eval.py` | 检索命中率评估，用于量化优化效果 |
| `app.py` | Streamlit 演示界面 |

## 后续可扩展方向（简历亮点延伸）

- **换用生产级向量库**：Chroma → Milvus/Qdrant，支持分布式部署和更复杂的元数据过滤
- **加入 Rerank 环节**：用 `bge-reranker-v2-m3` 对 Top-K 结果二次精排，进一步提升相关性
- **扩充真实知识库**：接入基金招募说明书 PDF、监管政策文件，用 `unstructured` 处理表格和版面
- **Query改写/意图识别**：先判断问题是否需要检索、是否需要拆解为多个子问题
- **构建更大规模评估集**：用 LLM-as-judge 对答案质量打分，形成准确率提升的量化报告（这是简历里最有说服力的数据支撑）
