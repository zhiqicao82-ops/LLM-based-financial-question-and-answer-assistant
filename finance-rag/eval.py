"""
简单效果评估脚本
用一批人工标注的 (问题, 应命中的来源文档) 测试集，
评估检索环节的命中率（Hit Rate），这是简历里"提升问答准确率"的量化依据。

运行方式：
    python eval.py
"""
import config
from rag_chain import build_retriever

# 测试集：question -> 期望检索到的来源文件名
EVAL_SET = [
    {"question": "夏普比率的计算公式是什么？", "expected_source": "faq_basics.md"},
    {"question": "LPR是由谁发布的？", "expected_source": "faq_basics.md"},
    {"question": "成长精选混合基金的风险等级是多少？", "expected_source": "product_info.md"},
    {"question": "天天盈货币基金的申购费率是多少？", "expected_source": "product_info.md"},
    {"question": "稳健增利债券基金适合什么样的投资者？", "expected_source": "product_info.md"},
]


def evaluate_retrieval():
    retriever = build_retriever()
    hit_count = 0

    for case in EVAL_SET:
        results = retriever.invoke(case["question"])
        retrieved_sources = {doc.metadata.get("source") for doc in results[: config.FINAL_K]}
        hit = case["expected_source"] in retrieved_sources
        hit_count += int(hit)
        status = "✓" if hit else "✗"
        print(f"{status} 问题：{case['question']}  期望来源：{case['expected_source']}  实际召回：{retrieved_sources}")

    hit_rate = hit_count / len(EVAL_SET)
    print(f"\n检索命中率 (Hit Rate@{config.FINAL_K}): {hit_rate:.2%} ({hit_count}/{len(EVAL_SET)})")
    return hit_rate


if __name__ == "__main__":
    evaluate_retrieval()
