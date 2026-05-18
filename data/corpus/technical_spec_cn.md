---
doc_id: technical_spec_cn
title: 技术规格 CN
language: zh
owner: Architecture
version: 2026.03
---

# RAG 问答服务技术规格

## 检索模式
服务必须支持 vector-only 和 hybrid 两种检索模式。vector-only 使用语义向量相似度排序，适合自然语言问题；hybrid 将向量相似度与 BM25 关键词分数融合，适合中英文混合、专有名词、制度条款编号等场景。

## 重排序
重排序器必须通过配置开关启用或关闭，不允许改代码后才能切换。重排序器只处理第一阶段召回的 top-k 文档块，并输出最终上下文列表。上线前需要比较 vector-only、hybrid、hybrid+rerank 三种配置的 Context Precision、Faithfulness 和端到端延迟。

## 缓存
服务应缓存规范化后的 query、检索模式、reranker 开关和模型版本组合。缓存命中时仍需记录结构化日志，包括 request_id、cache_hit、latency_ms 和 token_usage 字段。缓存 TTL 推荐从 5 到 15 分钟开始压测。

## 性能目标
单实例至少支持 5 个并发请求。90% 的问答请求端到端延迟应低于 10 秒。评估报告必须给出 p50、p95、p90 延迟以及拒答率、缓存命中率和 token 使用量。
