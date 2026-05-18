---
doc_id: compliance_guide_en
title: Compliance Guide EN
language: en
owner: Compliance
version: 2026.02
---

# Compliance Guide

## Data Classification
Internal documents are classified as Public, Internal, Confidential, or Restricted. Confidential and Restricted documents must not be pasted into public tools unless an approved enterprise agreement and data protection review are in place.

## Personal Data
Personal data includes names, personal email addresses, phone numbers, government identifiers, precise address, payroll information, and health-related information. Systems must avoid logging raw personal data. Logs should use redaction or stable non-reversible hashes where diagnosis needs correlation.

## Prompt Injection
A prompt injection is text that attempts to override system or developer instructions, exfiltrate hidden prompts, or force the assistant to ignore the retrieved context. The RAG service must treat retrieved text as untrusted data and follow the system instruction that answers must be grounded only in retrieved context.

## Safety Refusal
The assistant must refuse requests that are outside the internal knowledge base, have insufficient retrieval confidence, ask for secrets, or attempt to bypass security controls. A refusal should explain the limitation and provide a safe next step.

## 个人数据日志记录（中文摘要）
个人数据包括姓名、个人邮箱地址、电话号码、政府身份标识、精确地址、薪酬信息和健康相关信息。系统不得在日志中原样记录个人手机号和邮箱，应使用脱敏、遮蔽或稳定的不可逆哈希。若诊断需要关联同一用户，也应避免保存明文个人数据。
