# JuridixAssistant

JuridixAssistant is a specialized Legal AI Assistant designed to help legal professionals efficiently retrieve relevant clauses from large repositories of legal documents — even when queries are imprecise or vague.

## Problem Statement

Legal professionals frequently struggle to efficiently retrieve relevant clauses from extensive document repositories, especially when using approximate or fuzzy phrasing. Traditional keyword-based tools often fall short, leading to time-consuming manual searches. Additionally, the deployment environment in legal firms often prohibits cloud-based solutions due to confidentiality requirements, necessitating on-premises solutions that are computationally efficient.

## Our Solution

We present a Legal AI Assistant that bridges the gap between vague user queries and the precise language of legal documents through a novel architecture that includes:

- A small fine-tuned language model (SLM) for:
  - Classifying the query into one of nine frequently used legal document types (e.g., bail, cession, protocole, etc.)
  - Reformulating or clarifying the user’s input to improve retrieval relevance
- A Retrieval-Augmented Generation (RAG) pipeline that:
  - Uses legal domain-optimized embeddings to search for the most relevant clauses
  - Limits search to the predicted document type, thus improving speed and accuracy

Our system integrates techniques such as parameter-efficient fine-tuning, quantization, and hard distillation to meet strict performance and privacy requirements.

## Key Features

- **Document Type Classification**: Uses a lightweight SLM to predict which of the nine supported legal document types is most relevant to the query.
- **Query Refinement**: Automatically rewrites imprecise queries to improve retrieval accuracy.
- **RAG-based Retrieval**: Operates over a reduced and focused document subset for faster, more accurate results.
- **High Accuracy**: Achieves 94.7% top-5 recall on real-world legal documents.
- **Resource-Aware**: Designed to run efficiently on in-house servers without requiring cloud access.

## Repository Contents

This repository contains the code for:

- Fine-tuning the SLM on legal query classification tasks
- Prompt engineering strategies for reformulation
- Data generation and preprocessing pipelines for training

**Note:** The front-end and full-stack integration code shown in demonstrations is proprietary and owned by the partnering law firm. It is not included in this repository due to confidentiality agreements.

## Deployment

The system is intended for on-premises deployment, enabling confidential and efficient legal search directly within law firm infrastructure.

## Acknowledgements

This project was developed in collaboration with a law firm that provided domain expertise and access to anonymized legal data for training and evaluation.
