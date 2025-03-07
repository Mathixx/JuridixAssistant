#!/usr/bin/env python3
"""
Generate Synthetic Dataset with Deepseek-v3

This script generates a synthetic dataset of (query, relevant documents) pairs from a corpus
of legal documents without labelers by leveraging Deepseek-v3.
"""

import json
import uuid
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer

# For improved sentence-based chunking
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# For quality evaluation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Input document paths
DOCUMENTS_DIR = './documents'
TRAIN_FILES = [
    f'{DOCUMENTS_DIR}/142.2020 07 02 - Contrat de prêt aux entreprises - CRAVEDI - BRED BANQUE POPULAIRE (signé).pdf',
    f'{DOCUMENTS_DIR}/Compare - PROTO CRAVEDI - 21 12 23 - Version signature.pdf'
]
VAL_FILES = [f'{DOCUMENTS_DIR}/2019 05 16 - MATHO - Bail commercial GUICLA vdef.pdf']

# Output paths
DATA_DIR = './data_deepseek'
Path(DATA_DIR).mkdir(exist_ok=True)
CACHE_DIR = './cache'
Path(CACHE_DIR).mkdir(exist_ok=True)

TRAIN_CORPUS_FPATH = f'{DATA_DIR}/train_corpus.json'
VAL_CORPUS_FPATH = f'{DATA_DIR}/val_corpus.json'
TRAIN_QUERIES_FPATH = f'{DATA_DIR}/train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/train_relevant_docs.json'
VAL_QUERIES_FPATH = f'{DATA_DIR}/val_queries.json'
VAL_RELEVANT_DOCS_FPATH = f'{DATA_DIR}/val_relevant_docs.json'
TRAIN_DATASET_FPATH = f'{DATA_DIR}/train_dataset.json'
VAL_DATASET_FPATH = f'{DATA_DIR}/val_dataset.json'

# Chunk parameters (in approximate words)
CHUNK_WORD_LIMIT = 1000
OVERLAP_WORDS = 200

def init_llm():
    """Initialize Deepseek-v3 6B model for text generation"""
    logging.info("Loading Deepseek-v3 6B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-instruct", 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file with error handling."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
    return text

def cache_extracted_text(file_path: str) -> str:
    """Use caching to avoid reprocessing PDFs."""
    cache_file = os.path.join(CACHE_DIR, os.path.basename(file_path) + ".txt")
    if os.path.exists(cache_file):
        logging.info(f"Loading cached text for {file_path}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        logging.info(f"Extracting text from {file_path}")
        text = extract_text_from_pdf(file_path)
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text

def chunk_text(text: str, chunk_word_limit: int = CHUNK_WORD_LIMIT, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    """
    Split text into chunks using sentence boundaries.
    Each chunk contains approximately chunk_word_limit words,
    with an overlap of approximately overlap_words words between chunks.
    """
    sentences = sent_tokenize(text)
    # Compute word counts for each sentence
    word_counts = [len(sentence.split()) for sentence in sentences]
    chunks = []
    i = 0
    while i < len(sentences):
        current_words = 0
        j = i
        while j < len(sentences) and current_words < chunk_word_limit:
            current_words += word_counts[j]
            j += 1
        chunk = " ".join(sentences[i:j])
        chunks.append(chunk)
        # Determine overlap: backtrack until approximately overlap_words are reached
        overlap = 0
        k = j - 1
        while k >= i and overlap < overlap_words:
            overlap += word_counts[k]
            k -= 1
        # Ensure progress even if overlap_words cannot be met
        i = max(i + 1, k + 1)
    return chunks

def load_corpus(files: List[str], verbose: bool = False) -> Dict[str, str]:
    """Load and chunk PDF documents into a corpus using parallel processing and caching."""
    corpus = {}

    def process_file(file_path: str) -> List[Tuple[str, str]]:
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return []
        text = cache_extracted_text(file_path)
        chunks = chunk_text(text)
        return [(str(uuid.uuid4()), chunk) for chunk in chunks]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, fp) for fp in files]
        for future in futures:
            for node_id, chunk in future.result():
                corpus[node_id] = chunk

    if verbose:
        logging.info(f"Loaded and chunked into {len(corpus)} text chunks")
    return corpus

def generate_query_with_llm(model, tokenizer, prompt: str) -> str:
    """Generate a response using the Deepseek model with error handling."""
    formatted_prompt = f"### Instruction: {prompt}\n\n### Response:"
    try:
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error generating query: {e}")
        return ""

def generate_queries(
    corpus: Dict[str, str],
    model,
    tokenizer,
    num_questions_per_chunk: int = 2,
    prompt_template: str = None,
    verbose: bool = False,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Generate synthetic questions for each document chunk using Deepseek-v3.
    The prompt template is cleaned and formatted robustly.
    """
    # Use a cleaned-up prompt (no extra indentation)
    prompt_template = prompt_template or (
        "Contexte ci-dessous.\n\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "En utilisant uniquement les informations du contexte et non des connaissances préalables, "
        "générez {num_questions_per_chunk} questions qui pourraient être posées sur ce document juridique. "
        "Les questions doivent être diverses et se concentrer sur les termes juridiques clés, les conditions, "
        "les obligations et les détails importants du document. Limitez les questions aux informations fournies.\n\n"
        "Important: Les questions doivent être en français."
    )

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items(), desc="Generating queries"):
        prompt = prompt_template.format(
            context_str=text,
            num_questions_per_chunk=num_questions_per_chunk
        )
        response = generate_query_with_llm(model, tokenizer, prompt)
        # Clean up the generated response
        result = response.strip().split("\n")
        questions = [
            re.sub(r"^\s*\d+[\).:\s-]*", "", question).strip() for question in result
        ]
        questions = [q for q in questions if len(q) > 0]
        if verbose:
            logging.info(f"Generated questions for chunk {node_id}: {questions}")
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]
    return queries, relevant_docs

def evaluate_dataset_quality(queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, List[str]]):
    """
    Compute a simple quality metric: average cosine similarity between each query and its corresponding document.
    Logs a warning if the average similarity is below a threshold.
    """
    query_list, doc_list = [], []
    for qid, query in queries.items():
        doc_ids = relevant_docs.get(qid, [])
        if doc_ids:
            doc_text = corpus.get(doc_ids[0], "")
            if doc_text:
                query_list.append(query)
                doc_list.append(doc_text)
    if not query_list or not doc_list:
        logging.warning("No valid query-document pairs found for quality evaluation.")
        return
    vectorizer = TfidfVectorizer()
    combined = query_list + doc_list
    tfidf = vectorizer.fit_transform(combined)
    query_tfidf = tfidf[:len(query_list)]
    doc_tfidf = tfidf[len(query_list):]
    sims = [cosine_similarity(query_tfidf[i], doc_tfidf[i]).item() for i in range(len(query_list))]
    avg_sim = np.mean(sims)
    logging.info(f"Average cosine similarity between queries and documents: {avg_sim:.3f}")
    if avg_sim < 0.2:
        logging.warning("Low average query-document similarity. Generated queries might not be well aligned with the document content.")

def main():
    """Main function to generate the dataset and evaluate its quality."""
    model, tokenizer = init_llm()

    logging.info("Generating training corpus...")
    train_corpus = load_corpus(TRAIN_FILES, verbose=True)
    with open(TRAIN_CORPUS_FPATH, 'w', encoding='utf-8') as f:
        json.dump(train_corpus, f, ensure_ascii=False, indent=2)

    logging.info("Generating validation corpus...")
    val_corpus = load_corpus(VAL_FILES, verbose=True)
    with open(VAL_CORPUS_FPATH, 'w', encoding='utf-8') as f:
        json.dump(val_corpus, f, ensure_ascii=False, indent=2)

    logging.info("Generating training queries...")
    train_queries, train_relevant_docs = generate_queries(train_corpus, model, tokenizer, verbose=True)
    with open(TRAIN_QUERIES_FPATH, 'w', encoding='utf-8') as f:
        json.dump(train_queries, f, ensure_ascii=False, indent=2)
    with open(TRAIN_RELEVANT_DOCS_FPATH, 'w', encoding='utf-8') as f:
        json.dump(train_relevant_docs, f, ensure_ascii=False, indent=2)

    logging.info("Generating validation queries...")
    val_queries, val_relevant_docs = generate_queries(val_corpus, model, tokenizer, verbose=True)
    with open(VAL_QUERIES_FPATH, 'w', encoding='utf-8') as f:
        json.dump(val_queries, f, ensure_ascii=False, indent=2)
    with open(VAL_RELEVANT_DOCS_FPATH, 'w', encoding='utf-8') as f:
        json.dump(val_relevant_docs, f, ensure_ascii=False, indent=2)

    logging.info("Creating final datasets...")
    train_dataset = {
        'queries': train_queries,
        'corpus': train_corpus,
        'relevant_docs': train_relevant_docs,
    }
    with open(TRAIN_DATASET_FPATH, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)

    val_dataset = {
        'queries': val_queries,
        'corpus': val_corpus,
        'relevant_docs': val_relevant_docs,
    }
    with open(VAL_DATASET_FPATH, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=2)

    logging.info("Dataset generation complete!")
    logging.info(f"Training dataset saved to: {TRAIN_DATASET_FPATH}")
    logging.info(f"Validation dataset saved to: {VAL_DATASET_FPATH}")

    # Run a simple quality evaluation
    logging.info("Evaluating training dataset quality...")
    evaluate_dataset_quality(train_queries, train_corpus, train_relevant_docs)
    logging.info("Evaluating validation dataset quality...")
    evaluate_dataset_quality(val_queries, val_corpus, val_relevant_docs)

if _name_ == "_main_":
    main()