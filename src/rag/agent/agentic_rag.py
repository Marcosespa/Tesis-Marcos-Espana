#!/usr/bin/env python3
"""
Agente RAG con Mistral 7B vía Ollama para generación
y Weaviate (ImprovedSemanticSearch) para recuperación.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# PYTHONPATH support
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import weaviate
import numpy as np
from dataclasses import dataclass

from rag.search.improved_search import (
    ImprovedSemanticSearch,
    SearchStrategy,
    SearchResult,
)

try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None  # type: ignore


@dataclass
class RetrievedPassage:
    doc_id: str
    title: str
    pages: str
    text: str


def format_citations(passages: List[RetrievedPassage]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p.title} ({p.doc_id}) páginas {p.pages}")
    return "\n".join(lines)


def build_prompt(query: str, passages: List[RetrievedPassage]) -> str:
    context_blocks = []
    for i, p in enumerate(passages, 1):
        context_blocks.append(
            f"[Contexto {i}]\nTitulo: {p.title}\nDocumento: {p.doc_id}\nPaginas: {p.pages}\nTexto:\n{p.text}\n"
        )
    context = "\n\n".join(context_blocks)
    prompt = (
        "Eres un asistente experto en ciberseguridad. Responde en español, factual y conciso.\n"
        "Cita siempre las fuentes relevantes usando el índice [n]. Si la evidencia es insuficiente, dilo.\n\n"
        f"Pregunta: {query}\n\n"
        f"Contexto:\n{context}\n\n"
        "Instrucciones:\n"
        "- Integra las evidencias sin inventar.\n"
        "- Menciona los índices [n] al final de frases que apoyen.\n"
        "- Si hay pasos prácticos, ponlos en lista corta.\n"
    )
    return prompt


def generate_with_ollama(
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    if OllamaClient is None:
        raise RuntimeError("ollama no está instalado. Instala con: pip install ollama")
    client = OllamaClient(host=host)
    resp = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return resp.get("response", "").strip()


def agentic_rag(
    query: str,
    weaviate_host: str = "localhost",
    http_port: int = 8080,
    grpc_port: int = 50051,
    collection: str = "BookChunk",
    mistral_model: str = "mistral",
    k: int = 8,
    strategy: str = "multi_stage",
    rerank: bool = True,
) -> Dict[str, Any]:
    # Conectar a Weaviate
    client = weaviate.connect_to_local(host=weaviate_host, port=http_port, grpc_port=grpc_port)
    try:
        retriever = ImprovedSemanticSearch(client, collection)
        strat = SearchStrategy(strategy)
        results: List[SearchResult] = retriever.search(query, strategy=strat, k=k, rerank=rerank)

        passages: List[RetrievedPassage] = []
        for r in results:
            passages.append(
                RetrievedPassage(
                    doc_id=r.doc_id,
                    title=r.title,
                    pages=f"{r.page_start}-{r.page_end}",
                    text=r.highlighted_text or (r.content[:700] + ("..." if len(r.content) > 700 else "")),
                )
            )

        prompt = build_prompt(query, passages)
        answer = generate_with_ollama(model=mistral_model, prompt=prompt)

        return {
            "query": query,
            "answer": answer,
            "citations": format_citations(passages),
            "passages": passages,
        }
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG con Mistral (Ollama) + Weaviate")
    parser.add_argument("query", help="Consulta del usuario")
    parser.add_argument("--host", default="localhost", help="Host de Weaviate")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--collection", default="BookChunk")
    parser.add_argument("--model", default="mistral", help="Modelo Ollama (ej. mistral, mistral:7b)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--strategy", choices=["semantic", "hybrid", "multi_stage"], default="multi_stage")
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    args = parser.parse_args()

    out = agentic_rag(
        query=args.query,
        weaviate_host=args.host,
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        collection=args.collection,
        mistral_model=args.model,
        k=args.k,
        strategy=args.strategy,
        rerank=not args.no_rerank,
    )

    print("\n=== Respuesta ===\n")
    print(out["answer"])
    print("\n=== Citas ===\n")
    print(out["citations"])


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Agente RAG con Mistral 7B (transformers) y ImprovedSemanticSearch (Weaviate)

CLI de ejemplo:
  python src/rag/agent/agentic_rag.py "mejores prácticas de autenticación" --k 8 --strategy multi_stage
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Soporte de import relativo al repo
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]  # .../DatosTesis/src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import torch
import weaviate
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from rag.search.improved_search import (
    ImprovedSemanticSearch,
    SearchStrategy,
    SearchResult,
)


def load_mistral(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Carga Mistral con configuración segura para CPU/MPS/GPU."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs = {}
        # device_map auto intenta GPU/MPS si existen; fallback CPU
        kwargs["device_map"] = "auto"
        # dtype seguro si hay soporte, fallback float32
        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
        elif torch.backends.mps.is_available():
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return tokenizer, model
    except Exception as e:
        print(f"Error cargando Mistral: {e}")
        raise


def build_prompt(query: str, results: List[SearchResult]) -> str:
    """Construye prompt de Mistral con pasajes y citas en español."""
    context_blocks = []
    for i, r in enumerate(results, 1):
        snippet = r.content.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        meta = f"docId={r.doc_id} | title={r.title} | pages={r.page_start}-{r.page_end}"
        context_blocks.append(f"[P{i}] {meta}\n{snippet}")

    context = "\n\n".join(context_blocks)

    system_msg = (
        "Eres un asistente experto en ciberseguridad. Responde en español, de forma concisa,"
        " precisa y con fundamento. Usa SOLO la evidencia provista en CONTEXTO."
        " Incluye citas al final entre corchetes con docId:title:pages de los pasajes usados."
        " Si la evidencia es insuficiente, dilo y sugiere una nueva búsqueda."
    )

    user_msg = (
        f"Consulta: {query}\n\n"
        f"CONTEXTO (pasajes recuperados):\n{context}\n\n"
        "Instrucciones de respuesta:\n"
        "- Responde directo a la consulta.\n"
        "- No inventes. Si falta evidencia, sé explícito.\n"
        "- Añade al final una línea 'Fuentes:' con una lista de [docId:title:pages] relevantes.\n"
    )

    # Formato de diálogo Instruct de Mistral
    prompt = (
        f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"
    )
    return prompt


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 384) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extraer solo la parte de la respuesta del modelo tras [/INST]
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG con Mistral 7B y Weaviate",
    )
    parser.add_argument("query", help="Consulta de usuario")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--collection", default="BookChunk")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--strategy",
        choices=["semantic", "hybrid", "multi_stage"],
        default="multi_stage",
    )
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.2")
    args = parser.parse_args()

    # Conectar a Weaviate
    client = weaviate.connect_to_local(
        host=args.host, port=args.http_port, grpc_port=args.grpc_port
    )

    try:
        # Recuperación
        retriever = ImprovedSemanticSearch(client, args.collection)
        strategy = SearchStrategy(args.strategy)
        results = retriever.search(args.query, strategy=strategy, k=args.k, rerank=True)

        # LLM
        tokenizer, model = load_mistral(args.model_id)
        prompt = build_prompt(args.query, results)
        answer = generate_answer(tokenizer, model, prompt)

        print("\n=== Respuesta ===\n")
        print(answer)

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())


