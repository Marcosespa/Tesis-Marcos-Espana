#!/usr/bin/env python3
"""
CrewAI Agentic RAG - Sistema Multi-Agente General

Optimizaciones críticas:
1. Tools SIN lógica LLM innecesaria
2. max_iter=1 en TODOS los agentes (evita reflexión)
3. Tareas con contexto pre-construido (menos procesamiento)
4. Sin hardcoded queries
5. Limpieza segura

Uso:
  python crewai_agentic_rag.py "¿Qué es inteligencia artificial?"
"""

import argparse
import sys
import json
import time
import signal
import os
import atexit
from pathlib import Path
from typing import List, Dict, Any, Optional

# PYTHONPATH support
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ============================================================================
# MANEJO DE SEÑALES Y LIMPIEZA
# ============================================================================

def cleanup_processes():
    """Limpia todos los procesos hijos y conexiones"""
    try:
        print("\n🧹 Limpiando procesos...")
        
        # Cerrar conexiones Weaviate
        if 'weaviate_client' in globals() and weaviate_client:
            try:
                weaviate_client.close()
                print("  ✅ Weaviate cerrado")
            except:
                pass
        
        # NO terminar procesos automáticamente - solo cerrar conexiones
        print("  ✅ Limpieza de conexiones completada")
        
    except Exception as e:
        print(f"⚠️ Error en limpieza: {e}")

def signal_handler(signum, frame):
    """Manejador de señales para Ctrl+C"""
    print(f"\n⚠️ Señal recibida: {signum}")
    cleanup_processes()
    print("👋 Saliendo...")
    sys.exit(130)

# Registrar manejadores de señales - Solo Ctrl+C
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
# NO registrar SIGTERM para evitar terminaciones automáticas

import weaviate
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from rag.search.multi_class_search import (
    MultiClassSemanticSearch,
    AgentType,
)


# ============================================================================
# CONTEXTO COMPARTIDO OPTIMIZADO
# ============================================================================

class RAGContext:
    """Contexto compartido entre agentes - Optimizado"""
    def __init__(self, weaviate_client):
        self.weaviate_client = weaviate_client
        self.retriever = MultiClassSemanticSearch(weaviate_client)
        self.passages: List[Dict] = []
        self.query: str = ""
        self.iterations: int = 0
        # Atributos adicionales que faltaban
        self.original_query: str = ""
        self.current_answer: str = ""
        self.retrieved_passages: List[Dict] = []


# Instancia global (se inicializa en main)
rag_context: Optional[RAGContext] = None


# ============================================================================
# HERRAMIENTAS OPTIMIZADAS (Sin LLM interno)
# ============================================================================

@tool("expand_query")
def expand_query(query: str) -> str:
    """
    Expande una query con términos relacionados para mejorar la búsqueda.
    """
    try:
        # Generar variaciones de la query
        variations = [
            query,
            f"{query} definición concepto",
            f"{query} ejemplos casos uso",
            f"{query} implementación práctica",
            f"{query} mejores prácticas"
        ]
        
        # Remover duplicados y limitar
        unique_variations = list(dict.fromkeys(variations))[:5]
        
        print(f"  [expand_query] Generadas {len(unique_variations)} variaciones")
        
        return json.dumps({
            "success": True,
            "variations": unique_variations,
            "count": len(unique_variations)
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("search_weaviate")
def search_weaviate(query: str, k: int = 8, agent_type: str = "general") -> str:
    """
    Búsqueda en Weaviate con parámetros específicos.
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Mapear agent_type a AgentType - Solo usar tipos válidos
        agent_mapping = {
            "general": AgentType.GENERAL,
            "security": AgentType.SECURITY,
            "none": AgentType.GENERAL,
            "": AgentType.GENERAL
        }
        agent = agent_mapping.get(agent_type.lower(), AgentType.GENERAL)
        
        print(f"  [search_weaviate] Buscando '{query}' con agent={agent.value}, k={k}")
        
        # Búsqueda con reranking
        results = rag_context.retriever.search(
            query,
            agent=agent,
            k=k,
            rerank=True
        )
        
        # Convertir resultados
        passages = []
        for r in results[:k]:
            passages.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "pages": f"{r.page_start}-{r.page_end}",
                "text": r.content[:500],
                "score": r.final_score
            })
        
        rag_context.passages = passages
        rag_context.retrieved_passages = passages  # Guardar también aquí
        elapsed = time.time() - start
        
        print(f"  [search_weaviate] {len(passages)} docs en {elapsed:.2f}s")
        
        return json.dumps({
            "success": True,
            "count": len(passages),
            "passages": passages,
            "avg_score": sum(p["score"] for p in passages) / len(passages) if passages else 0
        }, ensure_ascii=False)
        
    except Exception as e:
        print(f"  [search_weaviate] ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool("smart_search")
def smart_search(query: str, k: int = 8) -> str:
    """
    Búsqueda directa en Weaviate sin procesamiento extra.
    
    OPTIMIZACIÓN: Sin reformulación, directo a la búsqueda.
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Búsqueda directa con reranking
        results = rag_context.retriever.search(
            query,
            agent=AgentType.GENERAL,
            k=k,
            rerank=True
        )
        
        # Convertir resultados
        passages = []
        for r in results[:8]:  # Limitar a top-8
            passages.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "pages": f"{r.page_start}-{r.page_end}",
                "text": r.content[:500],  # Texto reducido
                "score": r.final_score
            })
        
        rag_context.passages = passages
        elapsed = time.time() - start
        
        print(f"  [smart_search] {len(passages)} docs en {elapsed:.2f}s")
        
        return json.dumps({
            "success": True,
            "count": len(passages),
            "avg_score": sum(p["score"] for p in passages) / len(passages) if passages else 0
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})




# ============================================================================
# DEFINICIÓN DE AGENTES
# ============================================================================

def create_query_reformulator(llm) -> Agent:
    """Agente que mejora queries de forma simple"""
    return Agent(
        role="Query Enhancement Specialist",
        goal="Mejorar queries de usuario para optimizar búsquedas",
        backstory=(
            "Eres un experto en optimización de queries para cualquier tema. "
            "Tu trabajo es tomar la query del usuario y generar variaciones útiles "
            "para búsqueda. Siempre produces resultados rápidamente sin loops innecesarios."
        ),
        tools=[expand_query],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3  # Limitar iteraciones para evitar loops
    )


def create_answer_generator_with_search(llm) -> Agent:
    """Agente que busca documentos Y genera respuestas"""
    return Agent(
        role="Search & Answer Specialist",
        goal="Buscar documentos relevantes y generar respuestas precisas",
        backstory=(
            "Eres un asistente experto que combina búsqueda inteligente "
            "con síntesis de información. Primero buscas los mejores documentos "
            "usando estrategias adaptativas, luego generas respuestas claras y "
            "bien fundamentadas sobre cualquier tema. SIEMPRE citas tus fuentes con [n]."
        ),
        tools=[search_weaviate],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3  # Limitar iteraciones para evitar loops
    )






def create_quality_controller(llm) -> Agent:
    """Agente supervisor simplificado"""
    return Agent(
        role="Quality Control Manager",
        goal="Evaluar la calidad de respuestas y decidir si aprobar o refinar",
        backstory=(
            "Eres un evaluador de calidad especializado en sistemas RAG. "
            "Analizas respuestas generadas sobre cualquier tema y decides si son "
            "suficientes o necesitan refinamiento. Eres directo y eficiente en tus decisiones."
        ),
        tools=[],  # Sin herramientas de delegación
        llm=llm,
        verbose=True,
        allow_delegation=False,  # Desactivar delegación TODO
        max_iter=1 # Limitar iteraciones
    )


# ============================================================================
# DEFINICIÓN DE TAREAS
# ============================================================================

def create_reformulation_task(agent: Agent, query: str) -> Task:
    """Tarea simple de reformulación de query"""
    return Task(
        description=(
            f"Mejora esta consulta para optimizar la búsqueda:\n\n"
            f"Query original: '{query}'\n\n"
            f"Usa la herramienta 'expand_query' UNA SOLA VEZ para generar variaciones útiles.\n"
            f"No repitas la misma entrada. Genera variaciones diferentes si es necesario.\n\n"
            f"Formato de salida: Lista de queries optimizadas."
        ),
        agent=agent,
        expected_output=(
            "Lista de 3-5 queries optimizadas para búsqueda, incluyendo la original."
        )
    )


def create_search_and_generate_task(agent: Agent, query: str) -> Task:
    """Tarea combinada: buscar documentos y generar respuesta"""
    return Task(
        description=(
            f"Busca documentos relevantes y genera una respuesta completa para:\n\n"
            f"Query: '{query}'\n\n"
            f"Proceso OBLIGATORIO:\n"
            f"1. Usa 'search_weaviate' con estos parámetros EXACTOS:\n"
            f"   - query: '{query}'\n"
            f"   - k: 8\n"
            f"   - agent_type: 'general'\n"
            f"2. Analiza los documentos recuperados\n"
            f"3. Genera una respuesta completa y detallada\n\n"
            f"Requisitos OBLIGATORIOS de la respuesta:\n"
            f"- Responde en español, de forma clara y concisa\n"
            f"- Mínimo 200 palabras\n"
            f"- SIEMPRE cita fuentes con [n] al final de cada afirmación\n"
            f"- Si hay pasos prácticos, usa listas numeradas\n"
            f"- Si la evidencia es insuficiente, sé explícito\n"
            f"- Al final incluye sección 'Fuentes:' con [docId:title:páginas]\n\n"
            f"IMPORTANTE: NO solo digas 'APPROVE' - DEBES generar una respuesta completa.\n"
            f"NO inventes información. Solo usa evidencia de los documentos."
        ),
        agent=agent,
        expected_output=(
            "Respuesta completa en español con:\n"
            "- Respuesta principal (mínimo 200 palabras)\n"
            "- Citas inline con [n]\n"
            "- Sección 'Fuentes:' al final\n"
            "- Indicación clara si falta información"
        )
    )




def create_quality_control_task(agent: Agent, query: str) -> Task:
    """Tarea de control de calidad simplificado"""
    return Task(
        description=(
            f"Evalúa la calidad de la respuesta generada:\n\n"
            f"Query original: '{query}'\n"
            f"Iteración actual: {rag_context.iterations}\n\n"
            f"Criterios de evaluación:\n"
            f"1. ¿La respuesta aborda completamente la query?\n"
            f"2. ¿Hay suficientes fuentes citadas? (mínimo 2)\n"
            f"3. ¿La respuesta es clara y útil?\n\n"
            f"Decisión:\n"
            f"- Si todo OK → APPROVE (respuesta final)\n"
            f"- Si falta info y iteraciones<2 → REFINE (nueva búsqueda)\n"
            f"- Si iteraciones≥2 → APPROVE (mejor esfuerzo)\n\n"
            f"Responde solo con: APPROVE o REFINE"
        ),
        agent=agent,
        expected_output=(
            "Decisión simple: APPROVE o REFINE"
        )
    )


# ============================================================================
# SISTEMA RAG CON CREWAI
# ============================================================================

class CrewAIAgenticRAG:
    """Sistema RAG orquestado por CrewAI"""
    
    def __init__(
        self,
        weaviate_client,
        model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
        max_iterations: int = 2
    ):
        global rag_context
        rag_context = RAGContext(weaviate_client)
        
        self.max_iterations = max_iterations
        
        # Configurar LLM (Ollama via LangChain) - Formato correcto para LiteLLM
        self.llm = ChatOllama(
            model=f"ollama/{model}",  # Formato que espera LiteLLM
            base_url=ollama_host,
            temperature=0.3,
            format="json"  # Para respuestas estructuradas
        )
        
        # Crear agentes simplificados (solo 3 esenciales)
        self.agents = {
            "reformulator": create_query_reformulator(self.llm),
            "generator": create_answer_generator_with_search(self.llm),
            "controller": create_quality_controller(self.llm)
        }
    
    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Procesa una query con el sistema multi-agente"""
        
        print(f"\n{'='*80}")
        print(f"🚀 INICIANDO CREWAI AGENTIC RAG")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        rag_context.original_query = query
        current_query = query
        
        try:
            for iteration in range(self.max_iterations):
                rag_context.iterations = iteration + 1
                
                print(f"\n{'─'*80}")
                print(f"🔄 ITERACIÓN {iteration + 1}/{self.max_iterations}")
                print(f"{'─'*80}\n")
                
                # Fase 1: Reformulación
                print("📝 Fase 1: Reformulación de Query")
                reformulation_task = create_reformulation_task(
                    self.agents["reformulator"],
                    current_query
                )
                
                # Fase 2: Búsqueda y Generación (combinada)
                print("\n🔍✍️ Fase 2: Búsqueda y Generación de Respuesta")
                search_generate_task = create_search_and_generate_task(
                    self.agents["generator"],
                    current_query
                )
                
                # Fase 3: Control de Calidad
                print("\n🎯 Fase 3: Control de Calidad")
                quality_task = create_quality_control_task(
                    self.agents["controller"],
                    current_query
                )
                
                # Crear crew simplificado para esta iteración
                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=[
                        reformulation_task,
                        search_generate_task,
                        quality_task
                    ],
                    process=Process.sequential,  # Ejecutar en orden
                    verbose=verbose
                )
                
                # Ejecutar crew
                print("\n🎬 Ejecutando Crew...")
                result = crew.kickoff()
                
                # Parsear decisión del controlador
                try:
                    # El último task debe tener la decisión
                    quality_result = result
                    if "APPROVE" in str(quality_result).upper():
                        print("\n✅ Respuesta APROBADA por Quality Controller")
                        break
                    elif "REFINE" in str(quality_result).upper():
                        print("\n🔄 Quality Controller solicita REFINAMIENTO")
                        # Extraer nueva query sugerida
                        # (En producción parsearías el JSON del quality_task)
                        current_query = f"{query} ejemplos casos uso implementación"
                        continue
                except Exception:
                    # Si no se puede parsear, aprobar por defecto
                    break
            
            # Resultado final
            return {
                "query": rag_context.original_query,
                "final_query": current_query,
                "answer": rag_context.current_answer or str(result),
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido por usuario")
            return {
                "query": rag_context.original_query,
                "answer": "Proceso interrumpido por usuario",
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
        except Exception as e:
            print(f"\n❌ Error en procesamiento: {e}")
            return {
                "query": rag_context.original_query,
                "answer": f"Error: {str(e)}",
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }


# ============================================================================
# UTILIDADES
# ============================================================================

def format_crewai_output(result: Dict[str, Any]) -> str:
    """Formatea output de CrewAI para CLI"""
    lines = [
        "\n" + "="*80,
        " RESULTADO FINAL - CREWAI AGENTIC RAG",
        "="*80,
        f"\nQuery Original: {result['query']}",
        f"Query Final: {result['final_query']}",
        f"Iteraciones: {result['iterations']}",
        f"Agentes Utilizados: {', '.join(result['agents_used'])}",
        f"\n{'-'*80}",
        "\n📄 RESPUESTA:\n",
        "-"*80,
        result['answer'],
        f"\n{'-'*80}",
        "\n📚 FUENTES:\n",
        "-"*80,
    ]
    
    for i, p in enumerate(result['passages'], 1):
        lines.append(
            f"[{i}] {p['title']} ({p['doc_id']}) "
            f"páginas {p['pages']} - Score: {p['score']:.3f}"
        )
    
    lines.append("="*80 + "\n")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global weaviate_client
    
    parser = argparse.ArgumentParser(
        description="Agentic RAG con CrewAI - Sistema Multi-Agente Simplificado",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("query", help="Consulta del usuario")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="Modo verbose de CrewAI")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    
    args = parser.parse_args()
    
    weaviate_client = None
    
    print("=" * 80)
    print("🚀 CREWAI AGENTIC RAG - SISTEMA MULTI-AGENTE")
    print("=" * 80)
    print("💡 Presiona Ctrl+C en cualquier momento para salir limpiamente")
    print("🧹 El sistema limpiará automáticamente todos los procesos")
    print("=" * 80)
    
    try:
        # Conectar a Weaviate
        print("🔌 Conectando a Weaviate...")
        weaviate_client = weaviate.connect_to_local(
            host=args.host,
            port=args.http_port,
            grpc_port=args.grpc_port
        )
        
        # Inicializar sistema CrewAI RAG
        print("🚀 Inicializando sistema RAG...")
        rag_system = CrewAIAgenticRAG(
            weaviate_client=weaviate_client,
            model=args.model,
            ollama_host=args.ollama_host,
            max_iterations=args.max_iterations
        )
        
        # Procesar query
        print("🎯 Procesando consulta...")
        result = rag_system.process_query(args.query, verbose=args.verbose)
        
        # Mostrar resultado
        output = format_crewai_output(result)
        print(output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por usuario (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Usar la función de limpieza centralizada
        cleanup_processes()


if __name__ == "__main__":
    sys.exit(main())
