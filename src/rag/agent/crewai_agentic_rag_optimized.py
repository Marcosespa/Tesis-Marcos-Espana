#!/usr/bin/env python3
"""
CrewAI Agentic RAG - Optimized HyDE Architecture

Optimized 3-agent architecture:
1. HyDE Generator - Generates hypothetical answers
2. Retrieval & Response - Uses search_weaviate to find documents and generate final answer
3. Quality Evaluator - Evaluates response quality

Usage:
  python crewai_agentic_rag_optimized.py "What is MFA?"
"""
import os
import argparse
import sys
import json
import time
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional

# PYTHONPATH support
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import weaviate
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# Azure OpenAI (LangChain wrapper)
try:
    # Available in langchain-openai recent versions
    from langchain_openai import AzureChatOpenAI  # type: ignore
except Exception:
    AzureChatOpenAI = None  # type: ignore

# Direct Azure OpenAI client (to avoid litellm issues)
try:
    from openai import AzureOpenAI
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import List, Optional, Any
    AZURE_OPENAI_AVAILABLE = True
except Exception:
    AZURE_OPENAI_AVAILABLE = False
    AzureOpenAI = None

from rag.search.multi_class_search import (
    MultiClassSemanticSearch,
    AgentType,
)


# ============================================================================
# SIGNAL HANDLING AND CLEANUP
# ============================================================================

def cleanup_processes():
    """Clean up all child processes and connections"""
    try:
        print("\n🧹 Cleaning up processes...")
        
        # Close Weaviate connections
        if 'weaviate_client' in globals() and weaviate_client:
            try:
                weaviate_client.close()
                print("  ✅ Weaviate closed")
            except Exception:
                pass
        
        print("  ✅ Connection cleanup completed")
        
    except Exception as e:
        print(f"⚠️ Error in cleanup: {e}")

def signal_handler(signum, frame):
    """Signal handler for Ctrl+C"""
    print(f"\n⚠️ Signal received: {signum}")
    cleanup_processes()
    print("👋 Exiting...")
    sys.exit(130)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# SHARED CONTEXT
# ============================================================================

class RAGContext:
    """Shared context between agents"""
    def __init__(self, weaviate_client):
        self.weaviate_client = weaviate_client
        self.retriever = MultiClassSemanticSearch(weaviate_client)
        self.passages: List[Dict] = []
        self.query: str = ""
        self.iterations: int = 0
        self.original_query: str = ""
        self.current_answer: str = ""
        self.retrieved_passages: List[Dict] = []
    
    def reset(self):
        """Reset context for new query"""
        self.passages = []
        self.query = ""
        self.current_answer = ""
        self.retrieved_passages = []
        self.iterations = 0


# Global instance
rag_context: Optional[RAGContext] = None


# ============================================================================
# TOOLS
# ============================================================================

@tool
def search_weaviate(query: str, k: int = 8, agent_type: str = "general") -> str:
    """Search in Weaviate with specific parameters.
    
    Args:
        query: The search query string
        k: Number of results to retrieve (default: 8)
        agent_type: Agent type for search (general, security, none)
    
    Returns:
        JSON string with search results
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Map agent_type to AgentType
        agent_mapping = {
            "general": AgentType.GENERAL,
            "security": AgentType.SECURITY,
            "none": AgentType.GENERAL,
            "": AgentType.GENERAL
        }
        agent = agent_mapping.get(agent_type.lower(), AgentType.GENERAL)
        
        print(f"  [search_weaviate] Searching '{query}' with agent={agent.value}, k={k}")
        
        # Search with reranking
        results = rag_context.retriever.search(
            query,
            agent=agent,
            k=k,
            rerank=True
        )
        
        # Convert results
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
        rag_context.retrieved_passages = passages
        elapsed = time.time() - start
        
        print(f"  [search_weaviate] {len(passages)} docs in {elapsed:.2f}s")
        
        return json.dumps({
            "success": True,
            "count": len(passages),
            "passages": passages,
            "avg_score": sum(p["score"] for p in passages) / len(passages) if passages else 0
        }, ensure_ascii=False)
        
    except Exception as e:
        print(f"  [search_weaviate] ERROR: {e}")
        return json.dumps({"error": str(e)})


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_hyde_generator(llm) -> Agent:
    """Agent 1: HyDE Generator - Generates hypothetical answers without context"""
    return Agent(
        role="HyDE Generator",
        goal="Generate direct, concise hypothetical answers with neutral professional language",
        backstory=(
            "You are an expert in generating hypothetical answers for HyDE (Hypothetical Document Embeddings). "
            "You write direct, concise responses with neutral professional language. "
            "Avoid verbosity and excessive adjectives. Include key concepts and technical details without over-adjectivization. "
            "You do NOT search for information - you generate based on your knowledge."
        ),
        tools=[],  # No tools - pure generation
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1
    )


def create_retrieval_response_agent(llm) -> Agent:
    """Agent 2: Retrieval & Response - Uses search_weaviate to find documents and generate final response"""
    return Agent(
        role="Retrieval & Response Specialist",
        goal="Generate direct, concise final responses using retrieved documents with neutral professional language",
        backstory=(
            "You are an expert in RAG (Retrieval-Augmented Generation). You receive hypothetical answers "
            "from the HyDE Generator and use them to find the most relevant documents in the knowledge base. "
            "Then you generate direct, concise final answers using only the retrieved documents as evidence. "
            "You use neutral professional language, avoid verbosity, and ground responses accurately."
        ),
        tools=[search_weaviate],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )


def create_quality_evaluator(llm) -> Agent:
    """Agent 3: Quality Evaluator - Evaluates response quality with detailed criteria"""
    return Agent(
        role="Quality Evaluator",
        goal="Evaluate response quality across multiple dimensions and provide detailed feedback",
        backstory=(
            "You are an expert quality evaluator for RAG systems. You assess responses across three key dimensions: "
            "1) RELEVANCE - How well does the answer address the original question? "
            "2) GROUNDEDNESS - How well is the answer supported by the retrieved documents? "
            "3) COMPLETENESS - Is the answer comprehensive and complete? "
            "You provide detailed feedback and make decisions: APPROVE (high quality), REFINE (needs improvement), or REJECT (poor quality)."
        ),
        tools=[],  # No tools - pure evaluation
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_hyde_generation_task(agent: Agent, query: str) -> Task:
    """Task 1: Generate hypothetical answer without context"""
    return Task(
        description=(
            f"IMPORTANT: Respuesta directa, concisa, lenguaje neutral y profesional. "
            f"Sin adjetivación excesiva ni verborragia. Responde SOLO la pregunta.\n\n"
            f"Question: '{query}'\n\n"
            f"Requirements:\n"
            f"- Answer directly and concisely - NO verbosity or excessive adjectives\n"
            f"- Professional, neutral tone - avoid over-adjectivization\n"
            f"- Respond ONLY the question asked - no extra information\n"
            f"- Include key concepts, definitions, and explanations when necessary\n"
            f"- Use professional, technical language\n"
            f"- Be concise (50-200 words)\n"
            f"- Do NOT mention that this is hypothetical\n"
            f"- Write in the same style as technical documentation\n"
            f"- Include examples only when strictly relevant\n\n"
            f"Output: A direct, concise hypothetical answer with neutral professional language."
        ),
        agent=agent,
        expected_output=(
            "A direct, concise hypothetical answer (50-200 words) with neutral professional language, "
            "covering the question without mentioning it's hypothetical."
        )
    )


def create_retrieval_response_task(agent: Agent, query: str, hyde_task: Task, k: int = 5) -> Task:
    """Task 2: Se actualizará con context de la tarea anterior"""
    return Task(
        description=(
            f"IMPORTANT: Respuesta directa, concisa, lenguaje neutral profesional. "
            f"Sin verbosidad ni adjetivación excesiva. Responde SOLO la pregunta.\n\n"
            f"Use the hypothetical answer from the previous HyDE task to guide retrieval.\n\n"
            f"Original Query: '{query}'\n\n"
            f"MANDATORY Process:\n"
            f"1. Review the hypothetical answer from HyDE Generator (available in context)\n"
            f"2. Call 'search_weaviate' function with the query and k=8:\n"
            f"   - Pass the query string directly as the query parameter\n"
            f"   - Set k to 8 (integer)\n"
            f"   - Set agent_type to 'general' (string)\n"
            f"3. Generate the FINAL answer using ONLY the retrieved documents\n\n"
            f"MANDATORY Requirements for the final answer:\n"
            f"- Answer DIRECTLY and CONCISELY - avoid verbosity\n"
            f"- Professional, neutral language - NO excessive adjectives\n"
            f"- Respond ONLY the question - no extra information\n"
            f"- Answer in English, clearly and concisely\n"
            f"- Minimum 100 words\n"
            f"- Do NOT include citations or source references\n"
            f"- Ground ALL claims in the retrieved documents but don't cite them\n"
            f"- If there are practical steps, use numbered lists\n"
            f"- If evidence is insufficient, be explicit about limitations\n"
            f"- Write naturally without academic citations\n\n"
            f"IMPORTANT: This is the FINAL answer. Make it direct, concise, and well-grounded."
        ),
        agent=agent,
        expected_output=(
            "Final direct and concise answer in English with:\n"
            "- Main answer (minimum 100 words)\n"
            "- Neutral, professional tone without excessive adjectives\n"
            "- Clear indication if information is missing\n"
            "- All claims grounded in retrieved documents\n"
            "- No verbosity or unnecessary elaboration"
        ),
        context=[hyde_task]  # ✅ Esto permite acceder al output de hyde_task
    )


def create_quality_evaluation_task(agent: Agent, query: str, retrieval_task: Task, sources: List[Dict]) -> Task:
    """Task 3: Evaluará la respuesta de la tarea anterior"""
    return Task(
        description=(
            f"Evaluate the quality of the generated response across three key dimensions.\n\n"
            f"Original Query: '{query}'\n\n"
            f"The response to evaluate is available from the previous task.\n\n"
            f"Retrieved Sources: {len(sources)} documents\n"
            f"Source IDs: {[s.get('doc_id', 'N/A') for s in sources[:3]]}\n\n"
            f"Evaluation Criteria:\n\n"
            f"1. RELEVANCE (0-10):\n"
            f"   - Does the answer directly address the original question?\n"
            f"   - Is the response focused and on-topic?\n"
            f"   - Does it provide useful information?\n\n"
            f"2. GROUNDEDNESS (0-10):\n"
            f"   - Are claims supported by the retrieved documents?\n"
            f"   - Is there supporting evidence in retrieved docs?\n"
            f"   - Is there evidence for each major claim?\n\n"
            f"3. COMPLETENESS (0-10):\n"
            f"   - Is the answer comprehensive?\n"
            f"   - Does it cover the main aspects of the question?\n"
            f"   - Are there significant gaps?\n\n"
            f"Decision Framework:\n"
            f"- APPROVE: All scores ≥ 7, high quality response\n"
            f"- REFINE: Any score < 7 but > 4, needs improvement\n"
            f"- REJECT: Any score ≤ 4, poor quality\n\n"
            f"CRITICAL: You MUST use this EXACT output format:\n\n"
            f"DECISION: [APPROVE/REFINE/REJECT]\n\n"
            f"RELEVANCE: [Score 0-10] - [Brief one-sentence justification]\n\n"
            f"GROUNDEDNESS: [Score 0-10] - [Brief one-sentence justification]\n\n"
            f"COMPLETENESS: [Score 0-10] - [Brief one-sentence justification]\n\n"
            f"SUMMARY: [One paragraph summarizing the overall evaluation]"
        ),
        agent=agent,
        expected_output=(
            "Evaluation in EXACT format:\n"
            "DECISION: [APPROVE/REFINE/REJECT]\n\n"
            "RELEVANCE: [0-10] - [one sentence]\n\n"
            "GROUNDEDNESS: [0-10] - [one sentence]\n\n"
            "COMPLETENESS: [0-10] - [one sentence]\n\n"
            "SUMMARY: [one paragraph]\n"
        ),
        context=[retrieval_task]  # ✅ Accede al output de retrieval_task
    )


# ============================================================================
# CUSTOM AZURE OPENAI WRAPPER (to avoid litellm issues)
# ============================================================================

class DirectAzureChatOpenAI(BaseChatModel):
    """Wrapper personalizado usando directamente AzureOpenAI de OpenAI para evitar litellm"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    azure_endpoint: str
    api_key: str
    api_version: str
    azure_deployment: str
    temperature: float = 0.3
    max_tokens: int = 1000
    client: Any = None
    
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        azure_deployment: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        # Limpiar endpoint
        azure_endpoint_clean = azure_endpoint.rstrip('/')
        
        # Crear cliente Azure OpenAI directamente (como en test.py)
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint_clean,
            api_key=api_key,
        )
        
        # Llamar al constructor de BaseChatModel con los campos
        super().__init__(
            azure_endpoint=azure_endpoint_clean,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=azure_deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client
        )
    
    @property
    def _llm_type(self) -> str:
        return "azure_openai"
    
    def __str__(self) -> str:
        """Retornar string que litellm pueda entender como modelo Azure"""
        return f"azure/{self.azure_deployment}"
    
    def __repr__(self) -> str:
        """Retornar representación del wrapper"""
        return f"DirectAzureChatOpenAI(deployment={self.azure_deployment}, endpoint={self.azure_endpoint})"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Parámetros identificadores para evitar que litellm lo intercepte"""
        return {
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version
        }
    
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Método invoke que CrewAI espera"""
        result = self._generate(messages, **kwargs)
        return result.generations[0].message
    
    def __call__(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Método __call__ para compatibilidad con CrewAI"""
        return self.invoke(messages, **kwargs)
    
    def predict(self, text: str, **kwargs: Any) -> str:
        """Método predict para compatibilidad con LangChain"""
        msg = HumanMessage(content=text)
        result = self.invoke([msg], **kwargs)
        return result.content if hasattr(result, 'content') else str(result)
    
    def predict_messages(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Método predict_messages para compatibilidad con LangChain"""
        return self.invoke(messages, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convertir mensajes de LangChain a formato OpenAI
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
        
        # Llamar directamente a Azure OpenAI (como en test.py)
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        # Convertir respuesta a formato LangChain
        message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Para async, usar sync por ahora (puede mejorarse después)
        return self._generate(messages, stop, run_manager, **kwargs)


# ============================================================================
# OPTIMIZED RAG SYSTEM
# ============================================================================

class OptimizedCrewAIRAG:
    """Optimized RAG system with 3 specialized agents - Single iteration mode"""
    
    def __init__(
        self,
        weaviate_client,
        model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
        max_iterations: int = 1,  # Always 1 - kept for API compatibility
        llm_provider: str = "ollama",
        openai_api_key: str = None,
        # Azure OpenAI specific (optional)
        azure_endpoint: str = None,
        azure_deployment: str = None,
        azure_api_key: str = None,
        azure_api_version: str = "2024-12-01-preview"
    ):
        global rag_context
        rag_context = RAGContext(weaviate_client)
        
        # Force single iteration - always 1
        self.max_iterations = 1
        
        # Configure LLM according to provider
        if llm_provider.lower() == "openai":
            if ChatOpenAI is None:
                raise ImportError("ChatOpenAI is not available. Install: pip install langchain-openai")
            if not openai_api_key:
                raise ValueError("openai_api_key is required to use OpenAI")
            
            self.llm = ChatOpenAI(
                model=model,
                api_key=openai_api_key,
                temperature=0.3,
                max_tokens=1000
            )
            print(f"🤖 Using OpenAI: {model}")
        elif llm_provider.lower() == "azure_openai":
            # Usar wrapper personalizado que evita litellm usando AzureOpenAI directamente
            if not AZURE_OPENAI_AVAILABLE:
                raise ImportError(
                    "Azure OpenAI no está disponible. Instala: pip install openai langchain-core"
                )
            if not all([azure_endpoint, azure_deployment, azure_api_key]):
                raise ValueError("azure_endpoint, azure_deployment and azure_api_key are required for azure_openai")
            
            # Asegurar que el endpoint no termine en /
            azure_endpoint_clean = azure_endpoint.rstrip('/')
            
            # Crear wrapper personalizado que usa directamente AzureOpenAI (como test.py)
            # Esto evita completamente litellm ya que llama directamente a Azure OpenAI
            
            # Monkey patch para evitar que litellm intercepte nuestro wrapper
            import os
            # Establecer variables que indican que NO usar litellm
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = ""
            os.environ["LITELLM_MASTER_KEY"] = ""
            # Intentar deshabilitar litellm completamente
            try:
                import litellm
                # Guardar referencia a nuestro wrapper que se creará
                wrapper_ref = [None]
                
                # Deshabilitar litellm para nuestro wrapper
                original_completion = litellm.completion
                def patched_completion(*args, **kwargs):
                    # Verificar si el modelo es nuestro wrapper o si es un string que empieza con "azure/"
                    model_param = kwargs.get('model') or (args[0] if args else None)
                    
                    # Si es nuestro wrapper o si es un string azure/ que corresponde a nuestro wrapper
                    if isinstance(model_param, DirectAzureChatOpenAI):
                        wrapper = model_param
                    elif isinstance(model_param, str) and model_param.startswith('azure/') and wrapper_ref[0]:
                        wrapper = wrapper_ref[0]
                    else:
                        return original_completion(*args, **kwargs)
                    
                    # Extraer parámetros del wrapper
                    client = wrapper.client
                    messages = kwargs.get('messages', [])
                    # Llamar directamente a Azure OpenAI (como en test.py)
                    response = client.chat.completions.create(
                        model=wrapper.azure_deployment,
                        messages=messages,
                        temperature=wrapper.temperature,
                        max_tokens=wrapper.max_tokens
                    )
                    # Retornar el objeto response directamente (ya tiene el formato correcto)
                    return response
                
                litellm.completion = patched_completion
                
                # Crear wrapper y guardar referencia
                wrapper_instance = DirectAzureChatOpenAI(
                    azure_endpoint=azure_endpoint_clean,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                    temperature=0.3,
                    max_tokens=1000
                )
                wrapper_ref[0] = wrapper_instance
                self.llm = wrapper_instance
            except Exception as e:
                # Si falla el monkey patch, crear wrapper sin monkey patch
                print(f"⚠️ Warning: No se pudo hacer monkey patch de litellm: {e}")
                self.llm = DirectAzureChatOpenAI(
                    azure_endpoint=azure_endpoint_clean,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                    temperature=0.3,
                    max_tokens=1000
                )
            
            print(f"🤖 Using Azure OpenAI (Direct - bypasses litellm): deployment={azure_deployment} (api_version={azure_api_version})")
            print(f"   Endpoint: {azure_endpoint_clean}")
            print(f"   API Key (last 8): {azure_api_key[-8:]}")
            
        else:  # ollama by default
            self.llm = ChatOllama(
                model=f"ollama/{model}",
                base_url=ollama_host,
                temperature=0.3,
                num_ctx=8192,  # Aumentar contexto para Qwen
                request_timeout=900  # 15 minutos timeout para modelos lentos
            )
            print(f"🤖 Using Ollama: {model} (text mode, timeout=900s, context=8192)")
        
        # Create optimized agents
        print("🔍 Using Optimized HyDE Architecture (3 Specialized Agents)")
        self.agents = {
            "hyde_generator": create_hyde_generator(self.llm),
            "retrieval_response": create_retrieval_response_agent(self.llm),
            "quality_evaluator": create_quality_evaluator(self.llm)
        }
    
    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Process a query with the optimized 3-agent system"""
        
        print(f"\n{'='*80}")
        print("🚀 OPTIMIZED CREWAI RAG - 3 AGENT ARCHITECTURE")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print()
        
        # Reset context for new query
        rag_context.reset()
        rag_context.original_query = query
        current_query = query
        
        try:
            # Single iteration mode - always execute once
            rag_context.iterations = 1
            k_value = 5  # Optimizado para velocidad
            
            print(f"\n{'─'*80}")
            print("🔄 SINGLE ITERATION MODE")
            print(f"{'─'*80}\n")
            print(f"📊 Using k={k_value} for retrieval")
            
            # Create all tasks
            print("📝 Creating tasks...")
            hyde_task = create_hyde_generation_task(
                self.agents["hyde_generator"],
                current_query
            )
            
            retrieval_task = create_retrieval_response_task(
                self.agents["retrieval_response"],
                current_query,
                hyde_task,  # ✅ Pass HyDE task for context
                k=k_value
            )
            
            quality_task = create_quality_evaluation_task(
                self.agents["quality_evaluator"],
                current_query,
                retrieval_task,  # ✅ Pass Retrieval task for context
                rag_context.retrieved_passages or []
            )
            
            # Ejecutar todo en un solo crew secuencial
            print("\n🎬 Executing all tasks sequentially...")
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[hyde_task, retrieval_task, quality_task],
                process=Process.sequential,
                verbose=verbose
            )
            
            result = crew.kickoff()
            
            # El resultado será una lista con outputs de cada tarea
            try:
                if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 3:
                    # Acceder al contenido raw de cada task output
                    hyde_output = result.tasks_output[0].raw
                    retrieval_output = result.tasks_output[1].raw
                    quality_output = result.tasks_output[2].raw
                    
                    final_answer = retrieval_output
                    quality_evaluation = quality_output
                    
                    print(f"\n✅ Successfully parsed {len(result.tasks_output)} task outputs")
                    
                else:
                    # Fallback: intentar como string directo
                    print(f"\n⚠️ Using fallback parsing (tasks_output not available)")
                    final_answer = str(result)
                    quality_evaluation = "No structured evaluation available"
                    
            except AttributeError as e:
                print(f"⚠️ AttributeError accessing task outputs: {e}")
                final_answer = str(result)
                quality_evaluation = "Error accessing task outputs"
            except Exception as e:
                print(f"⚠️ Error parsing results: {e}")
                final_answer = str(result)
                quality_evaluation = "Error occurred - no evaluation available"
            
            return {
                "query": rag_context.original_query,
                "final_query": current_query,
                "answer": final_answer,
                "quality_evaluation": quality_evaluation,
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            return {
                "query": rag_context.original_query,
                "answer": "Process interrupted by user",
                "quality_evaluation": "Process interrupted - no evaluation available",
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
        except Exception as e:
            print(f"\n❌ Error in processing: {e}")
            return {
                "query": rag_context.original_query,
                "answer": f"Error: {str(e)}",
                "quality_evaluation": "Error occurred - no evaluation available",
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }


# ============================================================================
# UTILITIES
# ============================================================================

def format_output(result: Dict[str, Any]) -> str:
    """Format output for CLI"""
    lines = [
        "\n" + "="*80,
        " FINAL RESULT - OPTIMIZED CREWAI RAG",
        "="*80,
        f"\nOriginal Query: {result.get('query', '')}",
        f"Final Query: {result.get('final_query', '')}",
        f"Iterations: {result.get('iterations', 0)}",
        f"Agents Used: {', '.join(result.get('agents_used', []))}",
        f"Sources Retrieved: {len(result.get('passages', []))} documents",
        f"\n{'-'*80}",
        "\n📄 ANSWER:\n",
        "-"*80,
        result.get('answer', ''),
        f"\n{'-'*80}",
        "\n📊 METRICS:\n",
        "-"*80,
        f"• Documents retrieved: {len(result.get('passages', []))}",
        f"• Average relevance score: {sum(p['score'] for p in result.get('passages', [])) / len(result.get('passages', [])) if result.get('passages') else 0:.3f}",
        f"• Processing iterations: {result.get('iterations', 0)}",
        "="*80 + "\n"
    ]
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global weaviate_client
    
    parser = argparse.ArgumentParser(
        description="Optimized CrewAI RAG with 3 Specialized Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("query", help="User query")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="CrewAI verbose mode")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    
    # Arguments for OpenAI
    parser.add_argument("--llm-provider", choices=["ollama", "openai", "azure_openai"], default="ollama", 
                       help="LLM provider: ollama, openai or azure_openai")
    parser.add_argument("--openai-api-key", help="OpenAI API Key (required if --llm-provider=openai)")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo", 
                       help="OpenAI model (e.g.: gpt-3.5-turbo, gpt-4, gpt-4-turbo)")
    
    # Azure OpenAI args
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint URL")
    parser.add_argument("--azure-deployment", default="gpt-4.1", help="Azure OpenAI deployment name (default: gpt-4.1)")
    parser.add_argument("--azure-api-key", default=None, help="Azure OpenAI API key")
    parser.add_argument("--azure-api-version", default="2024-12-01-preview", help="Azure OpenAI API version")
    
    args = parser.parse_args()
    
    weaviate_client = None
    
    print("=" * 80)
    print("🚀 OPTIMIZED CREWAI RAG - 3 AGENT ARCHITECTURE")
    print("=" * 80)
    print("💡 Press Ctrl+C at any time to exit cleanly")
    print("🧹 The system will automatically clean up all processes")
    print("=" * 80)
    
    try:
        # Connect to Weaviate
        print("🔌 Connecting to Weaviate...")
        weaviate_client = weaviate.connect_to_local(
            host=args.host,
            port=args.http_port,
            grpc_port=args.grpc_port
        )
        
        # Determine model according to provider
        if args.llm_provider == "openai":
            model_name = args.openai_model
            if not args.openai_api_key:
                print("❌ ERROR: --openai-api-key is required to use OpenAI")
                return 1
        elif args.llm_provider == "azure_openai":
            model_name = args.azure_deployment or ""
            azure_api_key = args.azure_api_key
            if not (args.azure_endpoint and args.azure_deployment and azure_api_key):
                print("ERROR: --azure-endpoint, --azure-deployment y --azure-api-key son requeridos")
                return 1
        else:
            model_name = args.model
        
        # Initialize Optimized RAG system
        print("🚀 Initializing Optimized RAG system...")
        rag_system = OptimizedCrewAIRAG(
            weaviate_client=weaviate_client,
            model=model_name,
            ollama_host=args.ollama_host,
            max_iterations=args.max_iterations,
            llm_provider=args.llm_provider,
            openai_api_key=args.openai_api_key,
            azure_endpoint=args.azure_endpoint,
            azure_deployment=args.azure_deployment,
            azure_api_key=azure_api_key,
            azure_api_version=args.azure_api_version
        )
        
        # Process query
        print("🎯 Processing query...")
        result = rag_system.process_query(args.query, verbose=args.verbose)
        
        # Show result
        output = format_output(result)
        print(output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        cleanup_processes()


if __name__ == "__main__":
    sys.exit(main())
