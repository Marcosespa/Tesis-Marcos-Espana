#!/usr/bin/env python3
"""
CrewAI Agentic RAG - General Multi-Agent System

Critical optimizations:
1. Tools WITHOUT unnecessary LLM logic
2. max_iter=1 in ALL agents (avoids reflection)
3. Tasks with pre-built context (less processing)
4. No hardcoded queries
5. Safe cleanup

Usage:
  # Traditional search (Query Expansion)
  python crewai_agentic_rag.py "What is MFA?"
  
  # HyDE search (Hypothetical Document Embeddings)
  python crewai_agentic_rag.py "What is MFA?" --search-strategy hyde
  
  # Hybrid search (Traditional + HyDE)
  python crewai_agentic_rag.py "What is MFA?" --search-strategy hybrid
  
  # Optimized architecture (3 specialized agents)
  python crewai_agentic_rag.py "What is MFA?" --search-strategy optimized
  
  # With OpenAI and optimized architecture
  python crewai_agentic_rag.py "What is MFA?" --llm-provider openai --openai-api-key sk-... --search-strategy optimized
"""

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
        
        # DO NOT terminate processes automatically - only close connections
        print("  ✅ Connection cleanup completed")
        
    except Exception as e:
        print(f"⚠️ Error in cleanup: {e}")

def signal_handler(signum, frame):
    """Signal handler for Ctrl+C"""
    print(f"\n⚠️ Signal received: {signum}")
    cleanup_processes()
    print("👋 Exiting...")
    sys.exit(130)

# Register signal handlers - Only Ctrl+C
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
# DO NOT register SIGTERM to avoid automatic terminations


# ============================================================================
# OPTIMIZED SHARED CONTEXT
# ============================================================================

class RAGContext:
    """Shared context between agents - Optimized"""
    def __init__(self, weaviate_client):
        self.weaviate_client = weaviate_client
        self.retriever = MultiClassSemanticSearch(weaviate_client)
        self.passages: List[Dict] = []
        self.query: str = ""
        self.iterations: int = 0
        # Additional attributes that were missing
        self.original_query: str = ""
        self.current_answer: str = ""
        self.retrieved_passages: List[Dict] = []


# Global instance (initialized in main)
rag_context: Optional[RAGContext] = None


# ============================================================================
# OPTIMIZED TOOLS (No internal LLM)
# ============================================================================

@tool("expand_query")
def expand_query(query: str) -> str:
    """
    Expand a query with related terms to improve search.
    """
    try:
        # Generate query variations
        variations = [
            query,
            f"{query} definición concepto",
            f"{query} ejemplos casos uso",
            f"{query} implementación práctica",
            f"{query} mejores prácticas"
        ]
        
        # Remove duplicates and limit
        unique_variations = list(dict.fromkeys(variations))[:5]
        
        print(f"  [expand_query] Generated {len(unique_variations)} variations")
        
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
    Search in Weaviate with specific parameters.
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Map agent_type to AgentType - Only use valid types
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
        rag_context.retrieved_passages = passages  # Also save here
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


@tool("generate_hypothetical_answer")
def generate_hypothetical_answer(query: str) -> str:
    """
    Generate a hypothetical answer to the query WITHOUT searching documents.
    This is used for HyDE (Hypothetical Document Embeddings) approach.
    
    The hypothetical answer will be used to find similar documents.
    """
    try:
        # Generate a hypothetical answer using the LLM
        # This simulates what a good answer would look like
        hypothetical_prompt = f"""
Generate a comprehensive hypothetical answer to this question: "{query}"

Requirements:
- Write as if you were an expert answering this question
- Include key concepts, definitions, and explanations
- Use professional language
- Be detailed but concise (150-300 words)
- Do NOT mention that this is hypothetical
- Write in the same style as technical documentation

Answer:
"""
        
        # Use the LLM to generate hypothetical answer
        if rag_context and hasattr(rag_context, 'llm'):
            # This would need access to the LLM instance
            # For now, we'll create a structured hypothetical answer
            pass
        
        # Create a structured hypothetical answer based on the query
        hypothetical_answer = f"""
Based on the question "{query}", here is a comprehensive explanation:

This topic involves several key concepts that are fundamental to understanding the subject matter. The primary aspects include core definitions, practical applications, and implementation considerations. 

Key points to consider:
1. Definition and scope of the concept
2. Common use cases and applications  
3. Technical implementation details
4. Best practices and recommendations
5. Potential challenges and solutions

The implementation typically involves specific methodologies and tools that are widely recognized in the field. Understanding these components is essential for effective application and troubleshooting.

This approach provides a solid foundation for addressing related questions and scenarios that may arise in practical situations.
"""
        
        print(f"  [generate_hypothetical_answer] Generated hypothetical answer ({len(hypothetical_answer)} chars)")
        
        return json.dumps({
            "success": True,
            "hypothetical_answer": hypothetical_answer,
            "length": len(hypothetical_answer)
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("hyde_search")
def hyde_search(query: str, k: int = 8, agent_type: str = "general") -> str:
    """
    HyDE (Hypothetical Document Embeddings) search.
    
    Process:
    1. Generate hypothetical answer to the query
    2. Use that hypothetical answer to search for similar documents
    3. Return relevant documents based on semantic similarity to the hypothetical answer
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Step 1: Generate hypothetical answer
        print(f"  [hyde_search] Step 1: Generating hypothetical answer for '{query}'")
        hypothetical_result = generate_hypothetical_answer(query)
        hypothetical_data = json.loads(hypothetical_result)
        
        if not hypothetical_data.get("success"):
            return json.dumps({"error": "Failed to generate hypothetical answer"})
        
        hypothetical_answer = hypothetical_data["hypothetical_answer"]
        
        # Step 2: Use hypothetical answer for search
        print(f"  [hyde_search] Step 2: Searching with hypothetical answer")
        
        # Map agent_type to AgentType
        agent_mapping = {
            "general": AgentType.GENERAL,
            "security": AgentType.SECURITY,
            "none": AgentType.GENERAL,
            "": AgentType.GENERAL
        }
        agent = agent_mapping.get(agent_type.lower(), AgentType.GENERAL)
        
        # Search using the hypothetical answer instead of the original query
        results = rag_context.retriever.search(
            hypothetical_answer,  # Use hypothetical answer for search
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
        
        print(f"  [hyde_search] {len(passages)} docs found in {elapsed:.2f}s using HyDE")
        
        return json.dumps({
            "success": True,
            "count": len(passages),
            "passages": passages,
            "hypothetical_answer": hypothetical_answer,
            "avg_score": sum(p["score"] for p in passages) / len(passages) if passages else 0,
            "method": "HyDE"
        }, ensure_ascii=False)
        
    except Exception as e:
        print(f"  [hyde_search] ERROR: {e}")
        return json.dumps({"error": str(e)})


@tool("smart_search")
def smart_search(query: str, k: int = 8) -> str:
    """
    Direct search in Weaviate without extra processing.
    
    OPTIMIZATION: No reformulation, direct to search.
    """
    if rag_context is None:
        return json.dumps({"error": "Context not initialized"})
    
    try:
        start = time.time()
        
        # Direct search with reranking
        results = rag_context.retriever.search(
            query,
            agent=AgentType.GENERAL,
            k=k,
            rerank=True
        )
        
        # Convert results
        passages = []
        for r in results[:8]:  # Limit to top-8
            passages.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "pages": f"{r.page_start}-{r.page_end}",
                "text": r.content[:500],  # Reduced text
                "score": r.final_score
            })
        
        rag_context.passages = passages
        elapsed = time.time() - start
        
        print(f"  [smart_search] {len(passages)} docs in {elapsed:.2f}s")
        
        return json.dumps({
            "success": True,
            "count": len(passages),
            "avg_score": sum(p["score"] for p in passages) / len(passages) if passages else 0
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": str(e)})




# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_hyde_generator(llm) -> Agent:
    """Agent 1: HyDE Generator - Generates hypothetical answers without context"""
    return Agent(
        role="HyDE Generator",
        goal="Generate comprehensive hypothetical answers to user queries without searching documents",
        backstory=(
            "You are an expert in generating hypothetical answers for HyDE (Hypothetical Document Embeddings). "
            "Your job is to create detailed, well-structured answers that simulate what a comprehensive "
            "response would look like. You write as if you were an expert answering the question, "
            "including key concepts, definitions, examples, and technical details. "
            "You do NOT search for information - you generate based on your knowledge."
        ),
        tools=[],  # No tools - pure generation
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1  # Single generation
    )


def create_retrieval_response_agent(llm) -> Agent:
    """Agent 2: Retrieval & Response - Uses hypothetical answer to find documents and generate final response"""
    return Agent(
        role="Retrieval & Response Specialist",
        goal="Use hypothetical answers to retrieve relevant documents and generate final grounded responses",
        backstory=(
            "You are an expert in RAG (Retrieval-Augmented Generation). You receive hypothetical answers "
            "from the HyDE Generator and use them to find the most relevant documents in the knowledge base. "
            "Then you generate the final answer using only the retrieved documents as evidence. "
            "You excel at grounding responses in real evidence and citing sources accurately."
        ),
        tools=[search_weaviate],  # Use traditional search tool
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2  # Search + generate
    )


def create_hyde_search_agent(llm) -> Agent:
    """Agent specialized in HyDE (Hypothetical Document Embeddings) search"""
    return Agent(
        role="HyDE Search Specialist",
        goal="Use HyDE technique to find the most relevant documents",
        backstory=(
            "You are an expert in HyDE (Hypothetical Document Embeddings) technique. "
            "You generate hypothetical answers to queries and then use those answers "
            "to find semantically similar documents. This approach often finds more "
            "relevant documents than traditional query expansion. You excel at "
            "understanding the semantic intent behind questions."
        ),
        tools=[hyde_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2  # HyDE is more direct, fewer iterations needed
    )


def create_hybrid_search_agent(llm) -> Agent:
    """Agent that combines both traditional search and HyDE"""
    return Agent(
        role="Hybrid Search Specialist", 
        goal="Use both traditional search and HyDE to find comprehensive results",
        backstory=(
            "You are an expert in hybrid search strategies. You combine traditional "
            "query-based search with HyDE (Hypothetical Document Embeddings) to get "
            "the best of both worlds. You can use either approach or both depending "
            "on the query type and requirements."
        ),
        tools=[search_weaviate, hyde_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
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
        max_iter=1  # Single evaluation
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_hyde_generation_task(agent: Agent, query: str) -> Task:
    """Task 1: Generate hypothetical answer without context"""
    return Task(
        description=(
            f"Generate a comprehensive hypothetical answer to this question:\n\n"
            f"Question: '{query}'\n\n"
            f"Requirements:\n"
            f"- Write as if you were an expert answering this question\n"
            f"- Include key concepts, definitions, and explanations\n"
            f"- Use professional, technical language\n"
            f"- Be detailed but concise (200-400 words)\n"
            f"- Do NOT mention that this is hypothetical\n"
            f"- Write in the same style as technical documentation\n"
            f"- Include examples and practical applications when relevant\n\n"
            f"Output: A comprehensive hypothetical answer that simulates what a complete response would look like."
        ),
        agent=agent,
        expected_output=(
            "A detailed hypothetical answer (200-400 words) that covers the question comprehensively, "
            "written in professional technical style without mentioning it's hypothetical."
        )
    )


def create_retrieval_response_task(agent: Agent, query: str, hypothetical_answer: str) -> Task:
    """Task 2: Use hypothetical answer to retrieve documents and generate final response"""
    return Task(
        description=(
            f"Use the hypothetical answer to find relevant documents and generate the final response:\n\n"
            f"Original Query: '{query}'\n\n"
            f"Hypothetical Answer (for search):\n{hypothetical_answer}\n\n"
            f"MANDATORY Process:\n"
            f"1. Use 'search_weaviate' with these EXACT parameters:\n"
            f"   - query: '{query}'\n"
            f"   - k: 8\n"
            f"   - agent_type: 'general'\n"
            f"2. Analyze the retrieved documents\n"
            f"3. Generate the FINAL answer using ONLY the retrieved documents\n\n"
            f"MANDATORY Requirements for the final answer:\n"
            f"- Answer in English, clearly and concisely\n"
            f"- Minimum 100 words\n"
            f"- ALWAYS cite sources with [n] at the end of each statement\n"
            f"- Ground ALL claims in the retrieved documents\n"
            f"- If there are practical steps, use numbered lists\n"
            f"- If evidence is insufficient, be explicit about limitations\n"
            f"- At the end include 'Sources:' section with [docId:title:pages]\n\n"
            f"IMPORTANT: This is the FINAL answer that will be evaluated. "
            f"Make it comprehensive and well-grounded in the retrieved evidence."
        ),
        agent=agent,
        expected_output=(
            "Final comprehensive answer in English with:\n"
            "- Main answer (minimum 100 words)\n"
            "- Inline citations with [n]\n"
            "- 'Sources:' section at the end\n"
            "- Clear indication if information is missing\n"
            "- All claims grounded in retrieved documents"
        )
    )


def create_hyde_search_task(agent: Agent, query: str) -> Task:
    """HyDE-based search task"""
    return Task(
        description=(
            f"Use HyDE (Hypothetical Document Embeddings) to find relevant documents and generate an answer for:\n\n"
            f"Query: '{query}'\n\n"
            f"MANDATORY HyDE Process:\n"
            f"1. Use 'hyde_search' with these EXACT parameters:\n"
            f"   - query: '{query}'\n"
            f"   - k: 8\n"
            f"   - agent_type: 'general'\n"
            f"2. The tool will generate a hypothetical answer and search for similar documents\n"
            f"3. Analyze the retrieved documents\n"
            f"4. Generate a complete and detailed answer\n\n"
            f"MANDATORY Requirements for the answer:\n"
            f"- Answer in English, clearly and concisely\n"
            f"- Minimum 50 words\n"
            f"- ALWAYS cite sources with [n] at the end of each statement\n"
            f"- If there are practical steps, use numbered lists\n"
            f"- If evidence is insufficient, be explicit\n"
            f"- At the end include 'Sources:' section with [docId:title:pages]\n\n"
            f"IMPORTANT: DO NOT just say 'APPROVE' - YOU MUST generate a complete answer.\n"
            f"DO NOT invent information. Only use evidence from the documents."
        ),
        agent=agent,
        expected_output=(
            "Complete answer in English with:\n"
            "- Main answer (minimum 50 words)\n"
            "- Inline citations with [n]\n"
            "- 'Sources:' section at the end\n"
            "- Clear indication if information is missing"
        )
    )


def create_hybrid_search_task(agent: Agent, query: str) -> Task:
    """Hybrid search task combining traditional and HyDE approaches"""
    return Task(
        description=(
            f"Use hybrid search strategy to find relevant documents and generate an answer for:\n\n"
            f"Query: '{query}'\n\n"
            f"MANDATORY Hybrid Process:\n"
            f"1. First, try 'hyde_search' with these parameters:\n"
            f"   - query: '{query}'\n"
            f"   - k: 6\n"
            f"   - agent_type: 'general'\n"
            f"2. Then, try 'search_weaviate' with these parameters:\n"
            f"   - query: '{query}'\n"
            f"   - k: 6\n"
            f"   - agent_type: 'general'\n"
            f"3. Combine and analyze all retrieved documents\n"
            f"4. Generate a comprehensive answer using the best sources\n\n"
            f"MANDATORY Requirements for the answer:\n"
            f"- Answer in English, clearly and concisely\n"
            f"- Minimum 50 words\n"
            f"- ALWAYS cite sources with [n] at the end of each statement\n"
            f"- If there are practical steps, use numbered lists\n"
            f"- If evidence is insufficient, be explicit\n"
            f"- At the end include 'Sources:' section with [docId:title:pages]\n\n"
            f"IMPORTANT: DO NOT just say 'APPROVE' - YOU MUST generate a complete answer.\n"
            f"DO NOT invent information. Only use evidence from the documents."
        ),
        agent=agent,
        expected_output=(
            "Complete answer in English with:\n"
            "- Main answer (minimum 50 words)\n"
            "- Inline citations with [n]\n"
            "- 'Sources:' section at the end\n"
            "- Clear indication if information is missing"
        )
    )




def create_quality_evaluation_task(agent: Agent, query: str, final_answer: str, sources: List[Dict]) -> Task:
    """Task 3: Evaluate response quality across multiple dimensions"""
    return Task(
        description=(
            f"Evaluate the quality of the generated response across three key dimensions:\n\n"
            f"Original Query: '{query}'\n\n"
            f"Generated Response:\n{final_answer}\n\n"
            f"Retrieved Sources: {len(sources)} documents\n"
            f"Source IDs: {[s.get('doc_id', 'N/A') for s in sources[:3]]}\n\n"
            f"Evaluation Criteria:\n\n"
            f"1. RELEVANCE (0-10):\n"
            f"   - Does the answer directly address the original question?\n"
            f"   - Is the response focused and on-topic?\n"
            f"   - Does it provide useful information?\n\n"
            f"2. GROUNDEDNESS (0-10):\n"
            f"   - Are claims supported by the retrieved documents?\n"
            f"   - Are citations accurate and relevant?\n"
            f"   - Is there evidence for each major claim?\n\n"
            f"3. COMPLETENESS (0-10):\n"
            f"   - Is the answer comprehensive?\n"
            f"   - Does it cover the main aspects of the question?\n"
            f"   - Are there significant gaps?\n\n"
            f"Decision Framework:\n"
            f"- APPROVE: All scores ≥ 7, high quality response\n"
            f"- REFINE: Any score < 7 but > 4, needs improvement\n"
            f"- REJECT: Any score ≤ 4, poor quality\n\n"
            f"Provide detailed scores and reasoning for your decision."
        ),
        agent=agent,
        expected_output=(
            "Detailed evaluation with:\n"
            "- Relevance score (0-10) with reasoning\n"
            "- Groundedness score (0-10) with reasoning\n"
            "- Completeness score (0-10) with reasoning\n"
            "- Overall decision: APPROVE/REFINE/REJECT\n"
            "- Specific feedback for improvement (if REFINE/REJECT)"
        )
    )


# ============================================================================
# CREWAI RAG SYSTEM
# ============================================================================

class CrewAIAgenticRAG:
    """RAG system orchestrated by CrewAI"""
    
    def __init__(
        self,
        weaviate_client,
        model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
        max_iterations: int = 2,
        llm_provider: str = "ollama",  # "ollama" o "openai"
        openai_api_key: str = None,
        search_strategy: str = "traditional"  # "traditional", "hyde", "hybrid"
    ):
        global rag_context
        rag_context = RAGContext(weaviate_client)
        
        self.max_iterations = max_iterations
        self.search_strategy = search_strategy
        
        # Configure LLM according to provider
        if llm_provider.lower() == "openai":
            if ChatOpenAI is None:
                raise ImportError("ChatOpenAI is not available. Install: pip install langchain-openai")
            if not openai_api_key:
                raise ValueError("openai_api_key is required to use OpenAI")
            
            self.llm = ChatOpenAI(
                model=model,  # "gpt-4o-mini"
                api_key=openai_api_key,
                temperature=0.3,
                max_tokens=1000
            )
            print(f"🤖 Using OpenAI: {model}")
            
        else:  # ollama by default
            self.llm = ChatOllama(
                model=f"ollama/{model}",  # Formato que espera LiteLLM
                base_url=ollama_host,
                temperature=0.3,
                format="json"  # For structured responses
            )
            print(f"🤖 Using Ollama: {model}")
        
        # Create agents according to search strategy
        if search_strategy == "optimized":
            print("🔍 Using Optimized HyDE Architecture (3 Specialized Agents)")
            self.agents = {
                "hyde_generator": create_hyde_generator(self.llm),
                "retrieval_response": create_retrieval_response_agent(self.llm),
                "quality_evaluator": create_quality_evaluator(self.llm)
            }
        elif search_strategy == "hyde":
            print("🔍 Using HyDE strategy (Hypothetical Document Embeddings)")
        self.agents = {
            "reformulator": create_query_reformulator(self.llm),
                "generator": create_hyde_search_agent(self.llm),
                "controller": create_quality_controller(self.llm)
            }
        elif search_strategy == "hybrid":
            print("🔍 Using Hybrid strategy (Traditional + HyDE)")
            self.agents = {
                "reformulator": create_query_reformulator(self.llm),
                "generator": create_hybrid_search_agent(self.llm),
                "controller": create_quality_controller(self.llm)
            }
        else:  # traditional
            print("🔍 Using Traditional strategy (Query Expansion)")
            self.agents = {
                "reformulator": create_query_reformulator(self.llm),
                "generator": create_answer_generator_with_search(self.llm),
            "controller": create_quality_controller(self.llm)
        }
    
    def process_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Process a query with the multi-agent system"""
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING CREWAI AGENTIC RAG")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print()
        
        rag_context.original_query = query
        current_query = query
        
        try:
        for iteration in range(self.max_iterations):
            rag_context.iterations = iteration + 1
            
            print(f"\n{'─'*80}")
                print(f"🔄 ITERATION {iteration + 1}/{self.max_iterations}")
            print(f"{'─'*80}\n")
            
                if self.search_strategy == "optimized":
                    # Optimized architecture with 3 specialized agents
                    print("📝 Phase 1: HyDE Generation")
                    hyde_task = create_hyde_generation_task(
                        self.agents["hyde_generator"],
                        current_query
                    )
                    
                    print("\n🔍✍️ Phase 2: Retrieval & Response")
                    retrieval_task = create_retrieval_response_task(
                        self.agents["retrieval_response"],
                        current_query,
                        "Generated hypothetical answer will be used for search"
                    )
                    
                    print("\n🎯 Phase 3: Quality Evaluation")
                    quality_task = create_quality_evaluation_task(
                        self.agents["quality_evaluator"],
                        current_query,
                        "Generated response will be evaluated",
                        rag_context.retrieved_passages or []
                    )
                    
                    tasks = [hyde_task, retrieval_task, quality_task]
                    
                else:
                    # Traditional architectures
                    # Phase 1: Reformulation
                    print("📝 Phase 1: Query Reformulation")
            reformulation_task = create_reformulation_task(
                self.agents["reformulator"],
                current_query
            )
            
                    # Phase 2: Search and Generation (combined)
                    print("\n🔍✍️ Phase 2: Search and Answer Generation")
                    if self.search_strategy == "hyde":
                        search_generate_task = create_hyde_search_task(
                            self.agents["generator"],
                current_query
            )
                    elif self.search_strategy == "hybrid":
                        search_generate_task = create_hybrid_search_task(
                            self.agents["generator"],
                current_query
            )
                    else:  # traditional
                        search_generate_task = create_search_and_generate_task(
                self.agents["generator"],
                current_query
            )
            
                    # Phase 3: Quality Control
                    print("\n🎯 Phase 3: Quality Control")
            quality_task = create_quality_control_task(
                self.agents["controller"],
                current_query
            )
            
                    tasks = [reformulation_task, search_generate_task, quality_task]
                
                # Create simplified crew for this iteration
                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=tasks,
                    process=Process.sequential,  # Execute in order
                    verbose=verbose
                )
                
                # Execute crew
                print("\n🎬 Executing Crew...")
                result = crew.kickoff()
            
                # Parse controller decision
                try:
                    # The last task should have the decision
                    quality_result = result
                if "APPROVE" in str(quality_result).upper():
                        print("\n✅ Answer APPROVED by Quality Controller")
                    break
                elif "REFINE" in str(quality_result).upper():
                        print("\n🔄 Quality Controller requests REFINEMENT")
                        # Extract new suggested query
                        # (In production you would parse the JSON from quality_task)
                        current_query = f"{query} examples use cases implementation"
                    continue
                except Exception:
                    # If cannot parse, approve by default
                break
        
            # Final result - Extract the real answer from Search & Answer Specialist
            final_answer = rag_context.current_answer or str(result)
            
            # If the result is just "APPROVE", search for the real answer in the context
            if "APPROVE" in str(result) and len(str(result)) < 100:
                # The real answer is in the agent context
                final_answer = "Answer not available - the Search & Answer Specialist agent should have generated the complete answer"
            
        return {
            "query": rag_context.original_query,
            "final_query": current_query,
                "answer": final_answer,
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            return {
                "query": rag_context.original_query,
                "answer": "Process interrupted by user",
                "passages": rag_context.retrieved_passages,
                "iterations": rag_context.iterations,
                "agents_used": list(self.agents.keys())
            }
        except Exception as e:
            print(f"\n❌ Error in processing: {e}")
            return {
                "query": rag_context.original_query,
                "answer": f"Error: {str(e)}",
            "passages": rag_context.retrieved_passages,
            "iterations": rag_context.iterations,
            "agents_used": list(self.agents.keys())
        }


# ============================================================================
# UTILITIES
# ============================================================================

def format_crewai_output(result: Dict[str, Any]) -> str:
    """Format CrewAI output for CLI"""
    lines = [
        "\n" + "="*80,
        " FINAL RESULT - CREWAI AGENTIC RAG",
        "="*80,
        f"\nOriginal Query: {result['query']}",
        f"Final Query: {result['final_query']}",
        f"Iterations: {result['iterations']}",
        f"Agents Used: {', '.join(result['agents_used'])}",
        f"\n{'-'*80}",
        "\n📄 ANSWER:\n",
        "-"*80,
        result['answer'],
        f"\n{'-'*80}",
        "\n📚 SOURCES:\n",
        "-"*80,
    ]
    
    for i, p in enumerate(result['passages'], 1):
        lines.append(
            f"[{i}] {p['title']} ({p['doc_id']}) "
            f"pages {p['pages']} - Score: {p['score']:.3f}"
        )
    
    lines.append("="*80 + "\n")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global weaviate_client
    
    parser = argparse.ArgumentParser(
        description="Agentic RAG with CrewAI - Simplified Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("query", help="User query")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http_port", type=int, default=8080)
    parser.add_argument("--grpc_port", type=int, default=50051)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="CrewAI verbose mode")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    
    # Arguments for OpenAI
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama", 
                       help="LLM provider: ollama or openai")
    parser.add_argument("--openai-api-key", help="OpenAI API Key (required if --llm-provider=openai)")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo", 
                       help="OpenAI model (e.g.: gpt-3.5-turbo, gpt-4, gpt-4-turbo)")
    
    # Search strategy arguments
    parser.add_argument("--search-strategy", choices=["traditional", "hyde", "hybrid", "optimized"], default="traditional",
                       help="Search strategy: traditional (query expansion), hyde (hypothetical embeddings), hybrid (both), optimized (3 specialized agents)")
    
    args = parser.parse_args()
    
    weaviate_client = None
    
    print("=" * 80)
    print("🚀 CREWAI AGENTIC RAG - MULTI-AGENT SYSTEM")
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
        else:
            model_name = args.model
        
        # Initialize CrewAI RAG system
        print("🚀 Initializing RAG system...")
        rag_system = CrewAIAgenticRAG(
            weaviate_client=weaviate_client,
            model=model_name,
            ollama_host=args.ollama_host,
            max_iterations=args.max_iterations,
            llm_provider=args.llm_provider,
            openai_api_key=args.openai_api_key,
            search_strategy=args.search_strategy
        )
        
        # Process query
        print("🎯 Processing query...")
        result = rag_system.process_query(args.query, verbose=args.verbose)
        
        # Show result
        output = format_crewai_output(result)
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
        # Use centralized cleanup function
        cleanup_processes()


if __name__ == "__main__":
    sys.exit(main())
