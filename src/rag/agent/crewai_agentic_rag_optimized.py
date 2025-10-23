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


# Global instance
rag_context: Optional[RAGContext] = None


# ============================================================================
# TOOLS
# ============================================================================

@tool("search_weaviate")
def search_weaviate(query: str, k: int = 8, agent_type: str = "general") -> str:
    """Search in Weaviate with specific parameters."""
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
        max_iter=1
    )


def create_retrieval_response_agent(llm) -> Agent:
    """Agent 2: Retrieval & Response - Uses search_weaviate to find documents and generate final response"""
    return Agent(
        role="Retrieval & Response Specialist",
        goal="Use search_weaviate to retrieve relevant documents and generate final grounded responses",
        backstory=(
            "You are an expert in RAG (Retrieval-Augmented Generation). You receive hypothetical answers "
            "from the HyDE Generator and use them to find the most relevant documents in the knowledge base. "
            "Then you generate the final answer using only the retrieved documents as evidence. "
            "You excel at grounding responses in real evidence and citing sources accurately."
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
    """Task 2: Use search_weaviate to retrieve documents and generate final response"""
    return Task(
        description=(
            f"Use search_weaviate to find relevant documents and generate the final response:\n\n"
            f"Original Query: '{query}'\n\n"
            f"Hypothetical Answer (for context):\n{hypothetical_answer}\n\n"
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
            f"- Do NOT include citations or source references\n"
            f"- Ground ALL claims in the retrieved documents but don't cite them\n"
            f"- If there are practical steps, use numbered lists\n"
            f"- If evidence is insufficient, be explicit about limitations\n"
            f"- Write as a natural, flowing response without academic citations\n\n"
            f"IMPORTANT: This is the FINAL answer that will be evaluated. "
            f"Make it comprehensive and well-grounded in the retrieved evidence, but write it naturally without citations."
        ),
        agent=agent,
        expected_output=(
            "Final comprehensive answer in English with:\n"
            "- Main answer (minimum 100 words)\n"
            "- Natural, flowing text without citations\n"
            "- Clear indication if information is missing\n"
            "- All claims grounded in retrieved documents\n"
            "- Professional but accessible tone"
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
# OPTIMIZED RAG SYSTEM
# ============================================================================

class OptimizedCrewAIRAG:
    """Optimized RAG system with 3 specialized agents"""
    
    def __init__(
        self,
        weaviate_client,
        model: str = "mistral",
        ollama_host: str = "http://localhost:11434",
        max_iterations: int = 1,
        llm_provider: str = "ollama",
        openai_api_key: str = None
    ):
        global rag_context
        rag_context = RAGContext(weaviate_client)
        
        self.max_iterations = max_iterations
        
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
            
        else:  # ollama by default
            self.llm = ChatOllama(
                model=f"ollama/{model}",
                base_url=ollama_host,
                temperature=0.3,
                format="json"
            )
            print(f"🤖 Using Ollama: {model}")
        
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
        print(f"🚀 OPTIMIZED CREWAI RAG - 3 AGENT ARCHITECTURE")
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
                
                # Phase 1: HyDE Generation
                print("📝 Phase 1: HyDE Generation")
                hyde_task = create_hyde_generation_task(
                    self.agents["hyde_generator"],
                    current_query
                )
                
                # Phase 2: Retrieval & Response
                print("\n🔍✍️ Phase 2: Retrieval & Response")
                retrieval_task = create_retrieval_response_task(
                    self.agents["retrieval_response"],
                    current_query,
                    "Generated hypothetical answer will be used for search"
                )
                
                # Phase 3: Quality Evaluation
                print("\n🎯 Phase 3: Quality Evaluation")
                quality_task = create_quality_evaluation_task(
                    self.agents["quality_evaluator"],
                    current_query,
                    "Generated response will be evaluated",
                    rag_context.retrieved_passages or []
                )
                
                # Create crew
                crew = Crew(
                    agents=list(self.agents.values()),
                    tasks=[hyde_task, retrieval_task, quality_task],
                    process=Process.sequential,
                    verbose=verbose
                )
                
                # Execute crew
                print("\n🎬 Executing Crew...")
                result = crew.kickoff()
                
                # Parse result - get the answer from the second task (Retrieval & Response)
                try:
                    # The result is a list of task results, we want the second one (Retrieval & Response)
                    if hasattr(result, '__iter__') and len(result) >= 2:
                        final_answer = str(result[1])  # Second task result
                    else:
                        final_answer = str(result)
                    
                    # Check quality evaluation (third task)
                    if hasattr(result, '__iter__') and len(result) >= 3:
                        quality_result = str(result[2])
                        if "APPROVE" in quality_result.upper():
                            print("\n✅ Answer APPROVED by Quality Evaluator")
                            break
                        elif "REFINE" in quality_result.upper():
                            print("\n🔄 Quality Evaluator requests REFINEMENT")
                            current_query = f"{query} examples use cases implementation"
                            continue
                    else:
                        break
                except Exception:
                    final_answer = str(result)
                    break
            
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

def format_output(result: Dict[str, Any]) -> str:
    """Format output for CLI"""
    lines = [
        "\n" + "="*80,
        " FINAL RESULT - OPTIMIZED CREWAI RAG",
        "="*80,
        f"\nOriginal Query: {result['query']}",
        f"Final Query: {result['final_query']}",
        f"Iterations: {result['iterations']}",
        f"Agents Used: {', '.join(result['agents_used'])}",
        f"Sources Retrieved: {len(result['passages'])} documents",
        f"\n{'-'*80}",
        "\n📄 ANSWER:\n",
        "-"*80,
        result['answer'],
        f"\n{'-'*80}",
        "\n📊 METRICS:\n",
        "-"*80,
        f"• Documents retrieved: {len(result['passages'])}",
        f"• Average relevance score: {sum(p['score'] for p in result['passages']) / len(result['passages']) if result['passages'] else 0:.3f}",
        f"• Processing iterations: {result['iterations']}",
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
    parser.add_argument("--llm-provider", choices=["ollama", "openai"], default="ollama", 
                       help="LLM provider: ollama or openai")
    parser.add_argument("--openai-api-key", help="OpenAI API Key (required if --llm-provider=openai)")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo", 
                       help="OpenAI model (e.g.: gpt-3.5-turbo, gpt-4, gpt-4-turbo)")
    
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
            openai_api_key=args.openai_api_key
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
