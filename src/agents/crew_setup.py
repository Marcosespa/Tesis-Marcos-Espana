#!/usr/bin/env python3
"""
CrewAI setup for multi-class RAG retrieval.

This module provides backward compatibility and convenience functions
for the YAML-based CrewAI configuration.

Usage examples:
    # Legacy approach (still supported)
    from src.agents.crew_setup import kickoff_query
    print(kickoff_query("What is NIST risk management?", agent_type="policy"))
    
    # New YAML-based approach (recommended)
    from src.agents.crew import kickoff_query_yaml
    print(kickoff_query_yaml("What is NIST risk management?", agent_type="policy"))
    
    # Using the full crew
    from src.agents.crew import MultiClassRagCrew
    crew = MultiClassRagCrew()
    result = crew.crew().kickoff(inputs={'query': 'What is NIST risk management?'})
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure src on sys.path when running as a script
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from crewai import Agent, Task, Crew, Tool
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "CrewAI is required. Install with: pip install crewai"
    ) from exc

try:
    # Import multi-class searcher
    from src.rag.search.multi_class_search import MultiClassSemanticSearch, AgentType
except ModuleNotFoundError:
    from rag.search.multi_class_search import MultiClassSemanticSearch, AgentType  # type: ignore


DEFAULT_CONFIG_PATH = str(SRC_DIR.parent / "configs" / "search_config.yaml")


def _ensure_searcher(config_path: Optional[str] = None) -> MultiClassSemanticSearch:
    """Create and cache a searcher instance."""
    cfg = config_path or DEFAULT_CONFIG_PATH
    return MultiClassSemanticSearch(config_path=cfg)


def _retrieve(query: str, agent_key: str, top_k: int = 8, config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    searcher = _ensure_searcher(config_path)
    agent_enum = AgentType(agent_key)
    results = searcher.search(query, agent=agent_enum, top_k=top_k)
    return [r.to_dict() for r in results]


def _make_retrieval_tool(agent_key: str, description: str) -> Tool:
    return Tool(
        name=f"retrieval_{agent_key}",
        description=f"Multi-class retrieval for {agent_key}: {description}",
        func=lambda q: _retrieve(str(q), agent_key=agent_key),
    )


def build_agents(config_path: Optional[str] = None) -> Dict[str, Agent]:
    """Create CrewAI agents per domain with bound retrieval tools.
    
    DEPRECATED: Use MultiClassRagCrew from src.agents.crew instead.
    This function is kept for backward compatibility.
    """
    # Ensure searcher initializes early (loads models lazily inside search)
    _ensure_searcher(config_path)

    tools = {
        "policy": _make_retrieval_tool("policy", "NIST policies and standards"),
        "research": _make_retrieval_tool("research", "USENIX research papers"),
        "attack": _make_retrieval_tool("attack", "MITRE ATT&CK framework"),
        "security": _make_retrieval_tool("security", "OWASP and security tools"),
        "ai": _make_retrieval_tool("ai", "AI security knowledge"),
        "training": _make_retrieval_tool("training", "Training resources (AnnoCTR)"),
        "general": _make_retrieval_tool("general", "All collections")
    }

    agents: Dict[str, Agent] = {
        "policy": Agent(
            role="PolicyAnalyst",
            goal="Responder sobre políticas/estándares NIST citando fuentes.",
            backstory="Especialista en NIST, FIPS, CSWP.",
            tools=[tools["policy"]],
            verbose=False,
            allow_delegation=True,
        ),
        "research": Agent(
            role="ResearchAnalyst",
            goal="Sintetizar hallazgos de papers (USENIX) con citas.",
            backstory="Investigador académico en seguridad.",
            tools=[tools["research"]],
            verbose=False,
            allow_delegation=True,

        ),
        "attack": Agent(
            role="AttackAnalyst",
            goal="Explicar tácticas/técnicas ATT&CK con referencias.",
            backstory="Analista de amenazas y técnicas MITRE.",
            tools=[tools["attack"]],
            verbose=False,
            allow_delegation=True,
        ),
        "security": Agent(
            role="SecurityAnalyst",
            goal="Recomendar mitigaciones y mejores prácticas (OWASP).",
            backstory="Ingeniero de seguridad de apps.",
            tools=[tools["security"]],
            verbose=False,
            allow_delegation=True,
        ),
        "ai": Agent(
            role="AISecAnalyst",
            goal="Relacionar normativa/prácticas de seguridad en IA con citas.",
            backstory="Especialista en seguridad de IA y guías NIST AI.",
            tools=[tools["ai"]],
            verbose=False,
            allow_delegation=True,
        ),
        "training": Agent(
            role="Trainer",
            goal="Proveer material de entrenamiento y definiciones con fuentes.",
            backstory="Instructor de ciberseguridad.",
            tools=[tools["training"]],
            verbose=False,
            allow_delegation=True,
        ),
        "general": Agent(
            role="Generalist",
            goal="Responder usando todas las colecciones con citas.",
            backstory="Asesor generalista de ciberseguridad.",
            tools=[tools["general"]],
            verbose=False,
            allow_delegation=True,
        ),
    }

    return agents


def kickoff_query(query: str, agent_type: str = "general", config_path: Optional[str] = None) -> str:
    """Run a single-turn task using a selected agent and return the final output string.
    
    DEPRECATED: Use kickoff_query_yaml from src.agents.crew instead.
    This function is kept for backward compatibility.
    """
    agents = build_agents(config_path=config_path)
    if agent_type not in agents:
        raise ValueError(f"Unknown agent_type '{agent_type}'. Valid: {list(agents.keys())}")

    agent = agents[agent_type]
    task = Task(
        description=(
            "Responde de forma concisa y cita fuentes de los pasajes recuperados. "
            "Incluye referencias en formato [collection:pages].\n\n"
            f"Pregunta: {query}"
        ),
        agent=agent,
        expected_output=(
            "Respuesta con bullets y citas. Si pocos pasajes, indica que la evidencia es limitada."
        ),
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":  
    import argparse

    parser = argparse.ArgumentParser(description="CrewAI kickoff over multi-class retrieval")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--agent", type=str, default="general",
                        choices=["policy", "research", "attack", "security", "ai", "training", "general"],
                        help="Agent/domain to use")
    parser.add_argument("--config", type=str, default=None, help="Path to search_config.yaml")
    parser.add_argument("--yaml", action="store_true", help="Use YAML-based configuration (recommended)")
    args = parser.parse_args()

    if args.yaml:
        # Use new YAML-based approach
        try:
            from src.agents.crew import kickoff_query_yaml
            print(kickoff_query_yaml(args.query, agent_type=args.agent, config_path=args.config))
        except ImportError:
            print("Error: YAML-based crew not available. Using legacy approach.")
            print(kickoff_query(args.query, agent_type=args.agent, config_path=args.config))
    else:
        # Use legacy approach
        print("Warning: Using legacy configuration. Consider using --yaml flag for recommended approach.")
        print(kickoff_query(args.query, agent_type=args.agent, config_path=args.config))


