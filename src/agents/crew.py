#!/usr/bin/env python3
"""
CrewAI crew configuration using YAML-based approach.

This module provides a CrewBase class for multi-class RAG retrieval
with domain-specific agents configured via YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Ensure src on sys.path when running as a script
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from crewai import Agent, Task, Crew
    from crewai.project import CrewBase, agent, crew, task
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "CrewAI is required. Install with: pip install crewai"
    ) from exc

try:
    # Import multi-class searcher
    from src.rag.search.multi_class_search import MultiClassSemanticSearch, AgentType
except ModuleNotFoundError:
    from rag.search.multi_class_search import MultiClassSemanticSearch, AgentType  # type: ignore


def _ensure_searcher(config_path: Optional[str] = None) -> MultiClassSemanticSearch:
    """Create and cache a searcher instance."""
    from src.agents.crew_setup import DEFAULT_CONFIG_PATH
    cfg = config_path or DEFAULT_CONFIG_PATH
    return MultiClassSemanticSearch(config_path=cfg)


def _make_retrieval_tool(agent_key: str, description: str, config_path: Optional[str] = None):
    """Create a retrieval tool for the specified agent."""
    from crewai import Tool
    
    def _retrieve(query: str, top_k: int = 8) -> list[dict[str, Any]]:
        searcher = _ensure_searcher(config_path)
        agent_enum = AgentType(agent_key)
        results = searcher.search(query, agent=agent_enum, top_k=top_k)
        return [r.to_dict() for r in results]
    
    return Tool(
        name=f"retrieval_{agent_key}",
        description=f"Multi-class retrieval for {agent_key}: {description}",
        func=_retrieve,
    )


@CrewBase
class MultiClassRagCrew:
    """Multi-class RAG crew for cybersecurity domain-specific retrieval."""

    agents_config = "configs/agents.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        # Initialize searcher early
        _ensure_searcher(config_path)

    @agent
    def policy(self) -> Agent:
        return Agent(
            config=self.agents_config['policy'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("policy", "NIST policies and standards", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def research(self) -> Agent:
        return Agent(
            config=self.agents_config['research'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("research", "USENIX research papers", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def attack(self) -> Agent:
        return Agent(
            config=self.agents_config['attack'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("attack", "MITRE ATT&CK framework", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def security(self) -> Agent:
        return Agent(
            config=self.agents_config['security'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("security", "OWASP and security tools", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def ai(self) -> Agent:
        return Agent(
            config=self.agents_config['ai'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("ai", "AI security knowledge", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def training(self) -> Agent:
        return Agent(
            config=self.agents_config['training'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("training", "Training resources (AnnoCTR)", self.config_path)],
            allow_delegation=True,
        )

    @agent
    def general(self) -> Agent:
        return Agent(
            config=self.agents_config['general'],  # type: ignore[index]
            verbose=False,
            tools=[_make_retrieval_tool("general", "All collections", self.config_path)],
            allow_delegation=True,
        )

    @task
    def analyze_query(self) -> Task:
        return Task(
            description=(
                "Responde de forma concisa y cita fuentes de los pasajes recuperados. "
                "Incluye referencias en formato [collection:pages].\n\n"
                "Pregunta: {query}"
            ),
            agent=self.general(),
            expected_output=(
                "Respuesta con bullets y citas. Si pocos pasajes, indica que la evidencia es limitada."
            ),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MultiClassRagCrew crew with all agents"""
        return Crew(
            agents=self.agents,  # type: ignore[attr-defined]
            tasks=self.tasks,  # type: ignore[attr-defined]
            verbose=False,
        )


def kickoff_query_yaml(query: str, agent_type: str = "general", config_path: Optional[str] = None) -> str:
    """Run a single-turn task using a selected agent and return the final output string."""
    crew_instance = MultiClassRagCrew(config_path=config_path)
    
    # Create a task with the selected agent
    agent = getattr(crew_instance, agent_type)()
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
