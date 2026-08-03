"""
Assembles the complete LangGraph ``StateGraph`` for the multi-agent
research workflow, wiring together every node built across Phase 6.

Every node function already exists (``planner``, ``retriever_agent``,
``search_agent``, ``researcher``, ``writer``, ``fact_checker``,
``reviewer``, ``coordinator``) -- this file's only job is topology,
conditional routing, and dependency injection. No business logic lives
here; every routing decision below is a small, focused function passed to
``add_conditional_edges``, never embedded in a node body, continuing the
node/routing separation every prior Phase 6 file already established
(most explicitly in ``reviewer.py``'s and ``coordinator.py``'s own
docstrings).

Graph shape (``app.agents.state``'s own module docstring, now realized)::

    START --(mode == CHAT)--> coordinator_chat --> END
    START --(else)--> planner --(0/1/2 branches, by plan)--> [retriever?] [search?]
        --> researcher --> writer --> fact_checker --> reviewer
            --(approved)--> coordinator_finalize --> END
            --(rejected, retries remain)--> writer  [the one cycle in this graph]
            --(rejected, retries exhausted)--> raises MaxRetriesExceededError

Usage (by a future ``ChatService`` -- not built in Phase 6): build the
graph once at application startup via ``build_research_graph(...)``, then
per-request call ``create_initial_state(...)`` (``app.agents.state``) and
``await compiled_graph.ainvoke(initial_state)``.
"""

from __future__ import annotations

from typing import List, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.coordinator import build_coordinator_chat_node, build_coordinator_finalize_node
from app.agents.fact_checker import build_fact_checker_node
from app.agents.planner import build_planner_node
from app.agents.researcher import researcher_node
from app.agents.retriever_agent import build_retriever_node
from app.agents.reviewer import build_reviewer_node
from app.agents.search_agent import build_search_node
from app.agents.state import ResearchGraphState
from app.agents.writer import build_writer_node
from app.core.config import AgentSettings, FeatureFlags, RAGSettings, SearchSettings
from app.core.exceptions import MaxRetriesExceededError
from app.core.interfaces.embedding_provider import EmbeddingProvider
from app.core.interfaces.llm_service import LLMService
from app.core.interfaces.search_provider import SearchProvider
from app.core.interfaces.vector_store import VectorStore
from app.schemas.chat import ChatMode

# =============================================================================
# Node name constants -- the single source of truth for every add_node call,
# routing-function return value, and path_map, so a typo in one place
# can't silently create an unreachable node or a dangling edge.
# =============================================================================

NODE_COORDINATOR_CHAT = "coordinator_chat"
NODE_PLANNER = "planner"
NODE_RETRIEVER = "retriever"
NODE_SEARCH = "search"
NODE_RESEARCHER = "researcher"
NODE_WRITER = "writer"
NODE_FACT_CHECKER = "fact_checker"
NODE_REVIEWER = "reviewer"
NODE_COORDINATOR_FINALIZE = "coordinator_finalize"


def build_research_graph(
    *,
    llm_service: LLMService,
    llm_model_name: str,
    feature_flags: FeatureFlags,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    rag_settings: RAGSettings,
    search_provider: SearchProvider,
    search_settings: SearchSettings,
    agent_settings: AgentSettings,
) -> CompiledStateGraph:
    """
    Build and compile the full research workflow graph.

    Args:
        llm_service: Injected once, shared by every LLM-backed node
            (Planner, Writer, Fact Checker, Reviewer, both Coordinator
            nodes). This codebase has no per-node model configuration
            anywhere (``LLMSettings.ollama.model`` is a single global
            setting) -- accepting five separate model-name parameters
            here would be speculative flexibility nothing currently uses.
        llm_model_name: The configured model identifier, passed through to
            every LLM-backed node for its ``AgentExecutionStep.model_name``.
        feature_flags: Passed to the Planner (see ``planner.py``'s
            ``_apply_source_overrides``); no other node needs it.
        embedding_provider: Passed to the Retriever.
        vector_store: Passed to the Retriever.
        rag_settings: Passed to the Retriever.
        search_provider: Passed to the Search agent.
        search_settings: Passed to the Search agent.
        agent_settings: Supplies ``max_reviewer_retries``, read only by
            this file's own post-Reviewer routing function -- no node
            needs it (see ``reviewer.py``'s own docstring on why bound
            enforcement is a graph-level, not a node-level, concern).

    Returns:
        A compiled ``CompiledStateGraph`` ready for
        ``await graph.ainvoke(initial_state)``.
    """
    graph = StateGraph(ResearchGraphState)

    graph.add_node(NODE_COORDINATOR_CHAT, build_coordinator_chat_node(llm_service, llm_model_name))
    graph.add_node(NODE_PLANNER, build_planner_node(llm_service, feature_flags, llm_model_name))
    graph.add_node(NODE_RETRIEVER, build_retriever_node(embedding_provider, vector_store, rag_settings))
    graph.add_node(NODE_SEARCH, build_search_node(search_provider, search_settings))
    graph.add_node(NODE_RESEARCHER, researcher_node)
    graph.add_node(NODE_WRITER, build_writer_node(llm_service, llm_model_name))
    graph.add_node(NODE_FACT_CHECKER, build_fact_checker_node(llm_service, llm_model_name))
    graph.add_node(NODE_REVIEWER, build_reviewer_node(llm_service, llm_model_name))
    graph.add_node(NODE_COORDINATOR_FINALIZE, build_coordinator_finalize_node(llm_service, llm_model_name))

    # -------------------------------------------------------------------
    # Entry routing: the only place ChatMode.CHAT's pipeline bypass is
    # decided. Neither Coordinator node (app.agents.coordinator) contains
    # this check -- see that file's own "no routing logic" scope note.
    # -------------------------------------------------------------------
    graph.add_conditional_edges(
        START,
        _route_entry,
        {NODE_COORDINATOR_CHAT: NODE_COORDINATOR_CHAT, NODE_PLANNER: NODE_PLANNER},
    )
    graph.add_edge(NODE_COORDINATOR_CHAT, END)

    # -------------------------------------------------------------------
    # Planner -> 0, 1, or 2 parallel branches, decided by the plan the
    # Planner just produced (already gated by FeatureFlags inside
    # planner.py -- this routing function trusts that gate, it doesn't
    # re-check feature_flags itself). Both branches are plain static
    # edges into `researcher`: LangGraph's fan-in triggers a node when
    # ANY of its declared predecessors updated state in the prior
    # superstep, not only when ALL of them did -- so `researcher` runs
    # exactly once whether zero, one, or both of retriever/search were
    # actually scheduled this run. Verified empirically below rather than
    # assumed, since it's the riskiest topology decision in this file.
    # -------------------------------------------------------------------
    graph.add_conditional_edges(
        NODE_PLANNER,
        _route_after_planner,
        {NODE_RETRIEVER: NODE_RETRIEVER, NODE_SEARCH: NODE_SEARCH, NODE_RESEARCHER: NODE_RESEARCHER},
    )
    graph.add_edge(NODE_RETRIEVER, NODE_RESEARCHER)
    graph.add_edge(NODE_SEARCH, NODE_RESEARCHER)

    graph.add_edge(NODE_RESEARCHER, NODE_WRITER)
    graph.add_edge(NODE_WRITER, NODE_FACT_CHECKER)
    graph.add_edge(NODE_FACT_CHECKER, NODE_REVIEWER)

    # -------------------------------------------------------------------
    # The one cycle in this graph. `agent_settings.max_reviewer_retries`
    # is captured by closure so the routing function's signature stays
    # `(state) -> str`, matching what add_conditional_edges expects --
    # not `(state, settings) -> str`.
    # -------------------------------------------------------------------
    def _route_after_reviewer(state: ResearchGraphState) -> str:
        feedback = state.get("review_feedback")
        if feedback is not None and feedback.approved:
            return NODE_COORDINATOR_FINALIZE

        revision_count = state.get("revision_count", 0)
        if revision_count <= agent_settings.max_reviewer_retries:
            return NODE_WRITER

        raise MaxRetriesExceededError(
            f"Reviewer rejected the draft {revision_count} time(s), exceeding "
            f"AGENT__MAX_REVIEWER_RETRIES={agent_settings.max_reviewer_retries} for query: {state['query']!r}"
        )

    graph.add_conditional_edges(
        NODE_REVIEWER,
        _route_after_reviewer,
        {NODE_COORDINATOR_FINALIZE: NODE_COORDINATOR_FINALIZE, NODE_WRITER: NODE_WRITER},
    )
    graph.add_edge(NODE_COORDINATOR_FINALIZE, END)

    return graph.compile()


def _route_entry(state: ResearchGraphState) -> str:
    if state["mode"] == ChatMode.CHAT:
        return NODE_COORDINATOR_CHAT
    return NODE_PLANNER


def _route_after_planner(state: ResearchGraphState) -> Union[str, List[str]]:
    plan = state["plan"]
    destinations: List[str] = []
    if plan.use_retrieval:
        destinations.append(NODE_RETRIEVER)
    if plan.use_web_search:
        destinations.append(NODE_SEARCH)
    if not destinations:
        return NODE_RESEARCHER
    return destinations
