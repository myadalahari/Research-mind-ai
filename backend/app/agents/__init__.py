"""
The LangGraph multi-agent research workflow.

Eight agents (``Coordinator``, ``Planner``, ``Retriever``, ``Search``,
``Researcher``, ``Writer``, ``Fact Checker``, ``Reviewer``) wired into a
single ``StateGraph`` (``app.agents.graph``), sharing the state defined in
``app.agents.state``. See that module's docstring for the graph's shape
and the reasoning behind its state design.
"""

from __future__ import annotations
