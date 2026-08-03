"""
ORM model registry.

Importing this package guarantees every ``app.models.*`` module has been
imported at least once, which is what registers its model classes on
``Base.metadata`` (see ``app.database.session.Database.create_all``'s
docstring). Without this, whether ``documents``/``reports`` tables exist
after ``create_all()`` would depend on incidental import order elsewhere
in the app (a router importing ``conversation.py`` but not ``report.py``,
for example) -- a fragile, easy-to-silently-break guarantee. Application
startup should do ``import app.models`` (or rely on this package being
imported transitively) before calling ``Database.create_all()``.
"""

from __future__ import annotations

from app.models.conversation import AgentExecutionStep, ConversationTurn, ResearchSession, TurnCitation
from app.models.document import Document
from app.models.report import Report

__all__ = [
    "ResearchSession",
    "ConversationTurn",
    "TurnCitation",
    "AgentExecutionStep",
    "Document",
    "Report",
]
