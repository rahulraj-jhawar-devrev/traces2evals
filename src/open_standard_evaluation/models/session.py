from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Speaker(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    role: Speaker
    content: str
    timestamp: Optional[datetime] = None
    metadata: dict = {}


class SessionScore(BaseModel):
    name: str
    value: float
    source: str = "unknown"
    comment: Optional[str] = None


class NormalizedSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[Message]
    scores: list[SessionScore] = []
    metadata: dict = {}
    created_at: Optional[datetime] = None

    @property
    def num_turns(self) -> int:
        return len([m for m in self.messages if m.role in (Speaker.USER, Speaker.ASSISTANT)])

    @property
    def conversation_text(self) -> str:
        lines = []
        for msg in self.messages:
            if msg.role in (Speaker.USER, Speaker.ASSISTANT):
                label = "User" if msg.role == Speaker.USER else "Assistant"
                lines.append(f"{label}: {msg.content}")
        return "\n".join(lines)

    @property
    def user_messages_text(self) -> str:
        return " ".join(m.content for m in self.messages if m.role == Speaker.USER)
