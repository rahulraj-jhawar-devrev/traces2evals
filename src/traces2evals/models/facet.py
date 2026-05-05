from typing import Optional

from pydantic import BaseModel


class Facet(BaseModel):
    session_id: str
    facet_idx: int = 0
    summary: str
    num_turns: int
    user_id: Optional[str] = None
    user_text: Optional[str] = None
    token_count: Optional[int] = None
