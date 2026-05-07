from typing import Optional

from pydantic import BaseModel


class EvalItem(BaseModel):
    user_message: str
    reference_response: str
    judge_remarks: str
    cluster_id: int
    cluster_title: str
    source_session_id: str
    synthesis_type: str  # "verbatim" or "synthesized"
    capability_tested: Optional[str] = None
