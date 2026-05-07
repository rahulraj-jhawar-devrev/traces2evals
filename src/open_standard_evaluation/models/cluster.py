from typing import Optional

from pydantic import BaseModel


class ClusterInfo(BaseModel):
    cluster_id: int
    title: str
    description: str
    session_count: int
    user_count: Optional[int] = None
    parent_group: Optional[int] = None


class ClusterHierarchy(BaseModel):
    clusters: list[ClusterInfo]
    levels: list[dict[int, int]] = []
    quality_metrics: dict = {}
