from dataclasses import dataclass


@dataclass
class ThumbnailGroupResult:
    asin: str
    cluster_id: int
    visual_group: str
    group_size: int