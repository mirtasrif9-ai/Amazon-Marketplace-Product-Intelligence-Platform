from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchProcessingResult:
    """
    Stores the result of batch sentiment processing.
    """

    successful_products: list[dict[str, Any]] = field(
        default_factory=list
    )

    failed_products: list[dict[str, Any]] = field(
        default_factory=list
    )

    total_products: int = 0

    @property
    def successful_count(self) -> int:
        return len(self.successful_products)

    @property
    def failed_count(self) -> int:
        return len(self.failed_products)