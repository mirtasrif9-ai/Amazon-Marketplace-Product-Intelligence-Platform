from src.common.logger import setup_logging
from src.data_collection.collectors.oxylabs_sandbox_search import (
    OxylabsSandboxSearchCollector,
)


setup_logging()

collector = OxylabsSandboxSearchCollector()

products = collector.search(
    "https://sandbox.oxylabs.io/products"
)

print("\nCollected products:")

for product in products:
    print(product)