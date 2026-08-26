class CollectionError(Exception):
    """Base exception for data collection errors."""
    

class InputFileError(CollectionError):
    """Raised when an input file is missing or invalid."""


class RequestError(CollectionError):
    """Raised when a collection request fails."""


class BlockedPageError(CollectionError):
    """Raised when collection is blocked by Amazon."""


class ParsingError(CollectionError):
    """Raised when expected page data cannot be parsed."""


class ValidationError(CollectionError):
    """Raised when collected data fails validation."""