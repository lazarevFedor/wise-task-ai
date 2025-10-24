class LLMClientError(Exception):
    """Base LLM Client Error"""
    pass


class LLMTimeoutError(LLMClientError):
    """Timeout Error while connecting to LLM"""
    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(f'Operation {operation} timed out after {timeout_seconds} seconds.')


class LLMUnavailableError(LLMClientError):
    """LLM Unavailable Error"""
    def __init__(self, url: str, error_description: str):
        self.url = url
        self.error_description = error_description
        super().__init__(f'Operation {url} unavailable at {error_description}.')


class LLMOverloadError(LLMClientError):
    """LLM Mover Load Error"""
    def __init__(self, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        super().__init__(f'Reached maximum concurrent requests: {max_concurrent_requests}.')
