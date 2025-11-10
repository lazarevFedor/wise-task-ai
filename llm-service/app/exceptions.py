class LLMClientError(Exception):
    """Base LLM Client Error"""
    pass


class LLMTimeoutError(LLMClientError):
    """Timeout Error while connecting to LLM"""

    def __init__(self, operation: str, timeout_seconds: float):
        """
        Initialize the LLMTimeoutError.

        Args:
            operation (str): The operation that timed out.
            timeout_seconds (float):
            The duration in seconds after which the operation timed out.
        """
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(f'Operation {operation} timed out after '
                         f'{timeout_seconds} seconds.')


class LLMUnavailableError(LLMClientError):
    """LLM Unavailable Error"""

    def __init__(self, url: str, error_description: str):
        """
        Initialize the LLMUnavailableError.

        Args:
            url (str): The URL of the unavailable LLM service.
            error_description (str):
            A description of the error or reason for unavailability.
        """
        self.url = url
        self.error_description = error_description
        super().__init__(f'Operation {url} unavailable '
                         f'at {error_description}.')
