from gen.llm_service import llm_service_pb2_grpc, llm_service_pb2
from .logger import get_logger
from .llm_client import LLMClient
from .prompt_engine import PromptEngine
from .query_classifier import QueryClassifier


class LLMServiceServicer(llm_service_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        pass

    async def Generate(self, request, context):
        pass

    async def HealthCheck(self, request, context):
        pass
