import time
from concurrent import futures
from pathlib import Path
import grpc
import llm_service.llm_service_pb2_grpc as llm_service_pb2_grpc
import llm_service.llm_service_pb2 as llm_service_pb2
from logger import get_logger
from llm_client import LLMClient
from prompt_engine import PromptEngine
from query_classifier import QueryClassifier

logger = get_logger(__name__)


class LLMServiceServicer(llm_service_pb2_grpc.LLMServiceServicer):
    """
    gRPC servicer implementation for the LLM Service.

    Handles incoming requests for generating responses using the LLM
    and performing health checks on the service.
    """

    def __init__(self, llm_client: LLMClient, prompt_engine: PromptEngine):
        """
        Initialize the LLMServiceServicer.

        Args:
            llm_client (LLMClient): The LLM client instance for generating responses.
            prompt_engine (PromptEngine): The prompt engine for building prompts.
        """
        self.logger = logger
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        self.query_classifier = QueryClassifier()

        self.logger.info('LLMServiceServicer initialized')

    async def Generate(self, request, context):
        """
        Generate a response to the user's question using the LLM.

        Classifies the query type, builds a prompt with provided contexts,
        generates the answer, and returns a response with processing time.

        Args:
            request: The gRPC GenerateRequest containing the question and contexts.
            context: The gRPC servicer context.

        Returns:
            llm_service_pb2.GenerateResponse:
            The response with answer, processing time, and success status.
        """
        start_time = time.time()

        try:
            self.logger.info(
                f'Generate request received: question="{request.question}", '
                f'contexts_count={len(request.contexts)}'
            )

            template_type = self.query_classifier.classify(request.question)
            if request.contexts:
                context_text = '\n'.join(request.contexts)
            else:
                context_text = ''

            self.logger.debug(
                f'Template type: {template_type}'
            )

            prompt = self.prompt_engine.build_prompt(
                template_name=template_type,
                context=context_text,
                question=request.question,
            )

            self.logger.debug(
                'Sending prompt to LLM...'
            )
            answer = await self.llm_client.generate(prompt=prompt)
            processing_time = time.time() - start_time
            self.logger.debug(
                f'Generation complete: question="{request.question}", '
                f'Processing time: {processing_time}'
                f'Answer length: {len(answer)}'
            )

            return llm_service_pb2.GenerateResponse(
                answer=answer,
                processingTime=processing_time,
                errorMessage='',
                success=True,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = f'Generation error: {str(e)}'
            self.logger.error(error_message)
            return llm_service_pb2.GenerateResponse(
                answer='',
                processingTime=processing_time,
                errorMessage=error_message,
            )

    async def HealthCheck(self, request, context):
        """
        Perform a health check on the LLM service.

        Verifies if prompt templates are loaded and if prompts can be built.

        Args:
            request: The gRPC HealthCheckRequest.
            context: The gRPC servicer context.

        Returns:
            llm_service_pb2.HealthResponse: The health status response.
        """
        try:
            if not self.prompt_engine.templates:
                return llm_service_pb2.HealthResponse(
                    healthy=False,
                    status_message='Prompt templates not loaded',
                    modelLoaded='',
                )

            test_prompt = self.prompt_engine.build_prompt(
                'definition',
                context='Test context',
                question='Test question',
            )

            if not test_prompt:
                return llm_service_pb2.HealthResponse(
                    healthy=False,
                    status_message='Cannot build prompts',
                    modelLoaded='',
                )

            return llm_service_pb2.HealthResponse(
                healthy=True,
                status_message='Service is healthy',
                modelLoaded='llama3.2:3b-instruct-q4_K_M',
            )

        except Exception as e:
            return llm_service_pb2.HealthResponse(
                healthy=False,
                status_message=f'Health check failed: {str(e)}',
                modelLoaded='',
            )


async def serve_grpc(host: str = 'localhost', port: int = 8081):
    """
    Start gRPC-server.

    Initializes the LLMClient and PromptEngine, sets up the gRPC server
    with the LLMServiceServicer, and starts listening on the specified host and port.

    Args:
        host (str, optional): The host to bind the server to. Defaults to 'localhost'.
        port (int, optional): The port to bind the server to. Defaults to 8081.

    Returns:
        grpc.aio.Server: The started gRPC server instance.

    Raises:
        Exception: If server startup fails.
    """
    async with LLMClient() as llm_client:
        try:
            prompts_dir = Path(__file__).parent.parent / 'prompts'
            logger.info(f'Loading prompts from: {prompts_dir}')

            prompt_engine = PromptEngine(prompts_dir)

            await llm_client.initialize()
            logger.info('LLM client initialized successfully')

            server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )

            servicer = LLMServiceServicer(llm_client, prompt_engine)
            llm_service_pb2_grpc.add_LLMServiceServicer_to_server(servicer, server)

            listen_addr = f'{host}:{port}'
            server.add_insecure_port(listen_addr)

            logger.info(f'Starting gRPC server on {listen_addr}')
            await server.start()

            return server

        except Exception as e:
            logger.error(f'Failed to start gRPC server: {e}')
            raise
    return None
