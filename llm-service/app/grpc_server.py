import grpc
from time import time as now
from concurrent import futures
from pathlib import Path
from llm_service import llm_service_pb2_grpc, llm_service_pb2
from logger import get_logger
from llm_client import LLMClient
from prompt_engine import PromptEngine
from query_classifier import QueryClassifier
from config import config
from grpc_reflection.v1alpha import reflection
from exceptions import LLMTimeoutError

logger = get_logger(__name__)


class llmServiceServicer(llm_service_pb2_grpc.llmServiceServicer):
    """
    gRPC servicer implementation for the LLM Service.

    Handles incoming requests for generating responses using the LLM
    and performing health checks on the service.
    """

    def __init__(self, llm_client: LLMClient, prompt_engine: PromptEngine):
        """
        Initialize the llmServiceServicer.

        Args:
            llm_client (LLMClient): The LLM client instance for generating responses.
            prompt_engine (PromptEngine): The prompt engine for building prompts.
        """
        self.logger = logger
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        self.query_classifier = QueryClassifier()

        self.logger.info('llmServiceServicer initialized')

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
        start_time = now()

        try:
            self.logger.info(
                f'request_id={request.requestId} - '
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
            answer = await self.llm_client.generate(prompt=prompt, )
            processing_time = now() - start_time
            self.logger.debug(
                f'request_id={request.requestId} - '
                f'Generation complete: question="{request.question}", '
                f'Processing time: {processing_time}'
                f'Answer length: {len(answer)}'
            )

            return llm_service_pb2.GenerateResponse(
                answer=answer,
                processingTime=processing_time,
                errorMessage='',
            )

        except Exception as e:
            processing_time = now() - start_time
            if isinstance(e, LLMTimeoutError):
                error_message = "LLM_TIMEOUT"
            else:
                error_message = "LLM_UNAVAILABLE"
            self.logger.error(f'request_id={request.requestId} - ' + str(e))
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
                    status_message='LLM_UNHEALTH',
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
                    status_message='LLM_UNHEALTH',
                    modelLoaded='',
                )

            await self.llm_client.health_check()

            return llm_service_pb2.HealthResponse(
                healthy=True,
                status_message='Service is healthy',
                modelLoaded=config.LLM_DEFAULT_MODEL,
            )

        except Exception:
            return llm_service_pb2.HealthResponse(
                healthy=False,
                status_message='LLM_UNHEALTH',
                modelLoaded='',
            )


async def serve_grpc(host: str = 'localhost', port: int = 8084):
    """
    Start gRPC-server.

    Initializes the LLMClient and PromptEngine, sets up the gRPC server
    with the LLMServiceServicer, and starts listening on the specified host and port.

    Args:
        host (str, optional): The host to bind the server to. Defaults to 'localhost'.
        port (int, optional): The port to bind the server to. Defaults to 8084.

    Returns:
        grpc.aio.Server: The started gRPC server instance.

    Raises:
        Exception: If server startup fails.
    """
    host = config.LLM_GRPC_HOST or host
    port = config.LLM_GRPC_PORT or port
    llm_client = None

    try:
        prompts_dir = Path(__file__).parent.parent / 'prompts'
        logger.debug(f'Loading prompts from: {prompts_dir}')

        prompt_engine = PromptEngine(prompts_dir)
        llm_client = LLMClient()

        await llm_client.initialize()
        logger.debug('LLM client initialized successfully')

        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
        )

        SERVICE_NAMES = (
            llm_service_pb2.DESCRIPTOR.services_by_name['llmService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)

        servicer = llmServiceServicer(llm_client, prompt_engine)
        llm_service_pb2_grpc.add_llmServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{port}')

        logger.info(f'Starting gRPC server on http://{host}:{port}')
        await server.start()

        return server

    except Exception as e:
        logger.error(f'Failed to start gRPC server: {e}')
        await llm_client.close()
        raise
