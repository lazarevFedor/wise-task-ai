import time
from gen.llm_service import llm_service_pb2_grpc, llm_service_pb2
from .logger import get_logger
from .llm_client import LLMClient
from .prompt_engine import PromptEngine
from .query_classifier import QueryClassifier


class LLMServiceServicer(llm_service_pb2_grpc.LLMServiceServicer):
    def __init__(self, llm_client: LLMClient, prompt_engine: PromptEngine):
        self.logger = get_logger(__name__)
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        self.query_classifier = QueryClassifier()

        self.logger.info("LLMServiceServicer initialized")

    async def Generate(self, request, context):
        start_time = time.time()

        try:
            self.logger.info(
                f'Generate request received: question="{request.question}", '
                f'contexts_count={len(request.contexts)}'
            )

            template_type = self.query_classifier.classify(request.question)
            context_text = "\n".join(request.contexts) if request.contexts else ""
            self.logger.debug(
                f'Template type: {template_type}'
            )

            prompt = self.prompt_engine.build_prompt(
                template_name=template_type,
                context=context_text,
                question=request.question,
            )

            self.logger.debug(
                f'Sending prompt to LLM...'
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
                success=False,
            )

    async def HealthCheck(self, request, context):
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
