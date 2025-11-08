from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("question", "contexts")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    CONTEXTS_FIELD_NUMBER: _ClassVar[int]
    question: str
    contexts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, question: _Optional[str] = ..., contexts: _Optional[_Iterable[str]] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("answer", "processingTime", "errorMessage", "success")
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    PROCESSINGTIME_FIELD_NUMBER: _ClassVar[int]
    ERRORMESSAGE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    answer: str
    processingTime: float
    errorMessage: str
    success: bool
    def __init__(self, answer: _Optional[str] = ..., processingTime: _Optional[float] = ..., errorMessage: _Optional[str] = ..., success: bool = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "status_message", "modelLoaded")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    MODELLOADED_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status_message: str
    modelLoaded: str
    def __init__(self, healthy: bool = ..., status_message: _Optional[str] = ..., modelLoaded: _Optional[str] = ...) -> None: ...
