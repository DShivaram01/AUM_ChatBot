"""Process-local AssistantService registration without UI ownership."""

from core.assistant_service import AssistantService

_service: AssistantService | None = None


def set_assistant_service(service: AssistantService) -> None:
    global _service
    _service = service


def get_assistant_service() -> AssistantService:
    if _service is None:
        raise RuntimeError("AssistantService has not been initialized")
    return _service
