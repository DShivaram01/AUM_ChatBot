"""Task 38: the sink interface. Application code and EvaluationService must
not know or care which concrete sink(s) an event ends up in."""

from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation.schemas import EvaluationEvent


class EvaluationSink(ABC):
    @abstractmethod
    def record(self, event: EvaluationEvent) -> None:
        """Persist one sanitized event. Must not raise for a transient
        failure in a way that breaks the caller's request -- EvaluationService
        is responsible for isolating sink failures, not the sink itself."""
        raise NotImplementedError
