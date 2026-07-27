"""Cancellation as a control-plane operation.

Firing a token does not require the reasoner's cooperation, any more than
SIGINT requires asking a process nicely. The executor polls between operations.
"""
from __future__ import annotations


class Cancelled(Exception):
    """Raised by CancellationToken.check() once the token has been fired."""


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise Cancelled()
