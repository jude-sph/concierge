import pytest
from rtvoice.cancellation import Cancelled, CancellationToken


def test_token_starts_uncancelled():
    assert not CancellationToken().cancelled


def test_check_raises_after_cancel():
    tok = CancellationToken()
    tok.check()  # no raise
    tok.cancel()
    assert tok.cancelled
    with pytest.raises(Cancelled):
        tok.check()


def test_cancel_is_idempotent():
    tok = CancellationToken()
    tok.cancel()
    tok.cancel()
    assert tok.cancelled
