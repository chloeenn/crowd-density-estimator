"""Admission control: bounded concurrency and a bounded queue."""

import asyncio

import pytest

from app.errors import ApiError
from app.inference import InferenceService


def service(**kw) -> InferenceService:
    return InferenceService({"A": object(), "B": object()}, "A", **kw)


def test_default_part_must_be_loaded():
    with pytest.raises(ValueError, match="default part"):
        InferenceService({"B": object()}, "A")


def test_get_defaults_and_rejects_unknown_parts():
    s = InferenceService({"A": "model-a"}, "A")
    assert s.get(None) == s.get("A") == "model-a"
    with pytest.raises(ApiError) as e:
        s.get("B")
    assert e.value.status_code == 422 and "Available: A" in e.value.detail


def test_requests_beyond_running_plus_queue_get_503_with_retry_after():
    async def scenario():
        s = service(max_concurrent=1, max_queue=1, retry_after_s=7)
        gate = asyncio.Event()

        async def hold():
            async with s.slot():
                await gate.wait()

        running, queued = asyncio.create_task(hold()), asyncio.create_task(hold())
        await asyncio.sleep(0.01)  # one is running, one is waiting for the semaphore
        with pytest.raises(ApiError) as e:
            async with s.slot():
                pass
        gate.set()
        await asyncio.gather(running, queued)
        return s, e.value

    s, err = asyncio.run(scenario())
    assert err.status_code == 503 and err.code == "busy" and err.headers == {"Retry-After": "7"}
    assert s._pending == 0  # fully drained


def test_concurrency_never_exceeds_the_limit():
    async def scenario():
        s = service(max_concurrent=2, max_queue=10)
        running = peak = 0

        async def work():
            nonlocal running, peak
            async with s.slot():
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0.01)
                running -= 1

        await asyncio.gather(*(work() for _ in range(8)))
        return peak

    assert asyncio.run(scenario()) == 2


def test_a_failing_request_releases_its_slot():
    async def scenario():
        s = service(max_concurrent=1, max_queue=0)
        with pytest.raises(RuntimeError):
            async with s.slot():
                raise RuntimeError("inference blew up")
        async with s.slot():  # would raise 503 if the slot leaked
            return s._pending

    assert asyncio.run(scenario()) == 1
