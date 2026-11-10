"""Upload validation: the plan's 413 / 415 / 400 cases, and the in-memory guarantee."""

import json
import os
from tempfile import SpooledTemporaryFile
from typing import ClassVar

from images import encode, jpeg, noise
from PIL import Image
from starlette import formparsers


def post(client, data: bytes, content_type="image/jpeg", name="a.jpg"):
    return client.post("/api/predict", files={"file": (name, data, content_type)})


def err(r) -> tuple[str, str]:
    return r.json()["code"], r.json()["detail"]


def test_file_over_the_limit_is_413_before_the_body_is_read(client):
    r = post(client, os.urandom(15 * 1024 * 1024))  # the plan's 15 MB case
    assert r.status_code == 413 and err(r)[0] == "payload_too_large"
    assert "10 MB" in err(r)[1]


def test_file_slightly_over_the_limit_is_413_from_the_body_check(make_client):
    c = make_client(max_upload_mb=1)
    r = post(c, os.urandom(int(1.2 * 1024 * 1024)))  # under the Content-Length pre-check margin
    assert r.status_code == 413 and "1 MB" in err(r)[1]


def test_file_at_the_limit_is_accepted(make_client):
    data = jpeg(300, 300, quality=95)
    limit_mb = len(data) // (1024 * 1024) + 1
    assert post(make_client(max_upload_mb=limit_mb), data).status_code == 200


def test_text_file_is_415(client):
    r = post(client, b"hello", "text/plain", "notes.txt")
    assert r.status_code == 415 and err(r)[0] == "unsupported_media_type"
    assert "JPEG, PNG or WebP" in err(r)[1]


def test_missing_content_type_is_415(client):
    assert post(client, jpeg(), content_type="").status_code in (415, 422)


def test_truncated_jpeg_is_400(client):
    data = jpeg(300, 300)
    r = post(client, data[: len(data) // 2])
    assert r.status_code == 400 and err(r)[0] == "invalid_image"
    assert "truncated" in err(r)[1]


def test_text_bytes_declared_as_an_image_are_400(client):
    r = post(client, b"this is not an image", "image/jpeg")
    assert r.status_code == 400 and err(r)[0] == "invalid_image"


def test_empty_file_is_400(client):
    assert post(client, b"").status_code == 400


def test_a_real_image_in_an_unsupported_format_is_415(client):
    r = post(client, encode(noise(), "GIF"), "image/jpeg")  # lies about its type
    assert r.status_code == 415 and err(r)[0] == "unsupported_media_type"


def test_image_over_the_pixel_limit_is_413(make_client):
    c = make_client(max_image_pixels=1000)
    r = post(c, jpeg(40, 26))  # 1040 px: over the limit, under Pillow's own 2x bomb threshold
    assert r.status_code == 413 and "more than 1,000 pixels" in err(r)[1]


def test_pillows_own_bomb_guard_is_413_with_the_same_message(make_client, monkeypatch):
    c = make_client(max_image_pixels=1000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # Pillow raises at open past 2x this
    r = post(c, jpeg(300, 300))
    assert r.status_code == 413 and "more than 1,000 pixels" in err(r)[1]


def test_apps_do_not_share_a_pixel_limit(make_client):
    strict = make_client(max_image_pixels=1000)
    normal = make_client()  # created later; must not inherit (or override) the strict limit
    assert post(strict, jpeg(100, 100)).status_code == 413
    assert post(normal, jpeg(100, 100)).status_code == 200
    assert post(strict, jpeg(100, 100)).status_code == 413


def test_image_exactly_at_the_pixel_limit_is_accepted(make_client):
    assert post(make_client(max_image_pixels=1000), jpeg(40, 25)).status_code == 200


def test_error_responses_never_contain_a_stack_trace(client):
    for data, ctype in [(b"x", "text/plain"), (b"not an image", "image/jpeg"), (b"", "image/png")]:
        text = post(client, data, ctype).text
        assert "Traceback" not in text and 'File "' not in text
        assert set(json.loads(text)) == {"detail", "code"}


class _Recorder:
    """Stand-in for SpooledTemporaryFile that records whether any file hit the disk."""

    instances: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.inner = SpooledTemporaryFile(*args, **kwargs)  # noqa: SIM115 - wrapped on purpose
        _Recorder.instances.append(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def upload_2mb(client):
    data = jpeg(1100, 1100, quality=98)  # noise compresses poorly: well over 1 MB
    assert len(data) > 1024 * 1024
    assert post(client, data).status_code == 200


def test_uploads_larger_than_1mb_stay_in_memory(client, monkeypatch):
    """Starlette spools multipart files over 1 MB to disk unless told otherwise; the app
    raises that threshold above the upload limit (privacy: uploads never touch the disk)."""
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", _Recorder)
    _Recorder.instances.clear()
    upload_2mb(client)
    assert _Recorder.instances and not any(f._rolled for f in _Recorder.instances)


def test_control_the_default_threshold_would_have_written_to_disk(client, monkeypatch):
    """Proves the test above is sensitive: with Starlette's 1 MB default the same upload rolls
    over to a temporary file on disk."""
    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", _Recorder)
    monkeypatch.setattr(formparsers.MultiPartParser, "spool_max_size", 1024 * 1024)
    _Recorder.instances.clear()
    upload_2mb(client)
    assert any(f._rolled for f in _Recorder.instances)
