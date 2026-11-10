"""Settings come from the environment, and .env.example stays in sync with the defaults."""

from pathlib import Path

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "3")
    monkeypatch.setenv("MODEL_PART", "B")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.example, http://b.example,")
    s = Settings(_env_file=None)
    assert s.max_upload_mb == 3 and s.max_upload_bytes == 3 * 1024 * 1024 and s.model_part == "B"
    assert s.cors_origin_list == ["http://a.example", "http://b.example"]


def test_cors_defaults_to_the_vite_dev_server_never_a_wildcard():
    assert Settings(_env_file=None).cors_origin_list == ["http://localhost:5173"]


def test_env_example_documents_exactly_the_defaults():
    defaults = Settings(_env_file=None)
    documented = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            documented[key.lower()] = value
    assert documented, "no settings found in .env.example"
    for key, value in documented.items():
        assert str(getattr(defaults, key)) == value, key
    assert "alt_model_path" not in documented  # optional: commented out by default
