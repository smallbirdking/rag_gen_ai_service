from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class AppConfig:
    raw: Dict[str, Any]
    root_dir: Path

    @property
    def corpus_path(self) -> Path:
        return self.root_dir / self.raw["corpus"]["path"]

    @property
    def log_path(self) -> Path:
        return self.root_dir / self.raw["app"]["log_path"]


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = None) -> AppConfig:
    load_project_env()
    config_path = Path(path or os.getenv("RAG_CONFIG_PATH", "config/app.yaml"))
    if not config_path.is_absolute():
        # Resolve relative to current working directory first; fallback to repository root.
        cwd_candidate = Path.cwd() / config_path
        config_path = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Lightweight environment overrides for demo/deployment.
    env_overrides: Dict[str, Any] = {}
    if os.getenv("RAG_RETRIEVAL_MODE"):
        env_overrides.setdefault("retrieval", {})["mode"] = os.environ["RAG_RETRIEVAL_MODE"]
    if os.getenv("RAG_RERANKER_ENABLED"):
        env_overrides.setdefault("retrieval", {})["reranker_enabled"] = os.environ["RAG_RERANKER_ENABLED"].lower() == "true"
    if os.getenv("RAG_LLM_PROVIDER"):
        env_overrides.setdefault("llm", {})["provider"] = os.environ["RAG_LLM_PROVIDER"]
    if os.getenv("RAG_MODEL"):
        env_overrides.setdefault("llm", {})["model"] = os.environ["RAG_MODEL"]

    root_dir = config_path.parent.parent.resolve()
    return AppConfig(raw=deep_update(raw, env_overrides), root_dir=root_dir)
