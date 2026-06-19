from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    state_dir = root / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    model_name = os.environ.get('LLM_MODEL', 'claude-haiku-4-5-20251001')
    temperature = float(os.environ.get('LLM_TEMPERATURE', '0.7'))

    api_key: str | None = None
    base_url: str | None = None
    p = provider.lower()
    if p == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
    elif p == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
    elif p == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    elif p == 'openrouter':
        api_key = os.environ.get('OPENROUTER_API_KEY')
    elif p == 'custom':
        api_key = os.environ.get('CUSTOM_API_KEY')
        base_url = os.environ.get('CUSTOM_BASE_URL')
    elif p == 'ollama':
        base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

    judge_provider = os.environ.get('JUDGE_PROVIDER', provider)
    judge_model_name = os.environ.get('JUDGE_MODEL', model_name)

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / 'data',
        state_dir=state_dir,
        compact_threshold_tokens=int(os.environ.get('COMPACT_THRESHOLD_TOKENS', '500')),
        compact_keep_messages=int(os.environ.get('COMPACT_KEEP_MESSAGES', '4')),
        model=model_cfg,
        judge_model=judge_cfg,
    )
