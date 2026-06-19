from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + persistent User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / 'profiles')
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract stable facts and persist to User.md
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append user message to compact memory
        self.compact_memory.append(thread_id, 'user', message)

        # 3. Estimate prompt context (post-compaction)
        prompt_size = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_size
        )

        # 4. Generate response from persisted memory + compact context
        response = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant response to compact memory
        self.compact_memory.append(thread_id, 'assistant', response)

        # 6. Track agent token output
        resp_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + resp_tokens

        return {
            'response': response,
            'agent_tokens': resp_tokens,
            'prompt_tokens': prompt_size,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get('summary', '')  # type: ignore[assignment]
        messages: list[dict[str, str]] = ctx.get('messages', [])  # type: ignore[assignment]
        msg_text = ' '.join(m['content'] for m in messages)
        return estimate_tokens(profile_text) + estimate_tokens(summary) + estimate_tokens(msg_text)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get('summary', '')  # type: ignore[assignment]
        recent: list[dict[str, str]] = ctx.get('messages', [])  # type: ignore[assignment]

        if facts:
            lines = ['Dựa trên thông tin đã lưu về bạn:']
            label_map = {
                'name': 'Tên',
                'location': 'Nơi ở',
                'profession': 'Nghề nghiệp',
                'drink': 'Đồ uống yêu thích',
                'food': 'Món ăn yêu thích',
                'pet': 'Thú cưng',
                'style': 'Style trả lời',
            }
            for key, label in label_map.items():
                if key in facts:
                    lines.append(f'- {label}: {facts[key]}')
            # Include any extra facts not in the map
            for key, val in facts.items():
                if key not in label_map:
                    lines.append(f'- {key}: {val}')
            if summary:
                lines.append(f'\nLịch sử tóm tắt:\n{summary[:300]}')
            return '\n'.join(lines)

        if summary:
            return f'Lịch sử tóm tắt:\n{summary[:300]}'

        if recent:
            user_recent = [m['content'] for m in recent if m['role'] == 'user']
            if user_recent:
                return f'Trong phiên này bạn đã chia sẻ: {user_recent[-1][:200]}'

        return 'Xin chào! Hãy giới thiệu bản thân để mình có thể ghi nhớ và hỗ trợ bạn tốt hơn.'

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live LangGraph path — placeholder, falls back to offline."""
        return self._reply_offline(user_id, thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persisted User.md profile for a given user."""
                return profile_store.read_text(user_id)

            @tool
            def upsert_user_fact(user_id: str, key: str, value: str) -> str:
                """Write or update a single fact in User.md."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Updated {key} = {value}"

            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm,
                tools=[read_user_profile, upsert_user_fact],
                checkpointer=checkpointer,
            )
        except Exception:
            self.langchain_agent = None
