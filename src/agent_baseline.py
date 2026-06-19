from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only, no persistent User.md."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._get_session(thread_id)
        session.messages.append({'role': 'user', 'content': message})

        response = self._generate_response(session, message)
        session.messages.append({'role': 'assistant', 'content': response})

        resp_tokens = estimate_tokens(response)
        session.token_usage += resp_tokens

        # Prompt load: all accumulated context (grows linearly per turn)
        context_text = ' '.join(m['content'] for m in session.messages)
        context_tokens = estimate_tokens(context_text)
        session.prompt_tokens_processed += context_tokens

        return {
            'response': response,
            'agent_tokens': resp_tokens,
            'prompt_tokens': context_tokens,
        }

    def _generate_response(self, session: SessionState, message: str) -> str:
        """Deterministic offline reply using only current session history."""
        stop_words = {'gì', 'không', 'thì', 'và', 'là', 'mình', 'bạn', 'ai'}
        user_msgs = [m['content'] for m in session.messages if m['role'] == 'user']
        session_text = ' '.join(user_msgs)

        # Recall / summary questions
        recall_kws = ['tên gì', 'tên của mình', 'mình tên', 'nhắc lại', 'bạn biết', 'mô tả', 'tóm tắt',
                      'nghề gì', 'nghề nghiệp', 'ở đâu', 'nơi ở', 'đồ uống', 'món ăn', 'style', 'nuôi']
        if any(kw in message.lower() for kw in recall_kws):
            if len(user_msgs) > 1:
                # Try to surface name from this session
                name_m = re.search(r'(?:mình|tôi)\s+tên\s+(?:là\s+)?(\w+)', session_text)
                if name_m and name_m.group(1).lower() not in stop_words:
                    return (
                        f"Trong phiên này bạn đã giới thiệu tên là {name_m.group(1)}. "
                        f"Ngoài ra: {'; '.join(user_msgs[1:3])}"
                    )
                return f"Trong phiên này bạn đã chia sẻ: {'; '.join(user_msgs[:2])}"
            return "Xin chào! Mình chưa có thông tin về bạn trong phiên này."

        # Default acknowledgement
        return "Cảm ơn bạn đã chia sẻ. Mình đang ghi nhận thông tin trong phiên này."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live LangGraph path (placeholder — falls back to offline)."""
        return self._reply_offline(thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            llm = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None
