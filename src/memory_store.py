from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


def _extract_city_name(raw: str) -> str:
    """Return the leading title-case word(s) from raw, stopping at lowercase words."""
    stop_words = {'không', 'chứ', 'nhé', 'và', 'nữa', 'mà', 'để', 'vì', 'khi', 'thì', 'với', 'nên', 'vài', 'một'}
    words = re.split(r'\s+', raw.strip())
    result = []
    for w in words:
        clean = w.strip('.,!?:;')
        if not clean:
            continue
        if clean[0].isupper() and clean.lower() not in stop_words:
            result.append(clean)
        else:
            break
    return ' '.join(result)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r'[^\w\-]', '_', user_id)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return f"# User Profile: {user_id}\n\n"
        return p.read_text(encoding='utf-8')

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if line.startswith('- ') and ': ' in line:
                key, _, val = line[2:].partition(': ')
                result[key.strip()] = val.strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        existing = self.facts(user_id)
        content = self.read_text(user_id)
        new_line = f"- {key}: {value}"
        if key in existing:
            old_line = f"- {key}: {existing[key]}"
            content = content.replace(old_line, new_line, 1)
        else:
            if not content.endswith('\n'):
                content += '\n'
            content += new_line + '\n'
        self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a user message using heuristic patterns."""
    facts: dict[str, str] = {}

    # Skip if the entire message is a bare question (no declarative content)
    non_q = [s.strip() for s in re.split(r'[?!]', message) if s.strip() and '?' not in s]
    if not non_q and message.strip().endswith('?'):
        return {}

    # ── Name ────────────────────────────────────────────────────────────────
    stop_names = {'gì', 'không', 'thì', 'và', 'là', 'mình', 'bạn', 'ai', 'nào', 'sao'}
    for pat in [
        r'(?:mình|tôi|tao)\s+tên\s+(?:là\s+)?(\w+)',
        r'Chào\s+bạn[,.]?\s+(?:mình\s+)?tên\s+(?:là\s+)?(\w+)',
        r'tên\s+(?:là\s+)?(\w+)\b',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in stop_names:
                facts['name'] = name
                break

    # ── Location ────────────────────────────────────────────────────────────
    # Find locations mentioned negatively so we can exclude them
    excluded_locs: set[str] = set()
    for m in re.finditer(r'không\s+(?:còn\s+)?(?:ở|tại)\s+(\w+(?:\s+\w+)?)', message):
        excluded_locs.add(_extract_city_name(m.group(1)))

    negation_phrases = [
        'chỉ là nơi', 'chỉ ghé', 'không phải nơi ở', 'vừa bay ra họp',
        'chỉ đến họp', 'chỉ đến',
    ]
    loc_candidates: list[str] = []
    for pat in [
        r'(?:hiện(?:\s+tại)?|từ\s+\w+\s+này|giờ)\s+(?:mình\s+)?(?:đang\s+)?(?:ở|tại|làm việc ở)\s+(\w+(?:\s+\w+){0,2})',
        r'(?:đang\s+)?làm việc ở\s+(\w+(?:\s+\w+){0,2})',
        r'(?:mình|tôi)\s+(?:đang\s+)?(?:ở|tại)\s+(\w+(?:\s+\w+){0,2})',
        r'(?:đang\s+ở|ở tại)\s+(\w+(?:\s+\w+){0,2})',
    ]:
        for m in re.finditer(pat, message, re.IGNORECASE):
            city = _extract_city_name(m.group(1))
            if not city or city in excluded_locs:
                continue
            ctx_start = max(0, m.start() - 20)
            ctx_end = min(len(message), m.end() + 80)
            ctx = message[ctx_start:ctx_end]
            if not any(neg in ctx for neg in negation_phrases):
                non_cities = {'đây', 'đó', 'chỗ', 'nhà', 'đâu', 'nơi', 'trên', 'kia', 'đây'}
                if city.lower() not in non_cities:
                    loc_candidates.append(city)
    if loc_candidates:
        facts['location'] = loc_candidates[-1]

    # ── Profession ──────────────────────────────────────────────────────────
    prof_kw = r'(?:engineer|developer|manager|researcher|designer|architect|ops|scientist|analyst)'
    # Correction pattern: "không còn làm X nữa, giờ Y"
    corr = re.search(
        r'không\s+còn\s+làm\s+.+?(?:nữa)[^.]*[,\.]\s*(?:giờ\s+)?(?:chuyển sang|làm)\s+(.+?' + prof_kw + r')',
        message, re.IGNORECASE,
    )
    if corr:
        facts['profession'] = corr.group(1).strip().rstrip('.,!?')
    else:
        for pat in [
            r'(?:chuyển sang|giờ là|hiện là)\s+(.+?' + prof_kw + r')',
            r'(?:mình|tôi)\s+(?:đang\s+)?làm\s+(.+?' + prof_kw + r')\b',
            r'(?:nghề nghiệp|công việc)\s+(?:hiện tại\s+)?(?:là\s+)?(.+?' + prof_kw + r')',
        ]:
            m = re.search(pat, message, re.IGNORECASE)
            if m:
                ctx_before = message[max(0, m.start() - 30):m.start()]
                if 'không còn' not in ctx_before:
                    facts['profession'] = m.group(1).strip().rstrip('.,!?')
                    break

    # ── Food ────────────────────────────────────────────────────────────────
    food_m = re.search(r'món ăn yêu thích\s+(?:của\s+mình\s+)?(?:là\s+)?([^,\.!?\n]+)', message, re.IGNORECASE)
    if food_m:
        facts['food'] = food_m.group(1).strip()

    # ── Drink ───────────────────────────────────────────────────────────────
    drink_m = re.search(r'đồ uống yêu thích\s+(?:của\s+mình\s+)?(?:là\s+)?([^,\.!?\n]+)', message, re.IGNORECASE)
    if drink_m:
        facts['drink'] = drink_m.group(1).strip()

    # ── Pet ─────────────────────────────────────────────────────────────────
    pet_m = re.search(r'nuôi\s+(?:một\s+bé\s+|một\s+|bé\s+)?(\w+)\s+tên\s+(\w+)', message, re.IGNORECASE)
    if pet_m:
        facts['pet'] = f"{pet_m.group(1)} tên {pet_m.group(2)}"

    # ── Response style ───────────────────────────────────────────────────────
    has_short = bool(re.search(r'(?:trả lời|câu trả lời|giải thích).*ngắn\s*gọn|ngắn\s*gọn.*(?:trả lời|giải thích)', message, re.IGNORECASE))
    has_bullet3 = '3 bullet' in message or 'ba bullet' in message.lower()
    if has_bullet3:
        facts['style'] = '3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off'
    elif has_short:
        facts['style'] = 'ngắn gọn, có ví dụ thực tế, dạng bullet'

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages."""
    if not messages:
        return ''
    subset = messages[-max_items:]
    parts = []
    for msg in subset:
        role = 'Người dùng' if msg.get('role') == 'user' else 'Agent'
        content = msg.get('content', '')[:120]
        parts.append(f"{role}: {content}")
    return 'Tóm tắt hội thoại cũ:\n' + '\n'.join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {'messages': [], 'summary': '', 'compactions': 0}

    def _should_compact(self, thread_id: str) -> bool:
        st = self.state[thread_id]
        if len(st['messages']) <= self.keep_messages:
            return False
        all_text = ' '.join(m['content'] for m in st['messages'])
        return estimate_tokens(all_text) > self.threshold_tokens

    def _compact(self, thread_id: str) -> None:
        st = self.state[thread_id]
        messages: list[dict[str, str]] = st['messages']
        old = messages[:-self.keep_messages]
        keep = messages[-self.keep_messages:]
        new_summary = summarize_messages(old)
        existing = st['summary']
        st['summary'] = (existing + '\n\n' + new_summary).strip() if existing else new_summary
        st['messages'] = keep
        st['compactions'] = int(st['compactions']) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        self.state[thread_id]['messages'].append({'role': role, 'content': content})
        if self._should_compact(thread_id):
            self._compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]['compactions'])
