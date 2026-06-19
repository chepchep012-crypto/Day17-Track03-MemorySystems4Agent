from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    state_dir = tmp_path / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy_provider = ProviderConfig(
        provider='anthropic',
        model_name='claude-haiku-4-5-20251001',
        temperature=0.0,
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / 'data',
        state_dir=state_dir,
        compact_threshold_tokens=100,
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / 'profiles')

    # Default profile is empty-ish
    default = store.read_text('user1')
    assert 'user1' in default

    # Write and read back
    store.write_text('user1', '# User Profile: user1\n\n- name: Alice\n')
    content = store.read_text('user1')
    assert 'Alice' in content, f"Expected 'Alice' in: {content!r}"

    # Edit existing fact
    changed = store.edit_text('user1', '- name: Alice', '- name: Bob')
    assert changed, 'edit_text should return True when text was found and replaced'
    content = store.read_text('user1')
    assert 'Bob' in content
    assert 'Alice' not in content

    # Edit missing text returns False
    assert not store.edit_text('user1', 'no such line', 'x')

    # File size is non-zero
    assert store.file_size('user1') > 0

    print('PASS test_user_markdown_read_write_edit')


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    user_id = 'compact_user'
    thread_id = 'compact_thread'
    # Each message is ~400 chars → ~100 tokens, well above threshold=100 per message
    long_msg = 'Đây là một đoạn văn bản dài để kích hoạt compact memory trong bài test này. ' * 6

    for _ in range(5):
        agent.reply(user_id, thread_id, long_msg)

    compactions = agent.compaction_count(thread_id)
    assert compactions >= 1, f'Expected ≥1 compaction after 5 long turns, got {compactions}'
    print(f'PASS test_compact_trigger (compactions={compactions})')


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = 'recall_user'
    thread1 = 'intro_thread'

    # Session 1: introduce facts
    advanced.reply(user_id, thread1, 'Chào bạn, mình tên là TestUser.')
    advanced.reply(user_id, thread1, 'Mình ở Hà Nội và đang làm data scientist.')
    baseline.reply(user_id, thread1, 'Chào bạn, mình tên là TestUser.')
    baseline.reply(user_id, thread1, 'Mình ở Hà Nội và đang làm data scientist.')

    # Session 2 (fresh thread): recall
    thread2 = 'recall_thread'
    adv_result = advanced.reply(user_id, thread2, 'Mình tên gì và nghề nghiệp là gì?')
    base_result = baseline.reply(user_id, thread2, 'Mình tên gì và nghề nghiệp là gì?')

    adv_answer = adv_result.get('response', '')
    base_answer = base_result.get('response', '')

    assert 'TestUser' in adv_answer, (
        f'Advanced agent should recall name across sessions.\nGot: {adv_answer!r}'
    )
    assert 'TestUser' not in base_answer, (
        f'Baseline should NOT recall name in a fresh thread.\nGot: {base_answer!r}'
    )

    print('PASS test_cross_session_recall')
    print(f'  Advanced: {adv_answer[:120]}')
    print(f'  Baseline: {base_answer[:120]}')


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    user_id = 'load_user'
    thread_id = 'long_thread'
    long_msg = 'Đây là đoạn hội thoại rất dài để stress test prompt load. ' * 22

    for _ in range(8):
        advanced.reply(user_id, thread_id, long_msg)
        baseline.reply(user_id, thread_id, long_msg)

    adv_prompt = advanced.prompt_token_usage(thread_id)
    base_prompt = baseline.prompt_token_usage(thread_id)

    assert adv_prompt < base_prompt, (
        f'Advanced prompt load ({adv_prompt}) should be < baseline ({base_prompt}) '
        f'on a long thread thanks to compact memory.'
    )

    print('PASS test_compact_reduces_prompt_load_on_long_thread')
    print(f'  Advanced cumulative prompt tokens : {adv_prompt}')
    print(f'  Baseline cumulative prompt tokens : {base_prompt}')
    print(f'  Savings: {base_prompt - adv_prompt} tokens ({(base_prompt - adv_prompt) / base_prompt * 100:.1f}%)')


if __name__ == '__main__':
    import tempfile

    # Ensure Vietnamese characters render on Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        test_user_markdown_read_write_edit(p / 't1')
        test_compact_trigger(p / 't2')
        test_cross_session_recall(p / 't3')
        test_compact_reduces_prompt_load_on_long_thread(p / 't4')
    print('\nAll tests passed.')
