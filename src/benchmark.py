from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """0 / 0.5 / 1.0 based on how many expected facts appear in answer."""
    if not expected:
        return 1.0
    found = sum(1 for e in expected if e.lower() in answer.lower())
    ratio = found / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """0.0–1.0 quality score: coverage × length penalty."""
    if not expected or not answer.strip():
        return 0.0
    found = sum(1 for e in expected if e.lower() in answer.lower())
    coverage = found / len(expected)
    length_ok = 10 <= len(answer) <= 800
    return coverage * (0.8 + 0.2 * int(length_ok))


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: Any,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_recall_q = 0
    memory_growth = 0
    total_compactions = 0

    for conv in conversations:
        user_id: str = conv['user_id']
        thread_id: str = conv['id']
        recall_thread_id = f"{conv['id']}_recall"

        # Feed conversation turns
        for turn in conv.get('turns', []):
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get('agent_tokens', 0)
            total_prompt_tokens += result.get('prompt_tokens', 0)

        # Recall questions in a fresh thread
        for rq in conv.get('recall_questions', []):
            result = agent.reply(user_id, recall_thread_id, rq['question'])
            answer: str = result.get('response', '')
            total_agent_tokens += result.get('agent_tokens', 0)
            total_prompt_tokens += result.get('prompt_tokens', 0)

            expected: list[str] = rq.get('expected_contains', [])
            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            total_recall_q += 1

        total_compactions += agent.compaction_count(thread_id)

        if hasattr(agent, 'memory_file_size'):
            memory_growth += agent.memory_file_size(user_id)

    avg_recall = total_recall / total_recall_q if total_recall_q else 0.0
    avg_quality = total_quality / total_recall_q if total_recall_q else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        'Agent',
        'Agent tokens only',
        'Prompt tokens processed',
        'Cross-session recall',
        'Response quality',
        'Memory growth (bytes)',
        'Compactions',
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f'{r.recall_score:.2f}',
            f'{r.response_quality:.2f}',
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate
        return tabulate(data, headers=headers, tablefmt='pipe')
    except ImportError:
        col_widths = [max(len(str(h)), max((len(str(row[i])) for row in data), default=0)) for i, h in enumerate(headers)]
        def fmt_row(row: list) -> str:
            return '| ' + ' | '.join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)) + ' |'
        sep = '|' + '|'.join('-' * (w + 2) for w in col_widths) + '|'
        lines = [fmt_row(headers), sep] + [fmt_row(row) for row in data]
        return '\n'.join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / 'conversations.json'
    stress_path = config.data_dir / 'advanced_long_context.json'
    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    print('\n=== Standard Benchmark (conversations.json) ===\n')
    std_rows: list[BenchmarkRow] = []
    for name, agent in [
        ('Baseline', BaselineAgent(config=config, force_offline=True)),
        ('Advanced', AdvancedAgent(config=config, force_offline=True)),
    ]:
        row = run_agent_benchmark(name, agent, std_convs, config)
        std_rows.append(row)
    print(format_rows(std_rows))

    print('\n\n=== Long-Context Stress Benchmark (advanced_long_context.json) ===\n')
    stress_rows: list[BenchmarkRow] = []
    for name, agent in [
        ('Baseline', BaselineAgent(config=config, force_offline=True)),
        ('Advanced', AdvancedAgent(config=config, force_offline=True)),
    ]:
        row = run_agent_benchmark(name, agent, stress_convs, config)
        stress_rows.append(row)
    print(format_rows(stress_rows))

    print('\n\n=== Analysis ===')
    _print_analysis(std_rows, stress_rows)


def _print_analysis(std: list[BenchmarkRow], stress: list[BenchmarkRow]) -> None:
    base_std = next((r for r in std if r.agent_name == 'Baseline'), None)
    adv_std = next((r for r in std if r.agent_name == 'Advanced'), None)
    base_str = next((r for r in stress if r.agent_name == 'Baseline'), None)
    adv_str = next((r for r in stress if r.agent_name == 'Advanced'), None)

    if base_std and adv_std:
        print(f'\n[Standard] Recall — Baseline: {base_std.recall_score:.2f}, Advanced: {adv_std.recall_score:.2f}')
        print(f'  Advanced nhớ tốt hơn vì User.md lưu facts bền vững qua các session mới.')
        if adv_std.prompt_tokens_processed > base_std.prompt_tokens_processed:
            print(f'  Advanced dùng nhiều prompt tokens hơn ở hội thoại ngắn do overhead của User.md.')
        print(f'  Memory growth (Advanced): {adv_std.memory_growth_bytes} bytes, Compactions: {adv_std.compactions}')

    if base_str and adv_str:
        print(f'\n[Stress] Prompt tokens — Baseline: {base_str.prompt_tokens_processed}, Advanced: {adv_str.prompt_tokens_processed}')
        if adv_str.prompt_tokens_processed < base_str.prompt_tokens_processed:
            saving = base_str.prompt_tokens_processed - adv_str.prompt_tokens_processed
            pct = saving / base_str.prompt_tokens_processed * 100
            print(f'  Compact memory giảm {saving} tokens ({pct:.1f}%) so với baseline ở hội thoại dài.')
        print(f'  Advanced compactions: {adv_str.compactions}')
        print('  Compact tối ưu chủ yếu "Prompt tokens processed", không phải "Agent tokens only".')


if __name__ == '__main__':
    main()
