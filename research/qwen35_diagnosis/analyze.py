"""Recompute the diagnostic evidence and write a short Russian report."""
import hashlib
import json
import re
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / 'architecture_benchmark'

def read(name):
    return json.loads((ROOT / (name + '.json')).read_text(encoding='utf-8'))

def main():
    assert read('completion')['complete']
    protocol = read('protocol')
    for name, expected in protocol['original_inputs'].items():
        assert hashlib.sha256((BENCH / name).read_bytes()).hexdigest() == expected, name
    baseline = {r['id']: r for r in read('panel_baseline')['predictions']}
    controls = {}
    for name in ['batch1', 'long64', 'no_cache']:
        rows = read('control_' + name)['predictions']
        controls[name] = dict(n=len(rows), label_changes=sum(r['prediction'] != baseline[r['id']]['prediction'] for r in rows),
            text_changes=sum(r['output_text'] != baseline[r['id']]['output_text'] for r in rows),
            max_label_logprob_difference=max(abs(v - baseline[r['id']]['first_label_logprob'][k]) for r in rows for k, v in r['first_label_logprob'].items()),
            token_limit=sum(r['hit_token_limit'] for r in rows))
    results = {}
    leading_answers = {}
    for path in ROOT.glob('*.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if 'predictions' not in value:
            continue
        rows = value['predictions']
        # Recompute metrics without importing PyTorch while another experiment may run.
        m = value['metrics']['all']
        assert m['n'] == len(rows)
        assert m['correct'] == sum(r['prediction'] == r['expected'] for r in rows), path
        assert m['unknown_correct'] == sum(r['expected'] == 'unknown' and r['prediction'] == 'unknown' for r in rows)
        results[path.stem] = dict(**m, hit_token_limit=sum(r['hit_token_limit'] for r in rows),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        yesno = [r for r in rows if r.get('kind') == 'yesno']
        if yesno:
            correct = 0
            for r in yesno:
                text = r['output_text']
                if path.stem == 'thinking256':
                    text = text.split('</think>', 1)[1] if '</think>' in text else ''
                match = re.match(r'^\s*(yes|no|true|false)\b', text, re.I)
                correct += bool(match and match[1].lower() == r['expected'])
            leading_answers[path.stem] = dict(n=len(yesno), correct=correct)
    gate = read('validation_evidence_gate')['predictions']
    gate_metrics = dict(unknown_n=sum(r['expected'] == 'unknown' for r in gate),
        unknown_accepted=sum(r['expected'] == 'unknown' and r['gate_prediction'] == 'yes' for r in gate),
        known_n=sum(r['expected'] != 'unknown' for r in gate),
        known_rejected=sum(r['expected'] != 'unknown' and r['gate_prediction'] == 'no' for r in gate))
    evidence = dict(original_inputs_unchanged=True, archived_panel_matches=read('panel_baseline')['matches_archive'],
        panel_n=len(baseline), runtime_controls=controls, validation_gate=gate_metrics,
        selection=read('selection'), results=results, assets=read('asset_verification'), leading_answer_checks=leading_answers)
    (ROOT / 'verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Диагностика Qwen3.5-0.8B', '',
        'Исследовательский разбор после исходного бенчмарка; это не независимая проверка на промышленном потоке.', '',
        'Исходная модель: Qwen/Qwen3.5-0.8B, revision `' + protocol['revision'] + '`, FP32 CPU, 4 потока.',
        'Исходные тесты и их результаты не изменялись. Все новые варианты и ответы сохранены отдельно.', '',
        '## Технические контроли', '',
        f"Повторная генерация совпала с архивом на {evidence['archived_panel_matches']}/{len(baseline)} примерах.",
        'Веса, токенизатор и конфигурация проверены по хешам закреплённого репозитория. '
        'Отдельный официальный chat_template.jinja совпадает со встроенным шаблоном токенизатора.', '',
        '| Контроль | Примеров | Изменилось решений | Изменилось текстов | Максимальное изменение log P метки |',
        '|---|---:|---:|---:|---:|']
    for name, m in controls.items():
        lines.append(f"| {name} | {m['n']} | {m['label_changes']} | {m['text_changes']} | {m['max_label_logprob_difference']:.6g} |")
    lines += ['', 'Это ограниченные контроли. Они не доказывают идентичность всех реализаций на всех входах.', '',
        '## Формулировка и классы', '',
        'Отбор вариантов выполнен на 48 диагностических примерах validation: 12 success, 12 failure, 24 unknown. '
        'Это намеренно повышенная доля unknown для охвата четырёх причин неопределённости. '
        'Два лучших варианта один раз оценены на исходных 432 тестах с равными долями классов.', '',
        '| Вариант | Набор | Верно | Ложные тревоги | Пропуски сбоев, включая unknown | Верно unknown | Нарушения формата |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name, m in results.items():
        if name.startswith(('validation_', 'test_')):
            lines.append(f"| {name} | {m['n']} | {m['correct']}/{m['n']} | {m['false_alarm']}/{m['success_n']} | {m['missed_failure']}/{m['failure_n']} | {m['unknown_correct']}/{m['unknown_n']} | {m['invalid']} |")
    lines += ['', 'evidence_gate — отдельный вызов модели о достаточности свидетельств; '
        'при yes используется исходное решение модели, при no возвращается unknown. '
        'Исходные решения переиспользованы для оценки качества; скорость двух вызовов здесь не измерялась.',
        'label_abc заменяет названия классов буквами; fewshot добавляет три размеченных примера только из train; '
        'definitions явно отличает доказанное нарушение критерия от отсутствия данных; '
        'system_role переносит исходную инструкцию в system; label_order меняет только порядок перечисления ответов.', '',
        '## Равенство чисел, длина и рассуждение', '',
        '| Вариант | Верно строгий формат | Достигнут лимит токенов |', '|---|---:|---:|']
    for name, m in results.items():
        if name.startswith(('equality', 'thinking', 'sampled')):
            lines.append(f"| {name} | {m['correct']}/{m['n']} | {m['hit_token_limit']} |")
    lines += ['', 'Вторичная проверка первого явного yes/no/true/false в вопросах о равенстве '
        '(не проверяет правильность остального объяснения):', '',
        '| Вариант | Верное начало ответа |', '|---|---:|']
    for name, m in leading_answers.items():
        lines.append(f"| {name} | {m['correct']}/{m['n']} |")
    lines += ['', 'В thinking256 оценивается только финальный ответ после </think>. '
        'Если модель не завершила рассуждение, это NO_FINAL, а не неверный семантический ответ. '
        'Ограничение в 256 токенов не является оценкой всей потенциальной способности модели рассуждать.',
        'sampled_* — temperature=1, top_p=1, top_k=20, три фиксированных seed; '
        'это проверка чувствительности к sampling, а не точное воспроизведение всех рекомендуемых настроек серверного движка '
        '(presence_penalty здесь не реализован).', '',
        '## Ограничения вывода', '',
        '- Варианты подсказок выбраны после просмотра исходных ошибок; итог не является независимой оценкой переноса.',
        '- В сравнении меняются способы постановки задачи при неизменных весах. Историческую причину в обучающих данных это не устанавливает.',
        '- Проверена именно маленькая 0.8B-модель; вывод не распространяется на все размеры Qwen3.5.',
        '- Дополнительная информация и few-shot увеличивают стоимость входа. Сравнимый замер задержки этих новых вариантов ещё не выполнен.', '',
        'Воспроизведение: `python research/qwen35_diagnosis/run.py`, затем `python research/qwen35_diagnosis/analyze.py`.',
        '[Протокол](protocol.json), [проверка](verification.json), [лог](run.log).',
        'Официальный источник: https://huggingface.co/Qwen/Qwen3.5-0.8B', '']
    (ROOT / 'findings_ru.md').write_text('\n'.join(lines), encoding='utf-8')
    files = sorted(p for p in ROOT.iterdir() if p.is_file())
    baseline_files = [BENCH / name for name in ['common.py', 'protocol.json', 'train.json', 'validation.json',
        'test.json', 'diagnostic.json', 'Qwen3.5-0.8B.metadata.json', 'results/qwen35_08_test.json']]
    archive = ROOT.parent / 'qwen35_diagnosis_20261107.zip'
    entries = [('qwen35_diagnosis/' + p.name, p) for p in files]
    entries += [('architecture_benchmark/' + str(p.relative_to(BENCH)).replace('\\', '/'), p) for p in baseline_files]
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in entries}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in entries:
            z.write(p, name)
        z.writestr('SHA256.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    print(json.dumps(dict(controls=controls, gate=gate_metrics, selection=read('selection'), results=results, archive_bytes=archive.stat().st_size), indent=2))

if __name__ == '__main__':
    main()
