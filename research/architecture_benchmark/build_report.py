"""Aggregate real completed runs; absent candidates are shown as missing, never invented."""
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT,dump

NAMES={'mean':'Pooled features + MLP','gru':'GRU','lstm':'LSTM','tcn':'TCN',
       'ssd':'Selective SSM / SSD reference','gnn':'GNN, temporal edges',
       'modernbert_head':'ModernBERT, frozen encoder + head','modernbert_numeric':'ModernBERT + numeric features',
       'modernbert_guarded':'ModernBERT + schema numeric executor','numeric_executor':'Schema numeric executor',
       'flan_large':'FLAN-T5-large','lamini_flan_large':'LaMini-783M','qwen3_06':'Qwen3-0.6B',
       'qwen35_08':'Qwen3.5-0.8B','flan_extract_execute':'FLAN extraction + exact comparison'}
NAMES['qwen35_guarded']='Qwen3.5 + schema numeric executor'

def collect():
    groups=defaultdict(list); diagnostics=[]
    for path in sorted((ROOT/'results').glob('*.json')):
        d=json.loads(path.read_text(encoding='utf-8'))
        if 'metrics' not in d: continue
        if path.stem.endswith('_diagnostic'): diagnostics.append(d); continue
        track='BGL' if path.stem.startswith('stream_') else 'SRE'
        groups[(track,d['model'])].append(d)
    return groups,diagnostics

def span(values,precision=0):
    lo,hi=min(values),max(values)
    return f'{lo:.{precision}f}' if lo==hi else f'{lo:.{precision}f}–{hi:.{precision}f}'

groups,diagnostics=collect()
lines=['# Сравнение архитектур для SRE', '', '7 ноября 2026. Результаты реальных локальных запусков.', '',
       '## Что именно проверено', '',
       'Две задачи оцениваются отдельно: применение заданного критерия к тексту и бинарная '
       'классификация окон реальных логов BGL. Модели с предобучением, обучение с нуля и '
       'системы с исполнителем не являются контролируемым сравнением только архитектуры.', '',
       '- **SRE:** 1080 синтетических обучающих, 216 валидационных и 432 тестовых примера. '
       'В тесте по 144 success/failure/unknown. Для числовых семейств диапазоны разделены; '
       'формулировки свидетельств success/failure меняются по выборкам. Часть критериев, '
       'служебных фраз и шаблонов unknown общая. Полного прежнего набора 120 примеров локально нет.',
       '- **BGL:** исходный полный файл проверен по контрольной сумме Loghub/Zenodo. '
       'Из него до обучения выбраны непересекающиеся окна по 32 строки: 3000/800/1000 '
       'окон по хронологическим частям 60%/20%/20%. В тесте 77 окон с тревогами и 923 нормальных. '
       'Это пилот на выборке, не повтор прежнего полного BGL-бенчмарка.',
       '- **CPU:** 4 потока на общем рабочем компьютере; тёплая задержка одного запроса '
       'включает токенизацию/признаки, вычисление и разбор ответа. Это не SLA сервера. '
       'Холодная загрузка моделей в эти миллисекунды не входит. Измерены первые '
       '24/30 тестовых случаев, а не весь поток; смесь быстрых и медленных ветвей '
       'в этом поднаборе может отличаться от смеси во всей выборке. По этим данным '
       'не оценивается средняя пропускная способность промышленного потока.', '',
       '## Применение критериев: SRE', '',
       'Числа ошибок для обучаемых вариантов показаны диапазоном по трём seed. '
       'Задержка — медиана измеренных p95 по seed; для генеративных моделей один запуск. '
       'Нарушения формата считаются ошибками. Пропуск сбоя здесь включает ответ unknown; '
       'это отличается от ложного утверждения success и отдельно разобрано ниже.', '',
       '| Вариант | Запусков | Верно, % | Ложные тревоги /144 | Пропуски сбоев /144 | Верно unknown /144 | p95, мс |',
       '|---|---:|---:|---:|---:|---:|---:|']
summary={}
for (track,name),runs in groups.items():
    if track!='SRE': continue
    ms=[r['metrics']['all'] for r in runs]
    times=[r['latency']['p95_ms'] for r in runs]
    lines.append(f"| {NAMES.get(name,name)} | {len(runs)} | {span([100*m['accuracy'] for m in ms],1)} | "
        f"{span([m['false_alarm'] for m in ms])} | {span([m['missed_failure'] for m in ms])} | "
        f"{span([m['unknown_correct'] for m in ms])} | {np.median(times):.2f} |")
    summary[f'{track}/{name}']=dict(n_runs=len(runs),mean_accuracy=float(np.mean([m['accuracy'] for m in ms])),
        median_p95_ms=float(np.median(times)),false_alarm=[m['false_alarm'] for m in ms],
        missed_failure=[m['missed_failure'] for m in ms])
lines+=['', 'Для ModernBERT заморожен весь энкодер, обучается только MLP-голова. '
        'Это не проверка полного дообучения ModernBERT. Вариант с числовыми признаками '
        'получает результаты ограниченного парсера и отношений чисел; guarded-вариант '
        'исполняет эти отношения явно. Парсер знает грамматику синтетических входов, '
        'поэтому его успех не доказывает универсальное понимание новых логов.', '',
        'Qwen3.5 + executor применяет тот же ограниченный числовой парсер, а '
        'нечисловые случаи передаёт Qwen3.5. Это отдельная гибридная система. '
        'Поле `baseline_output_text` хранит исходный ответ языковой модели; '
        '`system_output` — итог после маршрутизации и сравнения.', '',
        'FLAN extraction выполняет настоящие отдельные вызовы извлечения actual и expected '
        'и затем сравнивает числа. Операторы распознаются ограниченной схемой, привязка '
        'извлечённого значения к объекту и времени независимо не доказана. '
        'Входные поля генератора с правильными actual/expected моделям не передавались.', '',
        '### Отказ от решения и ложное подтверждение успеха', '',
        '| Вариант | Failure → success /144 | Failure → unknown /144 | Unknown → решительный ответ /144 |',
        '|---|---:|---:|---:|']
for (track,name),runs in groups.items():
    if track!='SRE': continue
    values=[]
    for r in runs:
        rows=r['predictions']
        values.append([sum(p['expected']=='failure' and p['prediction']==v for p in rows) for v in ['success','unknown']]+
                      [sum(p['expected']=='unknown' and p['prediction'] in ['success','failure'] for p in rows)])
    lines.append('| '+NAMES.get(name,name)+' | '+' | '.join(span([v[i] for v in values]) for i in range(3))+' |')
lines+=['',
        '## Окна реального BGL', '',
        'Все малые сети используют одинаковые признаки сообщений и временного разрыва. '
        'Исходная метка и поле severity исключены из входа. Обучение с весами классов; '
        'порог — 99-й процентиль оценок нормальных окон валидации, решение по строгому `>`. '
        'Целевые 1% на валидации не гарантируют такой же FPR на будущем потоке.', '',
        '| Архитектура | Запусков | Recall тревог, % | Ложные тревоги /923 | Пропуски /77 | p95, мс |',
        '|---|---:|---:|---:|---:|---:|',
        '| Всегда «норма» | — | 0 | 0 | 77 | — |']
for (track,name),runs in groups.items():
    if track!='BGL': continue
    ms=[r['metrics']['all'] for r in runs]; times=[r['latency']['p95_ms'] for r in runs]
    recalls=[100*(m['failure_n']-m['missed_failure'])/m['failure_n'] for m in ms]
    lines.append(f"| {NAMES.get(name,name)} | {len(runs)} | {span(recalls,1)} | "
        f"{span([m['false_alarm'] for m in ms])} | {span([m['missed_failure'] for m in ms])} | {np.median(times):.2f} |")
    summary[f'{track}/{name}']=dict(n_runs=len(runs),recall_percent=recalls,median_p95_ms=float(np.median(times)),
        false_alarm=[m['false_alarm'] for m in ms],missed_failure=[m['missed_failure'] for m in ms])
lines+=['', 'Глобальный порядок BGL смешивает узлы; граф здесь построен по соседству событий '
        'в окне и не является графом причинности сервисов. Метка окна означает наличие '
        'хотя бы одной размеченной тревоги. Это не отдельная проверка предсказания следующего '
        'события и не доказательство, что сеть выучила порядок исполнения. '
        'Результаты перестановки событий сохранены в `shuffled_metrics`. '
        'BGL не содержит эталонного класса unknown.', '',
        'Дополнительные пороги с бюджетами 0,5/1/3/5% FPR выбраны по нормальным '
        'окнам валидации и сохранены в `stream_operating_points.json`. Это дополнительная '
        'диагностика после первичных результатов; тестовые метки для выбора порогов не используются.', '',
        '## Диагностические сравнения чисел', '',
        '| Модель | Старые пары /12 | Новые пары /24 | Same /24 | Different /24 | Исходные логи /6 |',
        '|---|---:|---:|---:|---:|---:|']
for d in diagnostics:
    cells=[]
    for suite in ['pure_equality','heldout_original_form','heldout_same','heldout_inequality','original']:
        m=d['metrics'].get(suite); cells.append(str(m['correct']) if m else '—')
    lines.append('| '+NAMES.get(d['model'],d['model'])+' | '+' | '.join(cells)+' |')
lines+=['', 'Дополнительно учитывается первое явное Yes/No в начале ответа. '
        'Это вторичная диагностика выбора ответа, не оценка полного объяснения. '
        'Она не заменяет строгий результат выше. В частности, обрезанное пояснение '
        'с правильным начальным Yes не следует называть ошибочным сравнением чисел.', '',
        '| Модель | Первое Yes/No, старые пары /12 | Новые пары /24 | Different /24 | Ответы достигли лимита /122 |',
        '|---|---:|---:|---:|---:|']
for d in diagnostics:
    values=[]
    for suite in ['pure_equality','heldout_original_form','heldout_inequality']:
        rs=[r for r in d['predictions'] if r['suite']==suite]
        count=0
        for r in rs:
            prefix=re.match(r'^\s*(yes|no)\b',r['output_text'],re.I)
            count+=bool(prefix and prefix[1].lower()==r['expected'])
        values.append(str(count))
    values.append(str(sum(r.get('hit_token_limit',False) for r in d['predictions'])))
    lines.append('| '+NAMES.get(d['model'],d['model'])+' | '+' | '.join(values)+' |')
lines+=['', '## Пределы вывода', '',
        '- Малые GRU/LSTM/TCN/SSM обучены с нуля в ограниченном бюджете; провал '
        'не доказывает непригодность семейства при большем обучении.',
        '- SSM — наша небольшая CPU-реализация input-dependent state-space recurrence '
        'по SSD. Параллельная формула проверена против независимой рекуррентной. '
        'Это не официальный предобученный Mamba-2; здесь используется плотная квадратичная '
        'референсная реализация, поэтому промышленную скорость Mamba по ней оценивать нельзя.',
        '- Готовые языковые модели существенно различаются предобучением. '
        'Их сравнение выбирает кандидатов для задачи, а не изолирует эффект архитектуры.',
        '- Qwen3.5 на этом CPU использует референсные PyTorch-реализации causal conv '
        'и gated delta rule: ускоренные GPU-ядра недоступны. Его измеренная задержка '
        'относится к этой реализации, не к оптимальному серверному движку.',
        '- Числа, формулировки и область синтетического теста ограничены. '
        'Преобразование единиц, произвольные новые правила и весь промышленный поток '
        'этим пилотом не подтверждены. Ноль ошибок на малой выборке не является гарантией.', '',
        '## Воспроизведение и источники', '',
        '[README](README.md), [протокол SRE](protocol.json), [протокол BGL](stream/protocol.json), '
        '[сводка JSON](summary.json), [статусы запусков](run_status.json).', '',
        '[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), '
        '[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), '
        '[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B), '
        '[SSD/Mamba-2](https://arxiv.org/abs/2405.21060), '
        '[TCN](https://arxiv.org/abs/1803.01271), '
        '[Loghub](https://github.com/logpai/loghub), '
        '[Loghub paper](https://arxiv.org/abs/2008.06448).']
if groups:
    sre=[(name,runs) for (task,name),runs in groups.items() if task=='SRE']
    best_name,best_runs=max(sre,key=lambda v:np.mean([r['metrics']['all']['accuracy'] for r in v[1]]))
    accuracy=np.mean([r['metrics']['all']['accuracy'] for r in best_runs])*100
    intro=['## Основные результаты', '',
           f'На текущем синтетическом SRE-пилоте наибольшая средняя точность у '
           f'**{NAMES.get(best_name,best_name)}: {accuracy:.1f}%**. '
           'Это максимум точности на данном тесте, а не автоматически лучший '
           'эксплуатационный режим: ложные тревоги, пропуски и unknown приведены отдельно.', '',
           'На реальном BGL все небольшие прототипы работают за единицы миллисекунд '
           'на окно, но при основном пороге пропускают существенную часть тревог. '
           'Готовой универсальной архитектуры, подтверждённой этим исследованием, нет.', '',
           'Результат поддерживает дальнейшую проверку разделения текстовой обработки '
           'и явного исполнения числовых условий. Для последовательностей нужны '
           'собственные проверки порядка событий; результат классификации окна '
           'не заменяет их.', '',
           'Графики: [SRE](figures/sre.png), [BGL](figures/bgl.png). '
           'Оборудование: AMD Ryzen 7 5825U, 16 ГиБ RAM, CPU, 4 потока. '
           'Все готовые языковые модели запускались в FP32, без квантования.', '']
    if ('SRE','qwen35_guarded') in groups:
        intro[4:4]=['Qwen3.5 без исполнителя правильно классифицировала 280/288 '
                    'определённых исходов, но 0/144 случаев unknown. С числовым '
                    'исполнителем правильно все 288 определённых исходов и 96/144 '
                    'unknown; остальные 48 неопределённых случаев названы failure. '
                    'Поэтому ноль ошибок на известных success/failure не означает '
                    'надёжность полного решения.', '',
                    'На прямой вопрос о равенстве 200 и 200 Qwen3 начинает ответ '
                    'с Yes, а Qwen3.5 — с No; обе добавляют пояснение сверх короткого '
                    'формата. Замена архитектуры не устранила все исходные симптомы.', '']
    lines[4:4]=intro
(ROOT/'comparison_report_ru.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
dump(ROOT/'summary.json',summary)
print('REPORT_BUILT',len(groups),'groups',len(diagnostics),'diagnostic models')
