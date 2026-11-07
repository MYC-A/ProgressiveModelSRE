# ProgressiveModelSRE: расследование LaMini / FLAN-T5

Исходная переписка: <https://chatgpt.com/share/6aeee973-04a0-83eb-92db-03756f111bc9>.
Текст прочитан через уже открытый Chrome; исходный архив скачан кнопкой в переписке.

Исследование отвечает на вопрос, почему `LaMini-Flan-T5-783M` говорит `No` на
`Are the numbers 200 and 200 equal? Answer yes or no.`

Продолжение — [сравнение архитектур для SRE](research/architecture_benchmark/comparison_report_ru.md):
готовые модели, обученные малые сети, числовые модули и отдельный пилот на
реальных хронологических окнах BGL. Код, протоколы и сырые результаты находятся
в `research/architecture_benchmark/`.

- `research/findings_ru.md` — отчёт с выводами и таблицами вмешательств.
- `research/prior_diagnosis/` — распакованный исходный архив. Все 40 записей его
  `MANIFEST.json` сверены с SHA256 файлов.
- `research/shared_conversation.txt` — снимок последних сообщений, включая
  незавершённое исследование промежуточных состояний.
- `research/shared_conversation_full.txt` — более ранние сообщения и завершённые ответы.
- `research/protocol.json`, `research/probes.frozen.json` — протокол и 122 запроса,
  зафиксированные до получения новых ответов.
- `research/results/` — новые ответы, трассы FP32 и результаты вмешательств.
- `research/analysis.json` — агрегаты и контрасты сравнения чисел.
- `research/evidence_verification.json` — проверка неизменности запросов,
  полноты результатов и восстановления исходных весов после вмешательств.
- `research_results_20261107.zip` — код, отчёт и исходные результаты без весов
  моделей; целостность каждого файла проверена по манифесту SHA256.

Веса сохраняются отдельно, в `A:\ProgressiveModelSRE-cache\models`, чтобы не
заполнять системный диск. Ревизии совпадают с исходным экспериментом:

| Модель | Ревизия |
|---|---|
| MBZUAI/LaMini-Flan-T5-783M | `7a1ff0207cbe75d6a1fcbcc7077ee0b6808ecf9f` |
| google/flan-t5-large | `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` |

Для воспроизведения из этой папки:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r research\requirements.lock.txt requests==2.34.2
.\.venv\Scripts\python.exe research\download_models.py
.\.venv\Scripts\python.exe research\run_pipeline.py
```

Отдельный этап:

```powershell
.\.venv\Scripts\python.exe research\investigate.py --model lamini_flan_large --stage baseline
.\.venv\Scripts\python.exe research\investigate.py --model lamini_flan_large --stage mechanism
.\.venv\Scripts\python.exe research\investigate.py --model lamini_flan_large --stage heads
.\.venv\Scripts\python.exe research\investigate.py --model lamini_flan_large --stage positions
.\.venv\Scripts\python.exe research\extract_donors.py
.\.venv\Scripts\python.exe research\investigate.py --model lamini_flan_large --stage hybrid
.\.venv\Scripts\python.exe research\investigate.py --model flan_large --stage baseline
.\.venv\Scripts\python.exe research\investigate.py --model flan_large --stage hybrid
.\.venv\Scripts\python.exe research\investigate.py --model flan_large --stage controls
```

Номера блоков и голов в результатах начинаются с 1. Номера нейронов — с 0.
Отключение FF сохраняет остаточный поток, пропуская только обновление блока.
Замена `dense` затрагивает `wi_0`, `wi_1`, `wo`; нормализации проверяются отдельно.
Веса на диске не изменяются. После замен проверяются восстановление исходных
хранилищ параметров и совпадение ответов и оценок с исходной моделью.
Перед заменами `run_pipeline.py` автоматически запускает `extract_donors.py`:
он сохраняет отдельные матрицы каждого блока и проверяет точное равенство
исходным тензорам. Это позволяет обойти предел памяти Windows при открытии
двух полных чекпойнтов одновременно.

Промежуточная проекция в словарь и `gradient × activation` — средства поиска
гипотез. Причинные утверждения требуют реального вмешательства и проверки
равных и неравных пар; успех на исходных 12 примерах не равен общему исправлению.
