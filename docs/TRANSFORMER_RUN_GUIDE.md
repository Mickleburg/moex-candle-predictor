# Инструкция: запуск Transformer эксперимента на отдельном PC (Windows)

## Что это

Эксперимент сравнивает Transformer архитектуру с лучшей на данный момент моделью —
LSTM v2 (walk-forward macro-F1=0.4778). Трансформер может дать прирост за счёт
self-attention: он динамически выбирает, какие из 32 последних свечей важны для
каждого предсказания.

Результат записывается в JSON файл. Его нужно отправить владельцу репозитория.

---

## Требования к железу

- RAM: минимум 8 GB (рекомендуется 16 GB)
- CPU: любой современный, чем больше ядер — тем быстрее
- Время выполнения: **~60-90 минут** (4 фолда × 3 seed × ~5 мин/run на типичном CPU)
- GPU не нужен — PyTorch CPU build

---

## Шаг 1. Проверь Python

Нужен Python **3.11 или 3.12** (рекомендуется). Проверь:

```powershell
python --version
```

Если Python не установлен или версия старше 3.10 — скачай с https://python.org/downloads
и установи версию 3.12.x (отмечь галочку "Add Python to PATH").

---

## Шаг 2. Установи Git

Проверь:
```powershell
git --version
```

Если не установлен: https://git-scm.com/download/win

---

## Шаг 3. Клонируй репозиторий (только чтение)

```powershell
git clone https://github.com/Mickleburg/moex-candle-predictor.git
cd moex-candle-predictor
```

Это клонирует репо без прав на запись в облако. Все изменения останутся локально.

---

## Шаг 4. Переключись на рабочую ветку

```powershell
git checkout ml-expirement
git status
```

Должно показать `On branch ml-expirement`, без изменений.

---

## Шаг 5. Создай виртуальное окружение

**Важно**: на корпоративных PC Windows AppLocker может блокировать DLL-файлы Python-пакетов
внутри папки проекта. Чтобы избежать этой ошибки, создавай venv в домашней папке пользователя.

**PowerShell:**
```powershell
python -m venv $env:USERPROFILE\ml-venv-moex
$env:USERPROFILE\ml-venv-moex\Scripts\Activate.ps1
```

Если PowerShell блокирует выполнение скриптов:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
$env:USERPROFILE\ml-venv-moex\Scripts\Activate.ps1
```

**CMD (Command Prompt):**
```cmd
python -m venv %USERPROFILE%\ml-venv-moex
%USERPROFILE%\ml-venv-moex\Scripts\activate.bat
```

> **Альтернатива** (если и `%USERPROFILE%` заблокирован): попробуй `C:\Temp\ml-venv-moex` или
> обратись к системному администратору за разрешением на запуск Python-расширений (.pyd).

---

## Шаг 6. Установи зависимости

```powershell
# Базовые зависимости
pip install -r ml/requirements_research.txt

# PyTorch CPU (важно: ставить отдельно с PyTorch-индекса)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Проверка
python -c "import torch; print('torch', torch.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
```

Ожидаемый вывод:
```
torch 2.x.x+cpu
sklearn 1.x.x
pandas 2.x.x
```

---

## Шаг 7. Скачай данные SBER

```powershell
python scripts/download_candles.py --ticker SBER --timeframe 1H --from 2020-01-01
```

Скрипт скачивает ~25000 свечей SBER 1H с MOEX ISS (бесплатно, без авторизации).
Займёт 1-2 минуты. Данные сохранятся в `data/raw/`.

Проверь:
```powershell
python -c "
import sys; sys.path.insert(0,'ml')
from src.data.load import load_candles
df = load_candles('data/raw', ticker='SBER', timeframe='1H')
print('Загружено свечей:', len(df))
print('Диапазон:', df['begin'].min(), '-', df['begin'].max())
"
```

Ожидаемый вывод:
```
Загружено свечей: 25061
Диапазон: 2020-01-03 ... 2026-06-01
```

---

## Шаг 8. Запусти эксперимент

**PowerShell:**
```powershell
$env:PYTHONIOENCODING = "utf-8"
python ml/scripts/sber_transformer_research.py
```

**CMD:**
```cmd
set PYTHONIOENCODING=utf-8
python ml/scripts/sber_transformer_research.py
```

Прогресс будет выглядеть так:
```
============================================================
Transformer experiment — SBER H1 triple-barrier
PyTorch 2.x.x+cpu  |  Device: cpu
Output: ml/docs/research/sber_h1_transformer_results_20260603_143022.json
============================================================

Loading SBER 1H data...
  25061 candles, first 21301 rows used (test excluded)
Building per-step features...
  Feature matrix: (25061, 14)

Walk-forward (4 folds x 3 seeds)...
  fold 1: 11968 train seqs, 1968 val seqs
    seed=7: macro=0.XXXX  ...
    ...
  fold 4: ...

============================
RESULTS
============================
  Transformer macro-F1: 0.XXXX +- 0.XXXX  (worst=0.XXXX)
  LSTM v2 baseline:     0.4778
  Delta vs LSTM:        +/-0.XXXX
  Conf > 0.50:          X.X%
  Total time:           XX.X min

Results saved to: ml/docs/research/sber_h1_transformer_results_YYYYMMDD_HHMMSS.json
Send this file to the project maintainer for analysis.
```

---

## Шаг 9. Найди и отправь результат

Результат сохраняется в:
```
moex-candle-predictor\ml\docs\research\sber_h1_transformer_results_YYYYMMDD_HHMMSS.json
```

Имя файла содержит дату и время запуска. Найди его:
```powershell
Get-ChildItem ml\docs\research\sber_h1_transformer_results_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

Отправь этот JSON файл владельцу репозитория.

---

## Что делать если что-то пошло не так

### ImportError: DLL load failed — "Политика управления приложениями заблокировала"

Windows AppLocker/WDAC блокирует Cython-расширения scikit-learn в папке проекта.
Создай venv в домашней папке — AppLocker обычно разрешает выполнение оттуда:

**PowerShell:**
```powershell
deactivate
python -m venv $env:USERPROFILE\ml-venv-moex
$env:USERPROFILE\ml-venv-moex\Scripts\Activate.ps1
pip install -r ml/requirements_research.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CMD:**
```cmd
deactivate
python -m venv %USERPROFILE%\ml-venv-moex
%USERPROFILE%\ml-venv-moex\Scripts\activate.bat
pip install -r ml/requirements_research.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Если и это не помогло — попробуй `C:\Temp\ml-venv-moex` или обратись к администратору.

### PowerShell запрещает скрипты
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip не находит torch
```powershell
# Убедись что venv активирован (в начале строки должно быть (venv-win))
ml\.venv-win\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### ModuleNotFoundError: No module named 'src'
```powershell
# Убедись что запускаешь из корня репозитория, не из ml/
cd moex-candle-predictor
python ml/scripts/sber_transformer_research.py
```

### UnicodeEncodeError при выводе
```powershell
$env:PYTHONIOENCODING = "utf-8"
python ml/scripts/sber_transformer_research.py
```

### Скачивание свечей зависло
Нажми Ctrl+C, подожди минуту, запусти снова. MOEX ISS иногда медленно отвечает.

### Не хватает памяти (MemoryError)
Можно уменьшить BATCH_SIZE в скрипте:
```python
# В начале sber_transformer_research.py найди и измени:
BATCH_SIZE = 128  # было 256
```

---

## Структура результирующего JSON

```json
{
  "experiment": "sber_h1_transformer",
  "timestamp": "20260603_143022",
  "system": { "python": "...", "torch": "...", "platform": "...", "cpu_count": N },
  "config": { ... все гиперпараметры ... },
  "baselines": {
    "et_wf_macro_f1": 0.4738,
    "lstm_v2_wf_macro_f1": 0.4778,
    "lstm_v2_conf050_sharpe": 6.38
  },
  "aggregate": {
    "mean_macro_f1": X.XXXX,
    "std_macro_f1": X.XXXX,
    "worst_fold_f1": X.XXXX,
    "mean_sell_f1": X.XXXX,
    "mean_hold_f1": X.XXXX,
    "mean_buy_f1": X.XXXX,
    "mean_conf_gt_050": X.XX,  // ключевая метрика: % сигналов с conf>0.50
    "delta_vs_lstm": +/-X.XXXX
  },
  "fold_records": [ ... детальные результаты по каждому fold+seed ... ],
  "total_training_seconds": XXXX.X
}
```

Ключевые числа для интерпретации:
- `mean_macro_f1` > 0.4778 → Transformer лучше LSTM
- `mean_conf_gt_050` > 0.013 (1.3%) → больше торговых сигналов
- `delta_vs_lstm` — прямое сравнение

---

## Быстрая проверка что всё правильно установлено

```powershell
python -c "
import sys; sys.path.insert(0,'ml')
import torch
from src.models.lstm_model import build_per_step_features
from ml.scripts.sber_transformer_research import CandleTransformer
import pandas as pd, numpy as np

# Проверка модели
m = CandleTransformer()
x = torch.randn(2, 32, 14)
out = m(x)
print('Model OK, output shape:', out.shape)
print('Parameters:', sum(p.numel() for p in m.parameters()))
print('All checks passed. Ready to run.')
"
```

Если видишь `All checks passed. Ready to run.` — можно запускать.
