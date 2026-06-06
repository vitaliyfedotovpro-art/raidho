# coder-public *(рабочее имя — финализировать перед публикацией)*

Кодер-агент с **компонуемо-эпизодической памятью** на Vector Symbolic
Architecture (VSA) и **provider-pluggable** LLM-бэкендом. Без привязки к
конкретному ассистенту или вендору.

> **Имя — placeholder.** «A-CODE» коллидирует с [Acode](https://acode.app),
> «V-CODE» с VS Code. Окончательное имя выбирается перед первым релизом.

## Зачем

Память агента — не RAG поверх векторной БД, а **структурная**: факты хранятся
как role-binding гипервекторы, эпизоды — через перестановки, идентичность
сущностей — нормализацией строк + таблицей алиасов (не косинусом). Similarity
считается bit-packed popcount'ом: **×32 RAM** против float, при бит-в-бит
идентичном ранкинге.

## Статус

| Слой | Состояние |
|---|---|
| VSA-память (`vsa/`) | ✅ готово, самодостаточно (только `numpy`), 5 регресс-тестов |
| Агентная петля (bash/read/write/list) | 🔜 порт из приватного прототипа, очистка от персон |
| Provider-pluggable бэкенд (Claude default + др.) | 🔜 |
| Auth: OAuth-логин ИЛИ свой API-ключ | 🔜 |
| Режимы: текстовый reasoning + агентный кодинг | 🔜 |

## Установка / тест

```bash
pip install -e .            # ядро (numpy)
pip install -e '.[dev]'     # + pytest
python tests/test_bitpack.py
# или: pytest
```

## Память — кратко

```python
from vsa import VSAMemory

mem = VSAMemory(D=10_000, seed=0, embed_fn=my_embedder)
mem.add_triple("Paris", "capital_of", "France")
mem.query({"subject": "Paris", "relation": "capital_of"}, "object")["answer"]
# → "France"
```

`embed_fn` инъектируется (любой эмбеддер: текст → `np.ndarray`); пакет не тянет
тяжёлых зависимостей. Дефолтный локальный эмбеддер — extra `.[embed]`
(sentence-transformers).

## Лицензия

Dual-license: **AGPL-3.0-or-later** (open source / исследования / non-commercial)
или коммерческая — см. [COMMERCIAL.md](COMMERCIAL.md).
