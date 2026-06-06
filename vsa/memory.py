"""
VSAMemory — полная композиционно-эпизодическая когнитивная память (Phase 3).

Не демо: факты (role-binding) + эпизоды (порядок через permutation) +
нормализация сущностей (грязные варианты → один канон) + персистентность
(переживает сессии). Валидировано Phase 0/1/2 (алгебра держит, grounding
выжил, дискриминатор бьёт косинус на реальном извлечении).

Геометрия (MAP, биполярная, D=10k):
    fact      = bundle( R_subj⊗a(s), R_rel⊗a(r), R_obj⊗a(o) )
    episode   = bundle( ρ⁰a(e₀), ρ¹a(e₁), …, ρⁿa(eₙ) )   (ρ = циклический сдвиг)
    a(concept)= ground(embedding) — SimHash; близкие концепты → близкие атомы.

Эмбеддер инъектируемый (`embed_fn`) — для детерминированных тестов без модели;
по умолчанию sentence-transformers (lazy).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np

from . import core

# Спецбуквы, которые NFKD не раскладывает (это отдельные буквы, не база+знак).
_SCAND = {"ð": "d", "þ": "th", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l"}


def _normalize_surface(s: str) -> str:
    """Ключ идентичности концепта по СТРОКЕ (не по эмбеддингу).

    casefold → скандинавские спецбуквы → NFKD + снятие диакритики → схлоп пробелов.
    «Zürich»=«ZÜRICH »=«zurich» → один ключ. Разные алфавиты не латинизируются
    (кириллица остаётся кириллицей) — кросс-алфавитную идентичность задаёт
    таблица алиасов, а не догадка."""
    s = s.strip().casefold()
    s = "".join(_SCAND.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.split())


class VSAMemory:
    ROLE_NAMES = ("subject", "relation", "object")

    def __init__(
        self,
        D: int = core.DEFAULT_D,
        embedder_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        seed: int = 0,
        normalize_threshold: float = 0.82,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        identity_mode: str = "string",
        aliases_table: dict[str, str] | None = None,
    ) -> None:
        self.D = D
        self._embedder_model = embedder_model
        self._normalize_threshold = float(normalize_threshold)
        # Как решается тождество сущностей: "string" (нормализация + алиасы, безопасно)
        # или "embedding" (legacy: косинус ≥ threshold — плодит ложные слияния).
        self._identity_mode = identity_mode
        self._alias_map = {
            _normalize_surface(k): _normalize_surface(v)
            for k, v in (aliases_table or {}).items()
        }
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._embed_fn = embed_fn
        self._model = None
        self._emb_dim: int | None = None
        self._proj: np.ndarray | None = None

        self._roles = {n: core.random_atoms(1, D, self._rng)[0] for n in self.ROLE_NAMES}

        # Канонический кодбук концептов. _atoms — float ±1 (нужны для bind/
        # bundle/permute); _atom_bits — их bit-pack (для popcount-cleanup).
        self._names: list[str] = []
        self._kinds: list[str] = []          # "entity" | "relation" | "event"
        self._atoms: list[np.ndarray] = []
        self._atom_bits: list[np.ndarray] = []
        self._embs: list[np.ndarray] = []
        self._index: dict[str, int] = {}     # surface (lower) → canonical idx
        self._aliases: dict[int, list[str]] = {}

        # Факты и эпизоды. Факты храним ТОЛЬКО bit-packed (_fact_bits) — это
        # доминирующий по RAM слой (×32 экономии); float ±1 реконструируем по
        # требованию для unbind топ-K (≤5/запрос). Ранкинг идентичен float-версии.
        self._fact_idx: list[tuple[int, int, int]] = []
        self._fact_bits: list[np.ndarray] = []
        self._fact_meta: list[dict] = []
        self._episodes: dict[str, dict] = {}
        self._ep_counter = 0

        # Процедуры (procedural memory). VSA хранит триггер + тело и матчит
        # триггер; ИСПОЛНЯЕТ тело — внешний интерпретатор (procedure_runner.py).
        self._procedures: dict[str, dict] = {}

        # Контракты (constraints). Второй род: НЕ исполняются — правило/тон
        # вшивается в системный промпт, когда триггер активен. Триггер:
        # always (всегда) | predicate | semantic (как у процедур).
        self._constraints: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Эмбеддинг / grounding
    # ------------------------------------------------------------------
    def _ensure_proj(self, emb_dim: int) -> None:
        if self._proj is None:
            self._emb_dim = emb_dim
            self._proj = core.make_projection(emb_dim, self.D, self._rng)

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            e = np.asarray(self._embed_fn(text), dtype=np.float32)
            n = np.linalg.norm(e)
            e = e / n if n > 0 else e
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._embedder_model)
            e = self._model.encode(text, normalize_embeddings=True).astype(np.float32)
        self._ensure_proj(e.shape[0])
        return e

    # ------------------------------------------------------------------
    # Кодбук с нормализацией сущностей
    # ------------------------------------------------------------------
    def _concept_index(self, concept: str, kind: str) -> int:
        """Канонический индекс концепта.

        Идентичность сущностей решается СТРОКОЙ (нормализация + таблица алиасов),
        а не эмбеддингом: косинус кодирует смысловую близость, а не тождество
        референта (напр. «Paris»↔«France»=0.84 > «Paris»↔«Texas»=0.64 — порог
        бессилен). Эмбеддинг оставлен только для recall (search/grounding).
        Старое поведение — identity_mode='embedding'. Bias: не уверены → НОВЫЙ
        концепт (дубль безвреден, ложное слияние — тихая порча фактов)."""
        key = _normalize_surface(concept)
        key = self._alias_map.get(key, key)          # декларативные алиасы
        if key in self._index:
            return self._index[key]
        emb = self._embed(concept)

        # Legacy: тождество по эмбеддингу. Off by default; НИКОГДА для событий
        # (реплики-эпизоды — не концепты для дедупа).
        if self._identity_mode == "embedding" and kind != "event":
            same = [i for i, k in enumerate(self._kinds) if k == kind]
            if same:
                E = np.stack([self._embs[i] for i in same])
                sims = E @ emb
                j = int(np.argmax(sims))
                if float(sims[j]) >= self._normalize_threshold:
                    canon = same[j]
                    self._index[key] = canon
                    if key != _normalize_surface(self._names[canon]):
                        self._aliases.setdefault(canon, []).append(concept)
                    return canon

        idx = len(self._names)
        self._names.append(concept)
        self._kinds.append(kind)
        atom = core.ground(emb, self._proj)
        self._atoms.append(atom)
        self._atom_bits.append(core.pack_bipolar(atom))
        self._embs.append(emb)
        self._index[key] = idx
        return idx

    def _cleanup(self, vec: np.ndarray, kind: str) -> tuple[str, float, int]:
        idxs = [i for i, k in enumerate(self._kinds) if k == kind]
        if not idxs:
            return "", 0.0, -1
        cb = np.stack([self._atom_bits[i] for i in idxs])
        sims = core.hamming_cosine(cb, core.pack_bipolar(vec), self.D)
        b = int(np.argmax(sims))
        return self._names[idxs[b]], float(sims[b]), idxs[b]

    # ------------------------------------------------------------------
    # Факты
    # ------------------------------------------------------------------
    def add_triple(self, subject: str, relation: str, obj: str, meta: dict | None = None) -> int:
        si = self._concept_index(subject, "entity")
        ri = self._concept_index(relation, "relation")
        oi = self._concept_index(obj, "entity")
        if (si, ri, oi) in self._fact_idx:           # дедуп: тот же триплет не дублируем
            return self._fact_idx.index((si, ri, oi))
        fact = core.bundle(np.stack([
            core.bind(self._roles["subject"], self._atoms[si]),
            core.bind(self._roles["relation"], self._atoms[ri]),
            core.bind(self._roles["object"], self._atoms[oi]),
        ]), self._rng)
        self._fact_idx.append((si, ri, oi))
        self._fact_bits.append(core.pack_bipolar(fact))   # храним только упаковку
        self._fact_meta.append(meta or {})
        return len(self._fact_bits) - 1

    def query(self, known: dict[str, str], target_role: str) -> dict:
        """known: {role: concept}; возвращает восстановленный концепт target_role
        + извлечённый триплет. Различает (X,r,Y) и (Y,r,X) по РОЛЯМ."""
        if not self._fact_bits:
            return {"answer": None, "score": 0.0, "triple": None, "fact_idx": -1}
        terms = []
        for role, concept in known.items():
            kind = "relation" if role == "relation" else "entity"
            ci = self._concept_index(concept, kind)
            terms.append(core.bind(self._roles[role], self._atoms[ci]))
        probe = core.bundle(np.stack(terms), self._rng)

        # Backtracking: проверяем топ-K ближайших фактов. Similarity — popcount по
        # упакованным фактам (= (mem @ probe)/D на ±1, ранкинг идентичен).
        sims_facts = core.hamming_cosine(
            np.stack(self._fact_bits), core.pack_bipolar(probe), self.D)
        # Карантинные факты не участвуют в структурном recall (маскируем).
        for i in range(len(sims_facts)):
            if self._is_quarantined(i):
                sims_facts[i] = -np.inf
        top_k = min(5, len(self._fact_bits))
        best_fidxs = np.argsort(-sims_facts)[:top_k]
        
        kind = "relation" if target_role == "relation" else "entity"
        
        best_result = {"answer": None, "score": -1.0, "triple": None, "fact_idx": -1, "_mq": -1.0}
        
        for fidx in best_fidxs:
            fidx = int(fidx)
            if not np.isfinite(sims_facts[fidx]):   # карантинный — пропускаем
                continue
            fact_vec = core.unpack_bipolar(self._fact_bits[fidx], self.D)  # ±1 из упаковки
            unbound = core.unbind(fact_vec, self._roles[target_role])
            name, score, _ = self._cleanup(unbound, kind)
            
            # Учитываем и релевантность факта запросу, и чистоту извлечения.
            # Клампим негативы: anti-correlated факт (sims<0) или грязное
            # извлечение (score<0) — не «успех». Без клампа neg×neg давал бы
            # ложно-высокое match_quality и мог обойти честное совпадение.
            match_quality = max(0.0, float(sims_facts[fidx])) * max(0.0, score)
            
            if match_quality > best_result["_mq"]:
                si, ri, oi = self._fact_idx[fidx]
                best_result = {
                    "answer": name,
                    "score": score,
                    "triple": (self._names[si], self._names[ri], self._names[oi]),
                    "fact_idx": fidx,
                    "_mq": match_quality
                }
            
            # Для 3-term bundle математическое матожидание score ≈ 0.5.
            # Если извлекли со score > 0.35 из релевантного факта — это успех, дальше не ищем.
            if score > 0.35 and sims_facts[fidx] > 0.2:
                break
                
        best_result.pop("_mq", None)
        return best_result

    def search(self, query: str, top_k: int = 8, include_quarantined: bool = False) -> list[dict]:
        """Similarity-recall по фактам (свободный запрос): эмбеддинг запроса vs
        эмбеддинг факта (= нормированная сумма эмбеддингов его концептов).
        Возвращает [{'triple': (s,r,o), 'score': cos, 'fact_idx': i, 'quarantined': bool}],
        отсортировано по убыванию. Это «ANN-слой снизу» — дополняет структурный query().

        Карантинные факты («забудь об этом») по умолчанию НЕ всплывают;
        include_quarantined=True нужен только команде возврата/листинга."""
        if not self._fact_idx:
            return []
        active = [
            i for i in range(len(self._fact_idx))
            if include_quarantined or not self._is_quarantined(i)
        ]
        if not active:
            return []
        qe = self._embed(query)
        rows = np.stack([
            self._embs[si] + self._embs[ri] + self._embs[oi]
            for (si, ri, oi) in (self._fact_idx[i] for i in active)
        ])
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        sims = rows @ qe
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "triple": (
                    self._names[self._fact_idx[active[o]][0]],
                    self._names[self._fact_idx[active[o]][1]],
                    self._names[self._fact_idx[active[o]][2]],
                ),
                "score": float(sims[o]),
                "fact_idx": active[o],
                "quarantined": self._is_quarantined(active[o]),
            }
            for o in order
        ]

    # ------------------------------------------------------------------
    # Карантин фактов («забудь об этом» — мягкое забывание без удаления)
    # ------------------------------------------------------------------
    def _is_quarantined(self, i: int) -> bool:
        m = self._fact_meta[i] if 0 <= i < len(self._fact_meta) else None
        return bool(m) and bool(m.get("quarantined"))

    def quarantine(self, fact_indices, reason: str = "") -> int:
        """Пометить факты как карантинные (флаг в _fact_meta, переживает сохранение).
        Факт остаётся в памяти, но не всплывает в recall. Возвращает число новых."""
        n = 0
        for i in fact_indices:
            if 0 <= i < len(self._fact_meta):
                meta = dict(self._fact_meta[i] or {})
                if not meta.get("quarantined"):
                    n += 1
                meta.update(quarantined=True, quarantine_reason=reason,
                            quarantine_ts=time.time())
                self._fact_meta[i] = meta
        return n

    def unquarantine(self, fact_indices) -> int:
        """Снять карантин — факт снова всплывает в recall. Возвращает число снятых."""
        n = 0
        for i in fact_indices:
            if 0 <= i < len(self._fact_meta) and (self._fact_meta[i] or {}).get("quarantined"):
                meta = dict(self._fact_meta[i])
                for k in ("quarantined", "quarantine_reason", "quarantine_ts"):
                    meta.pop(k, None)
                self._fact_meta[i] = meta
                n += 1
        return n

    def quarantined(self) -> list[dict]:
        """Всё, что сейчас в карантине: [{'fact_idx','triple','reason','ts'}]."""
        out = []
        for i, m in enumerate(self._fact_meta):
            if m and m.get("quarantined"):
                si, ri, oi = self._fact_idx[i]
                out.append({
                    "fact_idx": i,
                    "triple": (self._names[si], self._names[ri], self._names[oi]),
                    "reason": m.get("quarantine_reason", ""),
                    "ts": m.get("quarantine_ts"),
                })
        return out

    # ------------------------------------------------------------------
    # Эпизоды (порядок через permutation)
    # ------------------------------------------------------------------
    def add_episode(self, items: list[str], episode_id: str | None = None) -> str:
        idxs = [self._concept_index(it, "event") for it in items]
        vec = core.bundle(
            np.stack([core.permute(self._atoms[i], pos) for pos, i in enumerate(idxs)]),
            self._rng,
        )
        eid = episode_id or f"ep{self._ep_counter}"
        self._ep_counter += 1
        self._episodes[eid] = {"item_idx": idxs, "vec": vec}
        return eid

    def recall_at(self, episode_id: str, pos: int) -> str:
        ep = self._episodes[episode_id]
        if pos < 0 or pos >= len(ep["item_idx"]):
            return ""
        return self._cleanup(core.unpermute(ep["vec"], pos), "event")[0]

    def episode_order(self, episode_id: str) -> list[str]:
        ep = self._episodes[episode_id]
        return [self.recall_at(episode_id, p) for p in range(len(ep["item_idx"]))]

    def episode_items(self, episode_id: str) -> list[str]:
        """Точный список элементов эпизода (из кодбука, без lossy-recall)."""
        ep = self._episodes.get(episode_id)
        if not ep:
            return []
        return [self._names[i] for i in ep["item_idx"]]

    def successor(self, episode_id: str, item: str) -> str | None:
        ep = self._episodes[episode_id]
        n = len(ep["item_idx"])
        a = self._atoms[self._concept_index(item, "event")]
        sims = [(core.unpermute(ep["vec"], p) @ a) / self.D for p in range(n)]
        pos = int(np.argmax(sims))
        return self.recall_at(episode_id, pos + 1) if pos + 1 < n else None

    # ------------------------------------------------------------------
    # Процедуры (procedural memory)
    #
    # Роль VSA — ПОИСК, не исполнение: хранит процедуру и матчит «какая
    # подходит под текущую ситуацию». Тело — структурированная программа
    # (опкоды/аргументы/ветви/регистры) — лежит как dict, НЕ в гипервекторе
    # (ветвь и runtime-аргумент permutation-bundle закодировать не может).
    # Исполняет тело отдельный интерпретатор (procedure_runner.py).
    #
    # Триггер двух родов:
    #   predicate — точный паттерн (regex по тексту контекста), без VSA;
    #   semantic  — нечёткий матч по ЯКОРЯМ-примерам: каждый якорь кладётся в
    #               кодбук как концепт kind="trigger" (персистится), score =
    #               МАКСИМУМ косинуса контекста к любому якорю.
    #
    # Почему якоря-примеры, а не одно описание: на живом тексте короткая реплика
    # («надо переделать этот класс») — это ПРИМЕР ситуации, не парафраз
    # абстрактного описания, и косинус к описанию проваливается до ~0. Несколько
    # живых якорей резко поднимают recall (prototype-matching). Триггер задаёт
    # либо "examples": [...] (предпочтительно), либо "situation": <текст>
    # (трактуется как один якорь — обратная совместимость).
    # ------------------------------------------------------------------
    @staticmethod
    def _trigger_anchors(trigger: dict) -> list[str]:
        ex = trigger.get("examples")
        if ex:
            return list(ex)
        sit = trigger.get("situation")
        return [sit] if sit else []

    def add_procedure(self, proc_id: str, trigger: dict, body: dict,
                      meta: dict | None = None) -> str:
        """Сохранить процедуру. trigger = {"type":"predicate","pattern":<regex>}
        или {"type":"semantic","examples":[...]} (или legacy "situation":<текст>).
        body хранится как есть. Возвращает proc_id."""
        ttype = trigger.get("type")
        trigger_idxs: list[int] = []
        if ttype == "semantic":
            anchors = self._trigger_anchors(trigger)
            if not anchors:
                raise ValueError("semantic trigger: нужен 'examples' или 'situation'")
            trigger_idxs = [self._concept_index(a, "trigger") for a in anchors]
        elif ttype == "predicate":
            re.compile(trigger["pattern"])  # битый паттерн — падаем сразу, не в рантайме
        else:
            raise ValueError(f"unknown trigger type: {ttype!r}")
        self._procedures[proc_id] = {
            "trigger": trigger,
            "trigger_idxs": trigger_idxs,
            "body": body,
            "meta": meta or {},
        }
        return proc_id

    def match_trigger(self, context: str, threshold: float = 0.45,
                      top_k: int = 3) -> list[dict]:
        """Какие процедуры подходят под контекст. Предикаты — regex (score 1.0);
        семантика — МАКС косинус эмбеддинга контекста к якорям триггера (score),
        отсекается по threshold. Возвращает [{'proc_id','score','type'}] по
        убыванию score."""
        hits: list[dict] = []
        sem = []
        for pid, p in self._procedures.items():
            t = p["trigger"]
            if t.get("type") == "predicate":
                if re.search(t["pattern"], context):
                    hits.append({"proc_id": pid, "score": 1.0, "type": "predicate"})
            elif t.get("type") == "semantic":
                sem.append((pid, p))
        if sem:
            qe = self._embed(context)
            for pid, p in sem:
                # обратная совместимость: старый ключ trigger_idx (один) → список
                idxs = p.get("trigger_idxs")
                if idxs is None:
                    one = p.get("trigger_idx")
                    idxs = [one] if one is not None else []
                if not idxs:
                    continue
                score = max(float(self._embs[i] @ qe) for i in idxs)  # макс по якорям
                if score >= threshold:
                    hits.append({"proc_id": pid, "score": score, "type": "semantic"})
        hits.sort(key=lambda h: -h["score"])
        return hits[:top_k]

    def get_procedure(self, proc_id: str) -> dict | None:
        """Тело + meta процедуры (для интерпретатора). None если нет."""
        p = self._procedures.get(proc_id)
        if not p:
            return None
        return {"id": proc_id, "trigger": p["trigger"],
                "body": p["body"], "meta": p["meta"]}

    @property
    def procedures(self) -> list[str]:
        return list(self._procedures.keys())

    # ------------------------------------------------------------------
    # Контракты (constraints) — второй род процедурной памяти
    #
    # Procedure ИСПОЛНЯЕТСЯ (шаги). Constraint НЕ исполняется — его rule
    # вшивается в системный промпт, когда триггер активен. Это «как себя
    # вести», а не «что сделать»: тон (no-sycophancy), осторожность
    # (verify-before-claim), протокол (say-back). Триггер:
    #   always    — правило действует на каждом ходу;
    #   predicate — regex по контексту;
    #   semantic  — max косинус к якорям (как у процедур).
    # ------------------------------------------------------------------
    def add_constraint(self, cid: str, trigger: dict, rule: str,
                       meta: dict | None = None) -> str:
        """Сохранить контракт. trigger = {"type":"always"} |
        {"type":"predicate","pattern":..} | {"type":"semantic","examples":[..]}.
        rule — текст, вшиваемый в системный промпт. Возвращает cid."""
        ttype = trigger.get("type")
        trigger_idxs: list[int] = []
        if ttype == "always":
            pass
        elif ttype == "semantic":
            anchors = self._trigger_anchors(trigger)
            if not anchors:
                raise ValueError("semantic constraint: нужен 'examples' или 'situation'")
            trigger_idxs = [self._concept_index(a, "trigger") for a in anchors]
        elif ttype == "predicate":
            re.compile(trigger["pattern"])
        else:
            raise ValueError(f"unknown constraint trigger type: {ttype!r}")
        self._constraints[cid] = {
            "trigger": trigger,
            "trigger_idxs": trigger_idxs,
            "rule": rule,
            "meta": meta or {},
        }
        return cid

    def match_constraints(self, context: str, threshold: float = 0.45) -> list[dict]:
        """Активные контракты для текущего контекста: always (score 1.0) +
        predicate (regex) + semantic (max косинус ≥ threshold).
        Возвращает [{'id','rule','score','type'}] по убыванию score."""
        out: list[dict] = []
        sem = []
        for cid, c in self._constraints.items():
            t = c["trigger"]
            tt = t.get("type")
            if tt == "always":
                out.append({"id": cid, "rule": c["rule"], "score": 1.0, "type": "always"})
            elif tt == "predicate":
                if re.search(t["pattern"], context):
                    out.append({"id": cid, "rule": c["rule"], "score": 1.0, "type": "predicate"})
            elif tt == "semantic":
                sem.append((cid, c))
        if sem:
            qe = self._embed(context)
            for cid, c in sem:
                idxs = c.get("trigger_idxs") or []
                if not idxs:
                    continue
                score = max(float(self._embs[i] @ qe) for i in idxs)
                if score >= threshold:
                    out.append({"id": cid, "rule": c["rule"], "score": score, "type": "semantic"})
        out.sort(key=lambda h: -h["score"])
        return out

    @property
    def constraints(self) -> list[str]:
        return list(self._constraints.keys())

    # ------------------------------------------------------------------
    # Персистентность (память переживает сессии)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        # proj — фиксированная матрица emb_dim×D (~15 МБ, gaussian → плохо жмётся).
        # Раньше она писалась в .npz при КАЖДОМ save (т.е. каждую реплику бота) —
        # тяжёлый I/O без нужды, т.к. proj неизменна после первого embedding.
        # Теперь пишем её ОДИН раз в sidecar .proj.npy; основной .npz держит только
        # лёгкие roles/embs (~сотни КБ). Загрузка остаётся обратно-совместимой:
        # старый .npz с ключом proj внутри читается как прежде (см. load).
        if self._proj is not None and self._proj.size > 0:
            proj_path = base.with_suffix(".proj.npy")
            need_write = True
            if proj_path.exists():
                try:
                    if tuple(np.load(proj_path, mmap_mode="r").shape) == tuple(self._proj.shape):
                        need_write = False
                except Exception:
                    need_write = True
            if need_write:
                np.save(proj_path, self._proj)
        np.savez_compressed(
            base.with_suffix(".npz"),
            roles=np.stack([self._roles[n] for n in self.ROLE_NAMES]).astype(np.int8),
            embs=(np.stack(self._embs) if self._embs
                  else np.zeros((0, self._emb_dim or 1), np.float32)),
        )
        meta = {
            "D": self.D, "seed": self._seed, "emb_dim": self._emb_dim,
            "embedder_model": self._embedder_model,
            "normalize_threshold": self._normalize_threshold,
            "identity_mode": self._identity_mode,
            "alias_map": self._alias_map,
            "role_names": list(self.ROLE_NAMES),
            "names": self._names, "kinds": self._kinds,
            "index": self._index,
            "aliases": {str(k): v for k, v in self._aliases.items()},
            "fact_idx": self._fact_idx, "fact_meta": self._fact_meta,
            "episodes": {k: v["item_idx"] for k, v in self._episodes.items()},
            "ep_counter": self._ep_counter,
            # Процедуры JSON-safe целиком (trigger/trigger_idxs/body/meta) — тело
            # не в гипервекторе, так что сохраняется и грузится как есть.
            "procedures": self._procedures,
            "constraints": self._constraints,  # контракты тоже JSON-safe целиком
        }
        base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, embed_fn: Callable[[str], np.ndarray] | None = None) -> "VSAMemory":
        base = Path(path)
        arr = np.load(base.with_suffix(".npz"))
        meta = json.loads(base.with_suffix(".json").read_text())

        m = cls(D=meta["D"], embedder_model=meta.get("embedder_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                seed=meta["seed"], normalize_threshold=meta["normalize_threshold"],
                embed_fn=embed_fn,
                identity_mode=meta.get("identity_mode", "string"))
        m._alias_map = dict(meta.get("alias_map", {}))
        m._emb_dim = meta["emb_dim"]
        # proj: новый формат — sidecar .proj.npy; старый — ключ proj в .npz.
        proj_path = base.with_suffix(".proj.npy")
        if proj_path.exists():
            m._proj = np.load(proj_path)
        elif "proj" in arr.files:
            m._proj = arr["proj"]
        else:
            m._proj = None
        if m._proj is not None and m._proj.size == 0:
            m._proj = None  # пустой плейсхолдер старого формата → None (как при init)
        roles = arr["roles"].astype(np.float32)
        m._roles = {n: roles[i] for i, n in enumerate(meta["role_names"])}
        m._names = list(meta["names"])
        m._kinds = list(meta["kinds"])
        m._index = {k: int(v) for k, v in meta["index"].items()}
        m._aliases = {int(k): v for k, v in meta["aliases"].items()}
        embs = arr["embs"]
        m._embs = [embs[i] for i in range(len(m._names))]
        m._atoms = [core.ground(e, m._proj) for e in m._embs]  # детерминированно
        m._atom_bits = [core.pack_bipolar(a) for a in m._atoms]

        m._fact_idx = [tuple(t) for t in meta["fact_idx"]]
        m._fact_meta = meta["fact_meta"]
        m._fact_bits = [
            core.pack_bipolar(core.bundle(np.stack([
                core.bind(m._roles["subject"], m._atoms[si]),
                core.bind(m._roles["relation"], m._atoms[ri]),
                core.bind(m._roles["object"], m._atoms[oi]),
            ]), m._rng))
            for (si, ri, oi) in m._fact_idx
        ]
        m._episodes = {}
        for eid, idxs in meta["episodes"].items():
            idxs = list(idxs)
            vec = core.bundle(
                np.stack([core.permute(m._atoms[i], pos) for pos, i in enumerate(idxs)]),
                m._rng,
            )
            m._episodes[eid] = {"item_idx": idxs, "vec": vec}
        m._ep_counter = meta["ep_counter"]
        # Процедуры: .get для обратной совместимости со старыми снапшотами без ключа.
        m._procedures = meta.get("procedures", {})
        m._constraints = meta.get("constraints", {})  # .get для обратной совместимости
        return m

    # ------------------------------------------------------------------
    @property
    def n_facts(self) -> int:
        return len(self._fact_bits)

    @property
    def n_concepts(self) -> int:
        return len(self._names)

    @property
    def triples(self) -> list[tuple[str, str, str]]:
        return [(self._names[s], self._names[r], self._names[o]) for s, r, o in self._fact_idx]

    def aliases_of(self, concept: str) -> list[str]:
        key = concept.strip().lower()
        idx = self._index.get(key)
        return self._aliases.get(idx, []) if idx is not None else []

    def __repr__(self) -> str:
        return (f"VSAMemory(D={self.D}, facts={self.n_facts}, "
                f"concepts={self.n_concepts}, episodes={len(self._episodes)})")
