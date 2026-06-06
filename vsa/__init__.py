"""VSA — компонуемо-эпизодическая память на Vector Symbolic Architecture.

Биполярная MAP-модель (bind/bundle/permute), факты как role-binding, эпизоды
через перестановки, semantic-триггеры. Similarity считается через bit-packed
popcount (×32 RAM против float, ранкинг идентичен). Главный класс — VSAMemory.

Эмбеддер инъектируется (embed_fn) — пакет не тянет тяжёлых зависимостей.
"""

from .memory import VSAMemory

__all__ = ["VSAMemory"]
