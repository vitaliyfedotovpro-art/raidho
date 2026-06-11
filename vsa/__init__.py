"""VSA — compositional-episodic memory on a Vector Symbolic Architecture.

Bipolar MAP model (bind/bundle/permute): facts as role-binding, episodes via
permutations, semantic triggers. Similarity is computed with bit-packed popcount
(×32 RAM vs float, ranking identical). The main class is VSAMemory.

The embedder is injectable (embed_fn) — the package pulls in no heavy deps.
"""

from .memory import VSAMemory
from .state_memory import StateMemory

__all__ = ["VSAMemory", "StateMemory"]
