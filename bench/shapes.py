"""Single source of truth for the shape sweep.

Every backend module imports from here so all paths benchmark identical configs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterShape:
    name: str
    hidden: int
    experts: int

    @property
    def linear_shape(self) -> tuple[int, int]:
        return (self.hidden, self.experts)


@dataclass(frozen=True)
class ExpertShape:
    name: str
    hidden: int
    ffn: int

    @property
    def up_proj(self) -> tuple[int, int]:
        return (self.hidden, self.ffn)

    @property
    def down_proj(self) -> tuple[int, int]:
        return (self.ffn, self.hidden)


@dataclass(frozen=True)
class BlockShape:
    name: str
    hidden: int
    ffn: int
    experts: int
    top_k: int


ROUTER_SHAPES: list[RouterShape] = [
    RouterShape("mixtral-8x7b", 4096, 8),
    RouterShape("mixtral-8x22b", 6144, 8),
    RouterShape("qwen2-moe", 2048, 60),
    RouterShape("deepseek-moe", 2048, 66),
    RouterShape("olmoe", 2048, 64),
    RouterShape("stress", 8192, 256),
    RouterShape("router_stress_dense", 4096, 4096),
]

EXPERT_SHAPES: list[ExpertShape] = [
    ExpertShape("mixtral-8x7b", 4096, 14336),
    ExpertShape("qwen2-moe", 2048, 1408),
    ExpertShape("deepseek-moe", 2048, 1408),
    ExpertShape("olmoe", 2048, 1024),
]

BLOCK_SHAPES: list[BlockShape] = [
    BlockShape("mixtral-8x7b", hidden=4096, ffn=14336, experts=8, top_k=2),
    BlockShape("olmoe", hidden=2048, ffn=1024, experts=64, top_k=8),
]

BATCH_SIZES: list[int] = [1, 2, 4, 8, 32, 128, 512]


def router_by_name(name: str) -> RouterShape:
    for s in ROUTER_SHAPES:
        if s.name == name:
            return s
    raise KeyError(name)


def expert_by_name(name: str) -> ExpertShape:
    for s in EXPERT_SHAPES:
        if s.name == name:
            return s
    raise KeyError(name)


def block_by_name(name: str) -> BlockShape:
    for s in BLOCK_SHAPES:
        if s.name == name:
            return s
    raise KeyError(name)
