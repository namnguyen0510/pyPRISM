"""
pyprism_v2.partition
====================
Method registry and dispatcher.

Every entry declares

    fn        the callable
    kind      'single' (returns a bitmask) or 'multi' (returns a front)
    track     which substrate it is given: 'H1', 'H2' or 'both'
    max_n     largest register the method is run on (some are super-quadratic)
    needs     an optional external package that must be importable

``available()`` filters the registry against what is installed and how big the
instance is, so the driver never has to special-case a missing backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import baselines as B
from . import moo as M
from .prismpp import (PrismPPConfig, part_prismpp,
                       augmented_graph, seed_set, consensus,
                       boundary_polish, MOVE_NAMES)
from .prism import PrismConfig, part_prism

__all__ = ['PrismPPConfig', 'part_prismpp', 'augmented_graph',
           'seed_set', 'consensus', 'boundary_polish', 'MOVE_NAMES',
           'Method', 'REGISTRY', 'available', 'PrismConfig', 'part_prism']


@dataclass
class Method:
    name: str
    fn: Callable
    kind: str = 'single'          # 'single' | 'multi'
    track: str = 'H2'             # 'H1' | 'H2' | 'both'
    max_n: int = 10 ** 9
    needs: str | None = None
    note: str = ''


REGISTRY: dict = {}


def _reg(m: Method):
    REGISTRY[m.name] = m
    return m


# ---- trivial and structural ----------------------------------------------
_reg(Method('Naive', B.part_naive, track='H2', note='first n/2 indices'))
_reg(Method('Random', B.part_random, track='H2', note='best of 32 restarts'))
_reg(Method('Spectral', B.part_spectral, track='both',
            note='Fiedler bisection of the light-cone-augmented expansion'))
_reg(Method('Greedy', B.part_greedy, track='both', max_n=400,
            note='greedy growth from the highest-degree seed'))

# ---- local search ---------------------------------------------------------
# NOTE: Fiduccia-Mattheyses and the epsilon-constraint sweep are implemented
# in `baselines.part_fiduccia_mattheyses` and `moo.epsilon_constraint` but are
# deliberately not registered, so they do not appear in the benchmark.
_reg(Method('KL', B.part_kernighan_lin, track='both', max_n=1024,
            note='Kernighan-Lin pair-swap descent'))

# ---- community detection --------------------------------------------------
_reg(Method('Louvain', B.part_louvain, track='both', max_n=128,
            note='modularity maximisation, then balanced merge'))
_reg(Method('GirvanNewman', B.part_girvan_newman, track='both', max_n=64,
            note='edge-betweenness bisection, O(n^3)'))

# ---- official external packages -------------------------------------------
_reg(Method('METIS', B.part_metis, track='both', needs='pymetis',
            note='PyMetis on the light-cone-augmented clique expansion'))
_reg(Method('KaHyPar', B.part_kahypar, track='both', needs='kahypar',
            note='KaHyPar on the nets, connectivity-1 objective'))

# ---- multi-objective, native ----------------------------------------------
_reg(Method('NSGA-II', M.nsga2, kind='multi', track='H2', max_n=1024))
_reg(Method('SPEA2', M.spea2, kind='multi', track='H2', max_n=1024))
_reg(Method('MOEA/D', M.moead, kind='multi', track='H2', max_n=1024))
_reg(Method('WeightedSum', M.weighted_sum, kind='multi', track='H2',
            note='convex-hull-only by construction'))

# ---- multi-objective, pymoo ----------------------------------------------
for _a, _lbl in (('pymoo_nsga2', 'pymoo/NSGA-II'),
                 ('pymoo_nsga3', 'pymoo/NSGA-III'),
                 ('pymoo_sms', 'pymoo/SMS-EMOA'),
                 ('pymoo_age', 'pymoo/AGE-MOEA'),
                 ('pymoo_moead', 'pymoo/MOEA-D')):
    _reg(Method(_lbl,
                (lambda a: (lambda n, hg, **kw: M.pymoo_run(a, n, hg, **kw)))(_a),
                kind='multi', track='H2', max_n=1024, needs='pymoo'))


def available(n, extra_skip=()):
    """Registry entries runnable at this size with what is installed."""
    import importlib
    out = []
    for name, m in REGISTRY.items():
        if name in extra_skip or n > m.max_n:
            continue
        if m.needs:
            try:
                importlib.import_module(m.needs)
            except Exception:
                continue
        out.append(m)
    return out
