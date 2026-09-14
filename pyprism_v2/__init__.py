"""
pyprism_v2
==========
Symmetry- and irreducibility-aware partitioning of quantum circuits for
distributed execution.

Pipeline
--------
    random circuit -> Track 1 hypergraph            (symmetry-blind)
                   -> light cone + symmetry + irreducibility
                   -> Track 2 hypergraph
                   -> partition -> CD/CK Pareto front

Both objectives are static -- no statevector, no shots, no backend::

    E(A,B) = sum over crossings of e_i         [ebits, exact distribution]
    K(A,B) = sum over crossings of log gamma_i [knitting, quasiprobability]

Quick start
-----------
    >>> import pyprism_v2 as pp
    >>> layout, gates, adj = pp.random_circuit(20, 40, seed=0)
    >>> hg2 = pp.build_track2(20, gates)
    >>> front, diag = pp.part_prism(20, hg2, pp.PrismConfig())
    >>> pp.knee(front)[:2]
"""
from .gates import GATE_LIB, spec, gamma_of_gate, unitary_of, GateSpec
from .circuits import (Gate, FAMILIES, coupling_map, random_circuit,
                       circuit_symmetry, circuit_stats)
from .lightcone import light_cone_graph, lightcone_stats, centre_lightcone
from .hypergraph import (Net, Hypergraph, build_track1, build_track2,
                         extract_packets)
from .objectives import (net_costs, evaluate, mode_front, scalarised, knee,
                         balance, feasible, CostState, hypervolume, pareto,
                         objective_correlation)
from .io import (save_circuit, load_circuit, save_instance, load_instance,
                 write_manifest, to_qasm, save_qasm, to_qiskit, environment,
                 circuit_hash)
from .partition import REGISTRY, available, Method, PrismConfig, part_prism
from .benchmark import run_instance, sweep, DEFAULT_NS, DEFAULT_KS

__version__ = '2.0.0'

__all__ = [
    'GATE_LIB', 'spec', 'gamma_of_gate', 'unitary_of', 'GateSpec',
    'Gate', 'FAMILIES', 'coupling_map', 'random_circuit', 'circuit_symmetry',
    'circuit_stats', 'light_cone_graph', 'lightcone_stats', 'centre_lightcone',
    'Net', 'Hypergraph', 'build_track1', 'build_track2', 'extract_packets',
    'net_costs', 'evaluate', 'mode_front', 'scalarised', 'knee', 'balance',
    'feasible', 'CostState', 'hypervolume', 'pareto', 'objective_correlation',
    'save_circuit', 'load_circuit', 'save_instance', 'load_instance',
    'write_manifest', 'to_qasm', 'save_qasm', 'to_qiskit', 'environment',
    'circuit_hash',
    'REGISTRY', 'available', 'Method', 'PrismConfig', 'part_prism',
    'run_instance', 'sweep', 'DEFAULT_NS', 'DEFAULT_KS', '__version__',
]
