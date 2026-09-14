# pyprism_v2

Symmetry- and irreducibility-aware partitioning of quantum circuits for
distributed (multi-QPU) execution.

```
random quantum circuit
      │
      ▼
  Track 1 hypergraph            one net per gate, worst-case weights
      │                         (what a symmetry-blind partitioner sees)
      ▼
  light cone  +  symmetry  +  irreducibility
      │
      ▼
  Track 2 hypergraph            packets + cut-resolved, charge-graded weights
      │
      ▼
  partition  →  CD/CK Pareto front
```

## Two objectives

```
E(A,B) = Σ over crossings of e_i          [ebits, exact distribution]
K(A,B) = Σ over crossings of log γ_i      [knitting, quasiprobability]
```

## The two mechanisms

**Symmetry** — `zsupp(g) = {j : [U_g, Z_j] = 0}`. A maximal run of gates rooted
at a qubit `v`, all with `v ∈ zsupp`, executes with one cat-entangler
broadcast: one ebit however many gates it holds. The run breaks exactly where
the on-site symmetry at `v` breaks. A packet net is charged by the
connectivity−1 metric, which is exactly KaHyPar's native objective — that
correspondence is the argument for nets over edges.

**Irreducibility** — `χ` is the operator Schmidt rank across the local cut;
under U(1)_N it is graded by charge transfer, `χ = Σ_δ χ_δ`. From it
`e = ⌈log₂ χ⌉` and `γ`.

`SWAP` is the instructive case: it *is* U(1)_N-covariant and has a full charge
grading, yet `zsupp = ∅`, so it can never be packed. U(1) symmetry gives the
grading; on-site Σ_L symmetry gives the packing. Only the latter buys ebits.

## Install

```bash
pip install -e .                 # core: numpy only
pip install -e ".[all]"          # + pandas, matplotlib, PyMetis, kahypar, pymoo
```

`PyMetis`, `kahypar` and `pymoo` are optional throughout. Methods needing a
missing package are skipped, and everything else still runs.

**KaHyPar needs a config file that the pip wheel does not ship.** If it is
installed but unconfigured the run reports `KaHyPar (installed, unusable)` and
skips it. To enable it:

```bash
curl -LO https://raw.githubusercontent.com/kahypar/kahypar/master/config/km1_kKaHyPar_sea20.ini
python3 run_benchmark.py --kahypar-ini km1_kKaHyPar_sea20.ini
```

`$KAHYPAR_CONFIG` and a `.ini` in the working directory are both picked up
automatically. With KaHyPar unavailable, PRISM+ multi-starts from METIS alone,
or from KL/Spectral/Greedy if METIS is missing too — the run notes say which.

## Benchmark

```bash
python3 run_benchmark.py                       # n ∈ {20,50,100,200}, k ∈ {2,3,4}
python3 run_benchmark.py --quick               # fast smoke run
python3 run_benchmark.py --n 20,50 --k 2 --seeds 5
python3 run_benchmark.py --skip GirvanNewman,SPEA2 --workers 4
```

Depth scales with the register, `depth = k · n`. Every method optimises on the
substrate it is given (`H1` or `H2`) and every method is **scored** on the exact
Track 2 cost, because that is what a distributed compiler pays. Fronts are
compared by hypervolume against a per-instance reference point.


### Reproducibility

```python
import pyprism_v2 as pp
layout, gates, n, meta = pp.load_instance('study_out/circuits', 'n20_k2_s0')
hg2 = pp.build_track2(n, gates)          # identical to the original run
qc  = pp.to_qiskit(gates, n)             # a QuantumCircuit
```

Or from Qiskit directly:

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit.from_qasm_file('study_out/circuits/n20_k2_s0.qasm')
```

## Layout

```
pyprism_v2/
├── gates.py        gate library, zsupp, χ, charge grading, γ closed forms
├── circuits.py     coupling maps, circuit families, random generation
├── lightcone.py    causal-cone propagation and augmentation
├── hypergraph.py   Net, Hypergraph, Track 1/2 builders, packet extraction
├── objectives.py   E, K, mode front, CostState incremental evaluator
├── partition/
│   ├── baselines.py  single-objective partitioners
│   ├── prism.py      the 2-D replica ladder
│   ├── moo.py        Pareto optimisers, native and pymoo
│   └── __init__.py   method registry and availability filter
├── benchmark.py    run_instance, sweep
├── study.py        statistics and the full study writer
├── plotting.py     figure helpers and the palette
└── cli.py          pyprism-benchmark entry point
run_benchmark.py    the sweep driver
```

### Study output

The run writes a complete study into `--outdir`:

```
study_out/
├── REPORT.md            summary and test tables in markdown
├── summary.csv          per-method aggregate
├── raw.csv              one row per (instance, method)
├── instances.csv        circuit, substrate and light-cone metadata
├── pivot_E.csv          instances × methods, ebit cost
├── pivot_hv.csv         instances × methods, normalised hypervolume
├── stats_tests.csv      Wilcoxon / sign / Cliff's δ vs PRISM+, Holm-adjusted
├── study.json           everything machine-readable
├── manifest.json        provenance: environment, arguments, instance hashes
├── circuits/            every benchmarked circuit, saved
│   ├── n20_k2_s0.json.gz    lossless native format
│   └── n20_k2_s0.qasm       OpenQASM 2.0
└── figures/             17 individual PDFs
    fig_01_front_quality          violin + box of normalised hypervolume
    fig_02_ebits_relative         E / E_Naive
    fig_03_runtime                median wall-clock per method
    fig_04_scaling_n              E vs n, log-log
    fig_05_scaling_depth          E vs k
    fig_06_win_rank               win rate with mean rank
    fig_07_critical_difference    Friedman + Nemenyi CD diagram
    fig_08_effect_size            Cliff's δ vs each baseline
    fig_09_win_tie_loss           paired outcomes
    fig_10_performance_profile    Dolan–Moré
    fig_11_quality_vs_cost        hypervolume vs runtime frontier
    fig_12_heatmap                per-instance quality
    fig_13_substrate              net compression, H1 vs H2 size
    fig_14_lightcone              scrambling depth, coupling vs packing
    fig_15_prism_vs_prismplus     the seeding ablation
    fig_16_front_and_correlation  front richness, ρ(E,K)
    fig_17_fronts_<instance>      example CD/CK Pareto fronts
```
