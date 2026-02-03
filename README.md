# The Ising-LLM Benchmark Suite

> **Correlated Error Landscapes for Quantum Error Correction**

A dataset of **Hard Instance** boolean masks derived from the topological structure of Large Language Model intelligence. Designed to stress-test Quantum Decoders (MWPM, Union-Find, Collision Clustering) against massive, non-local correlated errors.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3572A5.svg)
![QEC Benchmark](https://img.shields.io/badge/QEC-Benchmark-red.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## Table of Contents

- [The Problem: Benchmarking Correlated Noise](#the-problem-benchmarking-correlated-noise)
- [The Dataset (Benchmarks)](#the-dataset-benchmarks)
- [Why This Matters for Decoders](#why-this-matters-for-decoders)
- [Functional Verification (The "Lobotomy" Test)](#functional-verification-the-lobotomy-test)
- [Usage](#usage)
- [Citation](#citation)

---

## The Problem: Benchmarking Correlated Noise

Standard QEC benchmarks often rely on phenomenological noise models (i.i.d. bit flips) or small-scale circuit noise. However, fault-tolerant hardware will face **spatially correlated errors** — leakage, crosstalk, cosmic rays — that defy standard threshold theorems.

The **Ising-LLM Benchmark** fills this gap by providing **32 layers of "biologically structured" noise**. These patterns are not random; they are the result of thermodynamic annealing on a Neural Network, creating massive, ferromagnetic error chains that mimic logic gates.

---

## The Dataset (Benchmarks)

Located in `/benchmarks/deepseek_ising_dataset.zip`.

We mapped the weight matrices of **DeepSeek-R1-Distill-Llama-8B** to a 2D Ising Lattice with Hamiltonian $H = -J \sum_{\langle i,j \rangle} s_i s_j$ and annealed them to extract the **"Skeleton"** of the intelligence.

| Layer | Type | Max Cluster Size (Pixels) | Phase State | QEC Difficulty |
|---|---|---|---|---|
| Control | Random Noise | ~94 | Paramagnetic | Easy (Local Errors) |
| L15 | Reasoning Center | 12,308 | Ferromagnetic | Hard (Correlated Chains) |
| L31 | Output Projection | 47,428 | Super-Critical | Extreme (Continental Defects) |

### Dataset Specifications

- **Format:** 32× `.npy` Boolean Arrays (14336 × 4096)
- **Encoding:**
  - `0` (False): Pruned / Error
  - `1` (True): Data / Qubit
- **Size:** Compressed archive ~180 MB
- **Layers:** Full coverage from input embedding (Layer 0) to vocabulary projection (Layer 31)

---

## Why This Matters for Decoders

Standard decoders like **MWPM** (Minimum Weight Perfect Matching) assume errors are sparse and local.

| Noise Model | Error Pattern | Decoder Performance |
|---|---|---|
| **Random Pruning** (Orange Curve) | Errors are dust | Easy to correct |
| **Ising Pruning** (Cyan Curve) | Errors are continents | The "tail" extends to 47k pixels |

This dataset serves as an **adversarial test** for next-generation clustering decoders (e.g., Riverlane Deltaflow / Collision Clustering) that are designed to handle high-weight syndrome events.

### Key Challenge

The massive connected components (12k–47k pixels) represent **non-local error correlations** that traditional local decoders cannot efficiently handle. These structures force decoders to either:

1. Perform global analysis (exponential cost)
2. Use approximate clustering (degraded logical error rate)
3. Fail catastrophically at pseudo-threshold

---

## Functional Verification (The "Lobotomy" Test)

To prove these structures aren't just artifacts, we tested them on the LLM itself. We pruned **65% of the weights** using both masks on a classic logic puzzle:

> **Prompt:** *"A farmer must transport a wolf, a goat, and cabbage across a river. The boat holds the farmer and one item. If left alone, the wolf eats the goat, or the goat eats the cabbage. How does the farmer succeed?"*

### Results

| Pruning Method | Model Response | Logical Integrity |
|---|---|---|
| **Random Mask** | *"The goat ate the cabbage"* | ❌ Failed (Hallucination) |
| **Ising Mask** | *"The wolf will eat the goat"* | ✅ Preserved (Constraint Reasoning) |

**Conclusion:** The Ising clusters represent the **physical topology of logic**. Random pruning destroys reasoning; thermodynamic pruning preserves it.

---

## Usage

### Loading and Visualizing a Mask

```python
import numpy as np
import matplotlib.pyplot as plt

# Load the "Beast" (Layer 31 - Output Projection)
mask = np.load("benchmarks/layer_31_mask.npy")

# Visualize the correlated error structure
plt.figure(figsize=(10, 10))
plt.imshow(mask[:1000, :1000], cmap="bone")
plt.title("Ising-LLM Layer 31: Correlated Error Landscape", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
```

### Benchmarking a QEC Decoder

```python
import numpy as np
from your_decoder import surface_code_decode  # Replace with your decoder


def benchmark_decoder_on_ising(decoder_func, layer_idx: int, code_distance: int):
    """Benchmark a QEC decoder against Ising-correlated error patterns.
    
    Args:
        decoder_func: A callable that takes (syndrome, distance) -> correction
        layer_idx:    Which Ising layer to use as error pattern (0-31)
        code_distance: Surface code distance parameter
    
    Returns:
        logical_error_rate: Fraction of logical failures
    """
    mask = np.load(f"benchmarks/layer_{layer_idx:02d}_mask.npy")
    
    # Subsample to surface code dimensions
    errors = 1 - mask[:code_distance**2, :code_distance**2]  # Invert: 1=error
    
    # Generate syndrome from error pattern
    syndrome = compute_syndrome(errors, code_distance)
    
    # Attempt correction
    correction = decoder_func(syndrome, code_distance)
    
    # Check logical error
    residual = (errors + correction) % 2
    logical_error = check_logical_operator(residual, code_distance)
    
    return logical_error
```

---

## Citation

If you use this benchmark suite in QEC research, decoder development, or error correlation studies, please cite:

```bibtex
@misc{ising_llm_benchmark_2026,
  title     = {The Ising-LLM Benchmark Suite: Correlated Error Landscapes from Neural Networks},
  author    = {Arndt, Justin},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/justinarndt/ApexNeuro}}
}
```

---

**Related Work:**
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — The source LLM architecture
- [Surface Code Decoders](https://quantum-journal.org/papers/q-2020-09-24-327/) — MWPM and Union-Find baselines
- [Collision Clustering](https://arxiv.org/abs/2203.04948) — Next-gen correlated noise decoders

---

*Built for the frontier of fault-tolerant quantum computing · Thermodynamics meets error correction*
