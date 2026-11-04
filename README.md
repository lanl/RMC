# Renovated Monte Carlo (RMC)

**RMC** is a Python library for sampling from high-dimensional distributions via Monte Carlo-like algorithms as well as advanced AI/ML-based generative models.

**RMC** is part of the FY25 LDRD-ER project "Generative Modeling for Sampling" (20250391ER)

**RMC** has been approved for public release by the Feynman Innovation Center (O#=O4918)

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.


## Documentation
To build code documentation use the sphinx package. Under docs folder execute

```
make clean
make html
```

## Authors and acknowledgment

Authors/developers: Cristina Garcia-Cardona (@crstngc), Pratik Khandagale(@pkhandag), Yen Ting Lin (@dblueeye)

Affiliation: Information Sciences Group (CCS/CAI-3), Los Alamos National Laboratory

### Target evaluation counts

RMC provides a common metric interface for reporting target-oracle
evaluation counts across classical and neural samplers.

```python
from rmc.metrics.evaluation_counts import get_evaluation_counts

counts = get_evaluation_counts(sampler)

training = counts["training"]
sampling = counts["sampling"]
total = counts["total"]

n_density = total["density"]
n_gradient = total["gradient"]
n_hvp = total["hvp"]
```

The returned structure is:

```python
{
    "training": {"density": ..., "gradient": ..., "hvp": ...},
    "sampling": {"density": ..., "gradient": ..., "hvp": ...},
    "total": {"density": ..., "gradient": ..., "hvp": ...},
}
```

Each operation is reported separately:

| Operation | Meaning |
| --- | --- |
| `density` | Target log-density evaluations |
| `gradient` | Target log-density gradient (score) evaluations |
| `hvp` | Target Hessian-vector product evaluations |

Counts represent logical target-oracle evaluations **per evaluation
point**, including batched evaluations. They are not counts of Python
function calls, hardware kernels, FLOPs, or elapsed time.

Reference-distribution computations are excluded. Cached target
evaluations are counted when originally computed, not when reused.

**Sampler conventions**

- **HMC, SMC, SVGD:** expose flat `sampler.evaluation_counts`
  dictionaries. The metric interprets these as sampling costs and
  assigns zero training costs.
- **PIS, DDS, LFIS, AS, ASBS:** expose separate `training` and
  `sampling` dictionaries. The metric preserves this distinction.
- Missing operation keys are interpreted as zero; a missing
  `evaluation_counts` attribute raises an error.
- Counts are cumulative for the sampler object. The metric returns
  an independent snapshot and does not reset counters.
- The `total` field is computed separately for each operation as
  `training + sampling`.

The counts are maintained by sampler instrumentation. Some counters
are incremented at target-oracle call sites; others use algorithmic
accounting or formulas for completed JAX-compiled operations.
PIS/DDS compiled-training accounting requires end-to-end validation.

Density, gradient, and HVP counts remain separate because their
relative computational costs depend on the target implementation.
