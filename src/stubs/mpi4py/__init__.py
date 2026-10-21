"""Import-only stub for mpi4py.

The Creamer_LDS_2024 evaluation path (lgssm_utilities -> inference_utilities)
imports mpi4py at module load but never calls it for STAM/correlation
prediction, which is pure numpy. This stub lets the published models be
evaluated without an MPI runtime. Any actual MPI use raises loudly rather
than silently returning a wrong answer.
"""
