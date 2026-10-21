from .. import MPI  # noqa: F401


def __getattr__(name):
    raise RuntimeError(f"mpi4py.util.pkl5.{name} is stubbed; no MPI runtime available.")
