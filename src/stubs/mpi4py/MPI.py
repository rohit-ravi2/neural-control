def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "mpi4py is stubbed: this process has no MPI runtime. "
        "Model fitting requires a real mpi4py install; evaluation does not."
    )


class _Comm:
    def __getattr__(self, name):
        return _unavailable


COMM_WORLD = _Comm()
