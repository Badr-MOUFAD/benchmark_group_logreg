from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

benchmark_logreg = Benchmark('./')

run_benchmark(benchmark_logreg, max_runs=25,
              n_jobs=1, n_repetitions=1,
              solver_names=["skglm"],
              dataset_names=['simulated'])
