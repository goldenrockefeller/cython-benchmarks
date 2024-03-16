import timeit
import glob
from pathlib import Path
from importlib import import_module

import pyximport; pyximport.install()

files = glob.glob('benchmarks/*/*.py')
for fnam in files:
    filepath_wo_ext = Path(fnam).parent / Path(fnam).stem
    benchmark_module = import_module((".").join(filepath_wo_ext.parts))

    cython_files = glob.glob(str(Path(fnam).parent) + "/*.pyx")

    for cython_fnam in cython_files:
        bench_name = Path(cython_fnam).stem
        filepath_wo_ext = Path(cython_fnam).parent / bench_name
        benchmark_module = import_module((".").join(filepath_wo_ext.parts))

files = glob.glob('benchmarks/*/*.py')
for fnam in files:
    filepath_wo_ext = Path(fnam).parent / Path(fnam).stem
    benchmark_module = import_module((".").join(filepath_wo_ext.parts))

    print("...")
    print(Path(fnam).stem)
    print()

    for item in dir(benchmark_module):
        if item.startswith("bm_"):
            benchmark_function = benchmark_module.__getattribute__ (item)
    base_time = timeit.timeit("benchmark_function()", globals=locals(), number=1)

    print(f"baseline: {base_time}")

    cython_files = glob.glob(str(Path(fnam).parent) + "/*.pyx")

    for cython_fnam in cython_files:
        bench_name = Path(cython_fnam).stem
        filepath_wo_ext = Path(cython_fnam).parent / bench_name
        benchmark_module = import_module((".").join(filepath_wo_ext.parts))

        for item in dir(benchmark_module):
            if item.startswith("bm_"):
                benchmark_function = benchmark_module.__getattribute__ (item)
        bench_time = timeit.timeit("benchmark_function()", globals=locals(), number=1)
        print(f"{bench_name}: {bench_time}, {base_time/bench_time:.2f}x")


