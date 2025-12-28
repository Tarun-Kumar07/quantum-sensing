import os

def pytest_addoption(parser):
    parser.addoption(
        "--num-threads",
        action="store",
        default="1",
        help="Number of threads to use for OMP_NUM_THREADS"
    )

def pytest_configure(config):
    num_threads = config.getoption("--num-threads")
    if num_threads:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
