'''
!!!IMPORTANT: RUN THIS BEFORE USING!!! THIS LINKS CPP WITH PYTHON
ALSO PIP INSTALL PYBIND11!!! ("b-b-but you said no libraries" syfm)
'''
#btw THESE libraries are here so you don't have to manually bind py and cpp, 
#this file isn't even needed for the project it can be run from terminal
import subprocess
import shlex
import sys
import sysconfig

EXT_NAME = "neuralbinding"
SRC_FILES = [
    "neuralbinding/binding.cpp",
    "neuralcpp/dense.cpp",
    "neuralcpp/relu.cpp",
    "neuralcpp/loss.cpp",
    "neuralcpp/optimizer.cpp",
    "neuralcpp/math_utils.cpp"
]

pybind_includes = subprocess.check_output(
    [sys.executable, "-m", "pybind11", "--includes"],
    text=True
).strip()
INCLUDE_FLAGS = f"-I neuralcpp {pybind_includes}"
OUTPUT_FILE = f"neuralbinding/{EXT_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"

cmd = (
    "c++ -O3 -Wall -shared -std=c++17 -fPIC "
    f"{INCLUDE_FLAGS} {' '.join(shlex.quote(src) for src in SRC_FILES)} "
    f"-o {shlex.quote(OUTPUT_FILE)} -undefined dynamic_lookup"
)

print("Running build command:")
print(cmd)
subprocess.check_call(cmd, shell=True)
print(f"Built {OUTPUT_FILE} successfully!")