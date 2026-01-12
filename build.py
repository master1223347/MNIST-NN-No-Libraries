'''
!!!IMPORTANT: RUN THIS BEFORE USING!!! THIS LINKS CPP WITH PYTHON
ALSO PIP INSTALL PYBIND11!!!
'''
import subprocess
import sysconfig

EXT_NAME = "neuralbinding"
SRC_FILES = [
    "neuralbinding/binding.cpp",
    "neuralcpp/dense.cpp",
    "neuralcpp/relu.cpp",
    "neuralcpp/loss.cpp",
    "neuralcpp/optimizer.cpp"
]

INCLUDE_FLAGS = "-I neuralcpp $(python3 -m pybind11 --includes)"
OUTPUT_FILE = f"neuralbinding/{EXT_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"

cmd = f"c++ -O3 -Wall -shared -std=c++17 -fPIC {INCLUDE_FLAGS} {' '.join(SRC_FILES)} -o {OUTPUT_FILE} -undefined dynamic_lookup"

print("Running build command:")
print(cmd)
subprocess.check_call(cmd, shell=True)
print(f"Built {OUTPUT_FILE} successfully!")
