from setuptools import setup, Extension
import pybind11

# Get the path to pybind11 include directory
import pybind11
pybind11_include = pybind11.get_include()

# Define the C++ extension module
ext_modules = [
  Extension(
    'prioritized_buffer',         # Name of the generated Python module
    ['bindings.cpp'],                         # Source files to compile
    include_dirs=[pybind11_include],          # Include pybind11 headers
    language='c++',                           # Specify C++ language
    extra_compile_args=['-std=c++17','-O3'],
  ),
]

# Use setuptools to compile the extension and install the Python module
setup(
  name='prioritized_buffer',
  version='1.0',
  author='SandSnip3r',
  ext_modules=ext_modules,
  install_requires=['pybind11'],              # Ensure pybind11 is installed
  zip_safe=False,
)
