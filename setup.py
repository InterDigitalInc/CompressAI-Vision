# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from setuptools import find_packages, setup

# only way to import "fastentrypoint.py" from this directory:

# The following line is modified by setver.bash
version = "0.0.0"

this_folder = Path(__file__).resolve().parent

path = this_folder / "requirements.txt"
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if path.is_file():
    with path.open("r") as f:
        install_requires = f.read().splitlines()

TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "flake8-comprehensions",
    "isort",
    "mypy",
]


def get_extra_requirements():
    extras_require = {
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "furo", "sphinxcontrib-mermaid"],
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


# exec(open(os.path.join(this_folder, "fastentrypoints.py")).read()) # not needed

# # https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
setup(
    name="compressai-vision",
    version=version,
    # install_requires = [
    #    "PyYAML",
    #    'docutils>=0.3', # # List here the required packages
    # ],
    install_requires=install_requires,  # instead, read from a file (see above)
    packages=find_packages(),  # # includes python code from every directory that has an "__init__.py" file in it.  If no "__init__.py" is found, the directory is omitted.  Other directories / files to be included, are defined in the MANIFEST.in file
    include_package_data=True,  # # conclusion: NEVER forget this : files get included but not installed
    # # "package_data" keyword is a practical joke: use MANIFEST.in instead
    # # WARNING: If you are using namespace packages, automatic package finding does not work, so use this:
    # packages=[
    #    'compressai_vision.subpackage1'
    # ],
    # scripts=[
    #    "bin/somescript"
    # ],
    # # "entry points" get installed into $HOME/.local/bin
    # # https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path
    entry_points={
        "console_scripts": [
            "compressai-vision = compressai_vision.cli.main:main",
        ]
    },
    # metadata for upload to PyPI
    author="CompressAI-Vision team",
    author_email="compressai@interdigital.com",
    description="Evaluation pipelines for Video Compression for Machine Vision on top of CompressAI",
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # project_urls={ # some additional urls
    #    'Tutorial': 'nada
    # },
)
