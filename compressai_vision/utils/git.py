# Copyright (c) 2022-2024, InterDigital Communications, Inc
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


import os
import subprocess
from shlex import quote
from typing import Iterable, Optional


def branch_name(rev: str = "HEAD", root: str = ".") -> str:
    cmd = f"git -C {quote(root)} rev-parse --abbrev-ref {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def common_ancestor_hash(
    rev1: str = "HEAD", rev2: Optional[str] = None, root: str = "."
) -> str:
    if rev2 is None:
        rev2 = main_branch_name(root=root)
    if branch_name(rev1, root=root) == branch_name(rev2, root=root):
        return commit_hash(rev=rev1, root=root)
    cmd = (
        "diff -u "
        f"<(git -C {quote(root)} rev-list --first-parent {quote(rev1)}) "
        f"<(git -C {quote(root)} rev-list --first-parent {quote(rev2)}) | "
        "sed -ne 's/^ //p' | head -1"
    )
    cmd_args = ["bash", "-c", cmd]
    result = subprocess.run(cmd_args, capture_output=True, check=True).stdout
    return result.decode("utf-8").rstrip()


def commit_hash(rev: str = "HEAD", short: bool = False, root: str = ".") -> str:
    options = "--short" if short else ""
    cmd = f"git -C {quote(root)} rev-parse {options} {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def diff(rev: str = "HEAD", root: str = ".") -> str:
    cmd = f"git -C {quote(root)} --no-pager diff --no-color {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def main_branch_name(
    root: str = ".", candidates: Iterable[str] = ("main", "master")
) -> str:
    r"""Returns name of primary branch (main or master)."""
    candidates_str = " ".join(quote(x) for x in candidates)
    cmd = f"git -C {quote(root)} branch -l {candidates_str}"
    lines = os.popen(cmd).read().rstrip().splitlines()
    lines = [_removeprefix(x, "* ").strip() for x in lines]
    assert len(lines) == 1
    return lines[0]


def _removeprefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s
