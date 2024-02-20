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

import concurrent.futures as cf
import multiprocessing
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional


def get_max_num_cpus():
    # return multiprocessing.cpu_count()
    # This number is not equivalent to the number of CPUs the current process can use.
    # Please see https://docs.python.org/3/library/multiprocessing.html
    try:
        # only available on some Unix platforms
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = multiprocessing.cpu_count()
    return num_cpus


def prevent_core_dump():
    # set no core dump at all
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_cmdlines_parallel(cmds: List[Any], logpath: Optional[Path] = None) -> None:
    def worker(cmd, id, logpath):
        print(f"--> job_id [{id:03d}] Running: {' '.join(cmd)}", file=sys.stderr)
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=prevent_core_dump,
        )

        if logpath is not None:
            plogpath = Path(str(logpath) + f".sub_p{id}")
            with plogpath.open("w") as f:
                for bline in p.stdout:
                    line = bline.decode()
                    f.write(line)
                f.flush()
            assert p.wait() == 0
        else:
            p.stdout.read()  # clear up

    with cf.ThreadPoolExecutor(get_max_num_cpus()) as exec:
        all_jobs = [
            exec.submit(worker, cmd, id, logpath) for id, cmd in enumerate(cmds)
        ]
        cf.wait(all_jobs)

    return


def run_cmdline(cmdline: List[Any], logpath: Optional[Path] = None) -> None:
    print(f"--> Running: {' '.join(cmdline)}", file=sys.stderr)

    if logpath is None:
        out = subprocess.check_output(cmdline).decode()
        if out:
            print(out)
        return

    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with logpath.open("w") as f:
        if p.stdout is not None:
            for bline in p.stdout:
                line = bline.decode()
                f.write(line)
    p.wait()
