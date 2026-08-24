"""Process-isolated SEA-RAFT worker used by FlowPi inference.

The policy process owns JAX slow/fast replicas and their Python schedulers. SEA-RAFT is kept in
this small Torch-only child so CUDA2 and Torch's host-side work cannot stall the state RPC.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np

from openpi.training.sea_raft import SeaRaftFlowExtractor


def process_main(
    ckpt_path: str | None,
    variant: str,
    iters: int | None,
    device: str,
    allow_random_init: bool,  # noqa: FBT001
    precision: str,
    request_queue: Any,
    result_queue: Any,
    shared_prev: Any,
    shared_curr: Any,
    shared_output: Any,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    """Load SEA-RAFT once and serve one latest-frame request at a time.

    The image tensors stay in shared ctypes buffers.  Passing the 480x640 frame pair through a
    multiprocessing.Queue would pickle roughly 11 MiB on every request and adds a second copy
    after the JAX/Torch process split.  The queue therefore carries only a request id; the small
    output is copied into a shared buffer for the same reason.
    """
    try:
        extractor = SeaRaftFlowExtractor(
            ckpt_path=ckpt_path,
            variant=variant,
            iters=iters,
            device=device,
            allow_random_init=allow_random_init,
            precision=precision,
        )
        prev = np.ndarray(input_shape, dtype=np.uint8, buffer=shared_prev)
        curr = np.ndarray(input_shape, dtype=np.uint8, buffer=shared_curr)
        output = np.ndarray(output_shape, dtype=np.float32, buffer=shared_output)
        while True:
            item = request_queue.get()
            if item is None:
                return
            request_id = int(item)
            try:
                started_at = time.perf_counter()
                flow = extractor.compute(prev, curr)
                if flow.shape != output_shape:
                    raise RuntimeError(
                        f"SEA-RAFT returned {flow.shape}, expected shared output {output_shape}"
                    )
                np.copyto(output, flow, casting="no")
                result_queue.put(
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "compute_ms": (time.perf_counter() - started_at) * 1000,
                    }
                )
            except BaseException as exc:
                result_queue.put(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return
    except BaseException as exc:
        with contextlib.suppress(Exception):
            result_queue.put({"kind": "fatal", "error": f"{type(exc).__name__}: {exc}"})
