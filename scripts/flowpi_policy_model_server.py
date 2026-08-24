"""FlowPi's low-GIL wrapper around DOMINO's lightweight policy model server.

The wire format remains DOMINO's JSON + ``__numpy_array__`` marker protocol.  Only the parser
and encoder are replaced with msgspec when it is available; this keeps the large RGB request
from holding CPython's GIL long enough to delay a concurrent fast-state response.
"""

from __future__ import annotations

import base64
from contextlib import suppress
import importlib
import pathlib
import socket
import sys
from typing import Any

import numpy as np

_DOMINO_SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "DOMINO" / "script"
if not _DOMINO_SCRIPT_ROOT.is_dir():
    raise FileNotFoundError(f"DOMINO script directory not found: {_DOMINO_SCRIPT_ROOT}")
if str(_DOMINO_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOMINO_SCRIPT_ROOT))

_domino_server = importlib.import_module("policy_model_server")


def _install_low_latency_server_socket() -> None:
    """Set TCP_NODELAY on accepted policy RPC sockets without editing DOMINO."""
    server_type = _domino_server.ModelServer
    original_handler = getattr(server_type, "_flowpi_original_handle_client", None)
    if original_handler is not None:
        return
    original_handler = server_type._handle_client  # noqa: SLF001

    def handle_client(self: Any, client_socket: socket.socket) -> Any:
        with suppress(OSError):
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return original_handler(self, client_socket)

    server_type._flowpi_original_handle_client = original_handler  # noqa: SLF001
    server_type._handle_client = handle_client  # noqa: SLF001


_install_low_latency_server_socket()

try:
    import msgspec
except ImportError:  # pragma: no cover - the RoboTwin environment provides msgspec
    msgspec = None


def _encode_hook(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == np.float32:
            dtype = "float32"
        elif value.dtype == np.float64:
            dtype = "float64"
        elif value.dtype == np.int32:
            dtype = "int32"
        elif value.dtype == np.int64:
            dtype = "int64"
        else:
            dtype = str(value.dtype)
        return {
            "__numpy_array__": True,
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
            "dtype": dtype,
            "shape": value.shape,
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_numpy_markers(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__numpy_array__") is True:
            raw = base64.b64decode(value["data"])
            return np.frombuffer(raw, dtype=value["dtype"]).reshape(tuple(value["shape"]))
        return {key: _decode_numpy_markers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_numpy_markers(item) for item in value]
    return value


if msgspec is not None:
    _encoder = msgspec.json.Encoder(enc_hook=_encode_hook)

    def numpy_to_json(data: Any) -> str:
        return _encoder.encode(data).decode("utf-8")

    def json_to_numpy(json_str: str) -> Any:
        return _decode_numpy_markers(msgspec.json.decode(json_str))

    _domino_server.numpy_to_json = numpy_to_json
    _domino_server.json_to_numpy = json_to_numpy


if __name__ == "__main__":
    _domino_server.main(_domino_server.parse_args_and_config())
