"""
Implementation of scatter-gather pattern for client and server usage.
"""

import argparse
import asyncio
import io
import socket
import struct
import sys
from typing import Tuple, Union, Callable, Optional

import numpy as np

try:
    from PIL import Image as PILImage  # Optional dependency for client input
except ImportError:  # Pillow may be absent on the server as it is only needed for some paths.
    PILImage = None
    print("PIL not found!", file=sys.stderr)


LEN_PREFIX_FMT = "!Q"  # 8-byte unsigned big-endian length prefix
LEN_PREFIX_SIZE = struct.calcsize(LEN_PREFIX_FMT)


###############################################################################
#                                                                             #
#                             Client/Helper Code                              #
#                                                                             #
###############################################################################


def _ensure_rgba_numpy(image: Union[np.ndarray, "PILImage.Image"]) -> np.ndarray:
    """
    Convert input to a numpy RGBA array of shape (512, 512, 4), dtype=uint8.

    :param image: PIL image or numpy array.
    :returns: A numpy array with shape (512, 512, 4) and dtype uint8.
    :raises ValueError: if shape or size is invalid.
    """
    if PILImage is not None and hasattr(PILImage, "Image") and isinstance(
        image, PILImage.Image
    ):
        rgba = image.convert("RGBA")
        if rgba.size != (512, 512):
            raise ValueError("Expected a 512x512 image.")
        arr = np.array(rgba, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.shape != (512, 512, 4):
            raise ValueError("Expected numpy array with shape (512, 512, 4).")
        if image.dtype != np.uint8:
            arr = image.astype(np.uint8, copy=False)
        else:
            arr = image
    else:
        raise ValueError("Unsupported image type, provide PIL.Image or numpy.ndarray.")
    return arr


def serialize_ndarray(arr: np.ndarray) -> bytes:
    """
    Serialize a numpy array to .npy bytes without using pickle.

    :param arr: numpy array to serialize
    :returns: bytes of the .npy file
    """
    buf = io.BytesIO()
    # allow_pickle=False ensures only safe .npy headers are used by np.load
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def deserialize_ndarray(data: bytes) -> np.ndarray:
    """
    Deserialize .npy bytes to a numpy array with allow_pickle disabled.

    :param data: serialized .npy bytes
    :returns: numpy array
    """
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


async def _async_send_and_receive(
    host: str,
    port: int,
    payload: bytes,
    *,
    read_exact_chunk: int = 1 << 16,
) -> bytes:
    """
    Async TCP client that sends a length-prefixed payload and returns the length-prefixed reply.

    :param host: server hostname or IP
    :param port: server port
    :param payload: request payload bytes
    :param read_exact_chunk: per-read chunk size
    :returns: response payload bytes
    """
    reader, writer = await asyncio.open_connection(host, port)

    try:
        writer.write(struct.pack(LEN_PREFIX_FMT, len(payload)))
        writer.write(payload)
        await writer.drain()

        # Read response length
        hdr = await reader.readexactly(LEN_PREFIX_SIZE)
        (resp_len,) = struct.unpack(LEN_PREFIX_FMT, hdr)

        # Read exact response body
        remaining = resp_len
        chunks = []
        while remaining:
            to_read = min(read_exact_chunk, remaining)
            chunk = await reader.readexactly(to_read)
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        writer.close()
        # Let the transport close cleanly without blocking the event loop
        try:
            await writer.wait_closed()
        except Exception:
            pass


def parse_server_addr(
    server_addr: Union[Tuple[str, int], str]
) -> Tuple[str, int]:
    """
    Parse server address passed either as (host, port) or 'host:port'.

    :param server_addr: tuple or string
    :returns: (host, port)
    :raises ValueError: if format is invalid
    """
    if isinstance(server_addr, tuple) and len(server_addr) == 2:
        host, port = server_addr
        if not isinstance(host, str) or not isinstance(port, int):
            raise ValueError("Tuple server address must be (str host, int port).")
        return host, port

    if isinstance(server_addr, str):
        if ":" not in server_addr:
            raise ValueError("String server address must be 'host:port'.")
        host, port_s = server_addr.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError as exc:
            raise ValueError("Invalid port number in server address.") from exc
        return host, port

    raise ValueError("Unsupported server address format.")


async def request_patch(
    image: Union[np.ndarray, "PILImage.Image"],
    server_addr: Union[Tuple[str, int], str],
) -> np.ndarray:
    """
    Send a 512x512x4 RGBA array to the server and receive a 512x512 result.

    The function is async, uses asyncio streams, and does not block the caller.

    :param image: input image (PIL Image or numpy array of shape (512, 512, 4))
    :param server_addr: either (host, port) or 'host:port'
    :returns: numpy array with shape (512, 512) (the server's response)
    :raises ValueError: on invalid input or protocol violations
    """
    arr = _ensure_rgba_numpy(image)
    req_bytes = serialize_ndarray(arr)
    host, port = parse_server_addr(server_addr)

    resp_bytes = await _async_send_and_receive(host, port, req_bytes)
    out_arr = deserialize_ndarray(resp_bytes)

    if not isinstance(out_arr, np.ndarray) or out_arr.shape != (512, 512):
        raise ValueError("Server returned unexpected array shape.")
    return out_arr

###############################################################################
#                                                                             #
#                                 Server Code                                 #
#                                                                             #
###############################################################################


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from a blocking socket.

    :param sock: connected socket
    :param n: number of bytes to read
    :returns: bytes buffer of length n
    :raises ConnectionError: if the connection closes prematurely
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data.")
        buf.extend(chunk)
    return bytes(buf)


def _handle_one_request(conn: socket.socket, handle_func: Optional[Callable[[np.ndarray, ...], np.ndarray]] = None) -> None:
    """
    Process exactly one request on a connected socket and send back the reply.

    Protocol:
    - Client sends 8-byte big-endian length, then .npy payload for (512, 512, 4) uint8.
    - Server responds with 8-byte big-endian length, then .npy payload for (512, 512).

    :param conn: connected socket
    :param handle_func: The function to process the numpy array from the client.
    :returns: None, output sent over the socket to the client.
    """
    # Dummy handle function for testing
    if handle_func is None:
        handle_func = lambda x: x[..., 3]
    # Read request header and body
    hdr = recv_exact(conn, LEN_PREFIX_SIZE)
    (req_len,) = struct.unpack(LEN_PREFIX_FMT, hdr)
    data = recv_exact(conn, req_len)

    # Deserialize safely
    arr = deserialize_ndarray(data)

    # Validate and compute result
    if not isinstance(arr, np.ndarray) or arr.shape != (512, 512, 4):
        raise ValueError("Invalid input shape, expected (512, 512, 4).")
    # Select last channel
    result = handle_func(arr)
    if result.shape != (512, 512):
        raise ValueError("Unexpected result shape after channel extraction.")

    # Serialize and send response
    out_bytes = serialize_ndarray(result)
    conn.sendall(struct.pack(LEN_PREFIX_FMT, len(out_bytes)))
    conn.sendall(out_bytes)


def run_server(
        port: int,
        handle_func: Optional[Callable[[np.ndarray, ...], np.ndarray]] = None,
        connection_queue_size: int = 100
) -> None:
    """
    Run a single-threaded TCP server that handles one request at a time.

    :param port: TCP port to bind.
    :param handle_func: Function to handle the processing of the input numpy array.
    :param connection_queue_size: How many connections should be kept in the queue.
    :returns: None
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        # Reuse address to avoid TIME_WAIT bind issues
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(connection_queue_size)  # connection queue of some size
        print(f"[server] listening on 0.0.0.0:{port}", file=sys.stderr)

        while True:
            conn, addr = srv.accept()
            try:
                if __debug__:
                    print(f"[server] connection from {addr}", file=sys.stderr)
                _handle_one_request(conn, handle_func)
                if __debug__:
                    print(f"[server] completed request for {addr}", file=sys.stderr)
            except Exception as exc:
                # Keep it simple and explicit. Client will see a broken pipe on failure.
                print(f"[server] error handling {addr}: {exc!r}", file=sys.stderr)
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                conn.close()


def _cli() -> None:
    """
    Command line entry point for running the server.

    Usage:
        python scatter_gather.py --port 5000
    """
    parser = argparse.ArgumentParser(description="Server.")
    parser.add_argument("--port", type=int, required=True, help="TCP port to listen on.")
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    _cli()
