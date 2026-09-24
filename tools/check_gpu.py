from __future__ import annotations

def main():
    try:
        import cupy as cp
    except Exception as exc:
        print("CUPY: NOT AVAILABLE")
        print(exc)
        raise SystemExit(1)

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        print(f"CUDA devices: {count}")
        if count < 1:
            raise SystemExit(2)

        for i in range(count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props.get("name", b"CUDA GPU")
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            total = int(props.get("totalGlobalMem", 0))
            print(f"[{i}] {name} | VRAM: {total / 1024**3:.2f} GiB")

        cp.cuda.Device(0).use()
        x = cp.arange(1_000_000, dtype=cp.float32)
        y = cp.sum(cp.sin(x) * 0.5)
        cp.cuda.Stream.null.synchronize()
        print(f"CuPy test: OK | result={float(y.get()):.3f}")
        print("Mucha can use backend = \"auto\" or \"cuda\".")
    except Exception as exc:
        print("CUDA TEST FAILED")
        print(repr(exc))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
