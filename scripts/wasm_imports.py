"""List the imports of a wasm module and fail if there are any.

A plain wasm32-unknown-unknown module gets no host functions in a browser
unless a binding layer supplies them, so an import there is a function the
crate cannot run without (a clock, a random source)."""

import sys


def leb(b, i):
    v = s = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return v, i


def name(b, i):
    n, i = leb(b, i)
    return b[i : i + n].decode(), i + n


def imports(path):
    b = open(path, "rb").read()
    assert b[:4] == b"\0asm", "not a wasm module"
    i, out = 8, []
    while i < len(b):
        sid = b[i]
        size, i = leb(b, i + 1)
        end = i + size
        if sid == 2:
            n, i = leb(b, i)
            for _ in range(n):
                mod, i = name(b, i)
                field, i = name(b, i)
                kind = b[i]
                i += 1
                if kind == 0:
                    _, i = leb(b, i)
                elif kind == 1:
                    i += 1
                    flags, i = leb(b, i)
                    _, i = leb(b, i)
                    if flags & 1:
                        _, i = leb(b, i)
                elif kind == 2:
                    flags, i = leb(b, i)
                    _, i = leb(b, i)
                    if flags & 1:
                        _, i = leb(b, i)
                elif kind == 3:
                    i += 2
                out.append(f"{mod}.{field}")
        i = end
    return out


if __name__ == "__main__":
    found = imports(sys.argv[1])
    for f in found:
        print("import:", f)
    sys.exit(1 if found else 0)
