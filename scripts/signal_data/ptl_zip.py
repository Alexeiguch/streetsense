"""Read members of the PTL full-resolution zips (their zip64 local-header offsets are
truncated to 32 bits, so zipfile/unzip fail). We take names/sizes from the central
directory and find the real offset by adding multiples of 2**32 until the local header
signature and file name match."""
import struct
import zipfile
import zlib


def entries(path):
    z = zipfile.ZipFile(path)
    with open(path, "rb") as f:
        for zi in z.infolist():
            if zi.is_dir() or zi.filename.startswith("__MACOSX"):
                continue
            for k in range(8):
                off = (zi.header_offset % (1 << 32)) + k * (1 << 32)
                f.seek(off)
                h = f.read(30)
                if len(h) < 30 or h[:4] != b"PK\x03\x04":
                    continue
                n, m = struct.unpack("<HH", h[26:30])
                if f.read(n).decode("utf-8", "replace") != zi.filename:
                    continue
                yield zi, off + 30 + n + m
                break


def read(path, zi, data_off):
    with open(path, "rb") as f:
        f.seek(data_off)
        raw = f.read(zi.compress_size)
    if zi.compress_type == zipfile.ZIP_STORED:
        return raw
    return zlib.decompress(raw, -15)
