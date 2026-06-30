"""
NorPix / StreamPix .seq file reader.

Reads raw 8-bit grayscale or JPEG-compressed .seq video files (a format
produced by NorPix StreamPix recording software).  Supports optional
seek-table acceleration via .mat sidecar files.
"""
from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# .seq image format codes
FRAME_FORMAT_RAW_GRAY = 100
FRAME_FORMAT_JPEG_GRAY = 102
FRAME_FORMAT_RAW_COLOR = 200
FRAME_FORMAT_JPEG_COLOR = 201


class SeqReader:
    """Read frames from a NorPix .seq file."""

    def __init__(self, filename):
        self.filename = str(filename)
        self.file = open(self.filename, "rb")
        self.header = {}
        self.timestamp_length = 10
        self.seek_table = None
        self._walk_retried = False
        self._parse_header()

    def _parse_header(self):
        data = self.file.read(1024)
        (
            self.header["magic_number"],
            self.header["name"],
            self.header["version"],
            self.header["header_size"],
            self.header["description"],
            self.header["image_width"],
            self.header["image_height"],
            self.header["bit_depth"],
            self.header["bit_depth_real"],
            self.header["image_size"],
            self.header["image_format"],
            self.header["allocated_frames"],
            self.header["origin"],
            self.header["true_image_size"],
            self.header["frame_rate"],
            self.header["description_format"],
            self.header["padding"],
        ) = struct.unpack("i24sii512sIIIIIiIIIdi428s", data)
        self.image_format = self.header["image_format"]
        if self.header["image_format"] == 101:
            self.header["image_format"] = 102
        self.bit_depth = self.header["bit_depth"]
        self.compressed = self.image_format in (FRAME_FORMAT_JPEG_GRAY, FRAME_FORMAT_JPEG_COLOR)
        if not self.compressed:
            expected = self.bit_depth / 8 * (self.header["image_height"] * self.header["image_width"]) + self.timestamp_length
            if expected != self.header["true_image_size"]:
                self.timestamp_length = int(
                    self.header["true_image_size"]
                    - (self.bit_depth / 8 * (self.header["image_height"] * self.header["image_width"]))
                )
                if self.timestamp_length < 0 or self.timestamp_length > 4096:
                    self.timestamp_length = 10
        else:
            detected = self._detect_timestamp_length_compressed()
            if detected is not None:
                self.timestamp_length = detected

    @property
    def num_frames(self):
        return self.header["allocated_frames"]

    @property
    def width(self):
        return self.header["image_width"]

    @property
    def height(self):
        return self.header["image_height"]

    def _detect_timestamp_length_compressed(self):
        if not self.compressed:
            return None
        try:
            self.file.seek(1024)
            sb = self.file.read(4)
            if len(sb) < 4:
                return None
            size = struct.unpack("i", sb)[0]
            if size <= 4:
                return None
            self.file.seek(1024 + 4)
            window = self.file.read(min(size + 4096, 16 * 1024 * 1024))
            jpeg_region = window[: size - 4 + 256]
            eoi_idx = jpeg_region.rfind(b"\xff\xd9")
            if eoi_idx < 0:
                return None
            soi_idx = window.find(b"\xff\xd8", eoi_idx + 2)
            if soi_idx < 0:
                return None
            ts = (soi_idx - (eoi_idx + 2)) - 4
            if 0 <= ts <= 4096:
                return ts
            return None
        except Exception:
            return None

    def _file_size(self):
        try:
            cur = self.file.tell()
            self.file.seek(0, 2)
            sz = self.file.tell()
            self.file.seek(cur, 0)
            return sz
        except Exception:
            return None

    def _validate_seek_table(self, table):
        if not table:
            return False
        offset, size = table[0]
        try:
            file_size = self._file_size()
            if offset < 0 or size <= 0:
                return False
            if file_size is not None and offset >= file_size:
                return False
            self.file.seek(offset)
            head = self.file.read(min(4, size))
            if self.compressed:
                return len(head) >= 2 and head[0] == 0xFF and head[1] == 0xD8
            return True
        except Exception:
            return False

    def _build_table_from_mat(self, seek_mat_path):
        try:
            import scipy.io as sio
            seek = sio.loadmat(seek_mat_path)["seek"].ravel().astype(np.int64)
        except Exception as e:
            print(f"  [seek mat load failed for {os.path.basename(seek_mat_path)}: {e}]")
            return None
        table = []
        try:
            for i, off in enumerate(seek):
                off = int(off)
                if i + 1 < len(seek):
                    size = int(seek[i + 1] - off - 4 - self.timestamp_length)
                else:
                    self.file.seek(off)
                    sb = self.file.read(4)
                    size = struct.unpack("i", sb)[0] - 4 if len(sb) == 4 else 0
                table.append((off + 4, size))
        except Exception as e:
            print(f"  [seek mat build failed: {e}]")
            return None
        return table

    def _build_table_from_walk(self):
        table = []
        file_size = self._file_size()
        if self.compressed:
            try:
                self.file.seek(1024, 0)
            except Exception:
                return table
            while True:
                size_bytes = self.file.read(4)
                if not size_bytes or len(size_bytes) < 4:
                    break
                try:
                    size = struct.unpack("i", size_bytes)[0]
                except Exception:
                    break
                if size <= 4:
                    break
                if file_size is not None and size > file_size:
                    break
                offset = self.file.tell()
                advance = size - 4 + self.timestamp_length
                if advance <= 0:
                    break
                if file_size is not None and offset + advance > file_size:
                    break
                try:
                    self.file.seek(advance, 1)
                except OSError:
                    break
                table.append((offset, size))
        else:
            for f in range(self.num_frames):
                offset = f * self.header["true_image_size"] + 1024
                table.append((offset, self.header["image_size"]))
        return table

    def build_seek_table(self, seek_mat_path=None):
        if self.seek_table is not None:
            return
        mat_table = None
        if seek_mat_path is not None and os.path.isfile(seek_mat_path):
            mat_table = self._build_table_from_mat(seek_mat_path)
            if mat_table is not None and self._validate_seek_table(mat_table):
                self.seek_table = mat_table
                return
            elif mat_table is not None:
                print(f"  [seek mat for {os.path.basename(seek_mat_path)} failed validation; walking file instead]")
        walk_table = self._build_table_from_walk()
        if walk_table and self._validate_seek_table(walk_table):
            self.seek_table = walk_table
            return
        self.seek_table = walk_table or mat_table or []

    def _decode_at(self, offset, size):
        self.file.seek(offset)
        data = self.file.read(size + self.timestamp_length)
        if self.compressed:
            jpg_bytes = data[:-self.timestamp_length] if self.timestamp_length > 0 else data
            im = Image.open(io.BytesIO(jpg_bytes))
            im.load()
            return np.array(im)
        if self.image_format == FRAME_FORMAT_RAW_GRAY and self.bit_depth == 8:
            arr = np.frombuffer(data[:-self.timestamp_length], dtype=np.uint8)
            return arr.reshape(self.header["image_height"], self.header["image_width"])
        raise NotImplementedError(f"Unsupported .seq format: image_format={self.image_format} bit_depth={self.bit_depth}")

    def read_frame(self, index):
        if self.seek_table is None:
            self.build_seek_table()
        if index < 0 or index >= len(self.seek_table):
            raise IndexError(f"frame {index} out of range (n={len(self.seek_table)})")
        try:
            offset, size = self.seek_table[index]
            return self._decode_at(offset, size)
        except Exception:
            if not self._walk_retried:
                self._walk_retried = True
                walk = self._build_table_from_walk()
                if walk and self._validate_seek_table(walk):
                    self.seek_table = walk
                    if index < len(walk):
                        offset, size = walk[index]
                        return self._decode_at(offset, size)
            raise

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
