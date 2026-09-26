"""Keep Windows Qt from resolving Conda's incompatible unversioned ICU DLL."""

import sys

_system_icu = None


def prepare_qt_runtime():
    global _system_icu
    if sys.platform != 'win32':
        return False
    if _system_icu is not None:
        return True
    import ctypes
    try:
        _system_icu = ctypes.WinDLL('icuuc.dll', winmode=0x00000800)
    except OSError:
        return False
    return True
