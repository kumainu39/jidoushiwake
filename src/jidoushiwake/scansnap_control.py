from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


USER32 = ctypes.WinDLL("user32", use_last_error=True)
KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)


FindWindowW = USER32.FindWindowW
FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
FindWindowW.restype = wintypes.HWND

SendMessageW = USER32.SendMessageW
SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
# Some Python builds lack wintypes.LRESULT; use pointer-sized c_ssize_t fallback
try:
    SendMessageW.restype = wintypes.LRESULT  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    try:
        SendMessageW.restype = ctypes.c_ssize_t  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        SendMessageW.restype = ctypes.c_long


WM_COPYDATA = 0x004A
WMUSER_ISVAPP = 0x0591


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", wintypes.DWORD),
        ("cbData", wintypes.DWORD),
        ("lpData", wintypes.LPVOID),
    ]


class SS_NOTIFY(ctypes.Structure):
    _fields_ = [
        ("Mode", ctypes.c_int),
        ("AppName", ctypes.c_char * 255),
    ]


class SS_SCAN(ctypes.Structure):
    _fields_ = [
        ("Mode", ctypes.c_int),
        ("ScanningSide", ctypes.c_bool),
        ("AppName", ctypes.c_char * 255),
    ]


@dataclass
class ScanSnapPaths:
    home_exe: Optional[str]
    sdk_exe: Optional[str]


def _read_app_path(app: str) -> Optional[str]:
    try:
        import winreg  # type: ignore[attr-defined]

        key_path = fr"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{app}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            val, _ = winreg.QueryValueEx(key, "")
            return val
    except Exception:
        return None


def find_paths() -> ScanSnapPaths:
    return ScanSnapPaths(
        home_exe=_read_app_path("PfuSsMon.exe"),
        sdk_exe=_read_app_path("PfuSsMonSdk.exe"),
    )


def ensure_scansnap_running(timeout_sec: int = 5) -> bool:
    paths = find_paths()
    hwnd = FindWindowW("ScanSnap Manager MainWndClass", None)
    if hwnd:
        return True
    if paths.home_exe and Path(paths.home_exe).exists():
        try:
            subprocess.Popen([paths.home_exe])
        except Exception:
            return False
        start = time.time()
        while time.time() - start < timeout_sec:
            time.sleep(0.5)
            hwnd = FindWindowW("ScanSnap Manager MainWndClass", None)
            if hwnd:
                return True
    return False


def _send_copydata_struct(hwnd: int, dw_data: int, struct_buf: bytes) -> int:
    """Send WM_COPYDATA with a COPYDATASTRUCT payload.

    ctypes' ``SendMessageW.argtypes`` uses ``LPARAM`` (an integer). For
    WM_COPYDATA, lParam must be the POINTER to COPYDATASTRUCT, which in ctypes
    means we should pass the address as an integer (``ctypes.addressof``) rather
    than ``byref(struct)``. Passing ``byref`` causes "argument 4: wrong type".
    """
    cd = COPYDATASTRUCT()
    cd.dwData = dw_data
    cd.cbData = len(struct_buf)
    # Allocate a stable buffer for the payload so the pointer remains valid
    lp = ctypes.create_string_buffer(struct_buf)
    cd.lpData = ctypes.cast(lp, wintypes.LPVOID)

    lparam = ctypes.addressof(cd)
    res = SendMessageW(hwnd, WM_COPYDATA, 0, lparam)
    return int(res)


def reserve(home_app_name: str = "ImageConnectionsForHome") -> int:
    """Reserve connection with ScanSnap Home via WM_COPYDATA notify."""
    hwnd = FindWindowW("ScanSnap Manager MainWndClass", None)
    if not hwnd:
        return -1
    notify = SS_NOTIFY()
    notify.Mode = 0  # 0: RESERVE
    notify.AppName = home_app_name.encode("ascii")[:254] + b"\x00"
    buf = bytes(bytearray(ctypes.string_at(ctypes.byref(notify), ctypes.sizeof(notify))))
    return _send_copydata_struct(hwnd, 2, buf)


def release(home_app_name: str = "ImageConnectionsForHome") -> int:
    hwnd = FindWindowW("ScanSnap Manager MainWndClass", None)
    if not hwnd:
        return -1
    notify = SS_NOTIFY()
    notify.Mode = 1  # 1: RELEASE
    notify.AppName = home_app_name.encode("ascii")[:254] + b"\x00"
    buf = bytes(bytearray(ctypes.string_at(ctypes.byref(notify), ctypes.sizeof(notify))))
    return _send_copydata_struct(hwnd, 2, buf)


def start_scan(home_app_name: str = "ImageConnectionsForHome", one_side: bool = True) -> int:
    hwnd = FindWindowW("ScanSnap Manager MainWndClass", None)
    if not hwnd:
        return -1
    scan = SS_SCAN()
    scan.Mode = 0
    scan.ScanningSide = bool(one_side)
    scan.AppName = home_app_name.encode("ascii")[:254] + b"\x00"
    buf = bytes(bytearray(ctypes.string_at(ctypes.byref(scan), ctypes.sizeof(scan))))
    return _send_copydata_struct(hwnd, 33, buf)


def reserve_and_scan() -> bool:
    if not ensure_scansnap_running():
        return False
    # Try reserve, ignore non-zero codes and proceed
    try:
        reserve()
    except Exception:
        pass
    rc = start_scan()
    return rc == 0
