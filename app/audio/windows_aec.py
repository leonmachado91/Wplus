"""Windows Acoustic Echo Cancellation (AEC) helper.

Strategy
--------
Open the microphone through the Windows *Communications* device role
(eCommunications).  In this role Windows applies its software APO chain
(AEC + Noise Suppression + AGC) before delivering PCM to the caller.

Implementation details
----------------------
- Uses only **ctypes** (built-in) + **winreg** (built-in) — zero extra deps.
- All failures are caught and logged at DEBUG level.
- The public function always returns either a valid sounddevice index or
  ``None`` (= fall back to the user-selected device, app runs normally).

Thread safety
-------------
``CoInitializeEx`` is safe to call multiple times in the same thread;
subsequent calls return RPC_E_CHANGED_MODE which we ignore.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import re
import winreg
from typing import Optional

logger = logging.getLogger(__name__)

# ── COM constants ────────────────────────────────────────────────────────────
_COINIT_MULTITHREADED  = 0x0
_CLSCTX_INPROC_SERVER  = 0x1
_S_OK                  = 0
_eCapture              = 1
_eCommunications       = 2   # ← the role that triggers Windows AEC pipeline

# IMMDeviceEnumerator vtable indices (0=QI, 1=AddRef, 2=Release, then own methods)
_VIDX_GetDefaultAudioEndpoint = 4  # [3]=EnumAudioEndpoints, [4]=GetDefaultAudioEndpoint
_VIDX_GetId                   = 5  # IMMDevice: [3]=Activate, [4]=OpenPropertyStore, [5]=GetId

# CLSID of the MMDeviceEnumerator COM class
_CLSID_STR_MMDeviceEnumerator = "BCDE0395-E52F-467C-8E3D-C4579291692E"
_IID_STR_IMMDeviceEnumerator  = "A95664D2-9614-4F35-A746-DE8DB63617E6"

# PKEY_Device_FriendlyName = {a45c254e-df1c-4efd-8020-67d146a850e0}, 14
_PKEY_FriendlyName = "{a45c254e-df1c-4efd-8020-67d146a850e0},14"

# PROPVARIANT.vt == VT_LPWSTR means the payload starts at byte offset 8
_VT_LPWSTR = 31


# ── public API ────────────────────────────────────────────────────────────────


def get_communications_mic_sounddevice_index() -> Optional[int]:
    """Return the sounddevice device index for the Windows Communications mic.

    Returns ``None`` on any failure — the caller should use the user-selected
    device instead.  This function never raises.
    """
    try:
        return _find_comms_mic_index()
    except Exception:
        logger.debug("Windows AEC: device query failed (non-fatal)", exc_info=True)
        return None


# ── internal helpers ──────────────────────────────────────────────────────────


def _guid_bytes(guid_str: str) -> ctypes.Array:
    """'BCDE0395-...' → little-endian GUID bytes array for CoCreateInstance."""
    import uuid
    g = uuid.UUID(guid_str.strip("{}"))
    return (ctypes.c_byte * 16)(*g.bytes_le)


def _vtable_call(obj_ptr, vtable_index: int, restype, *argtypes):
    """Build and return a callable for vtable[vtable_index] of a COM object."""
    vtable         = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.c_void_p))
    vtable_entries = ctypes.cast(vtable[0], ctypes.POINTER(ctypes.c_void_p))
    fn_ptr         = vtable_entries[vtable_index]
    FuncType       = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return FuncType(fn_ptr)


def _com_release(obj_ptr) -> None:
    """Call IUnknown::Release on a COM object pointer."""
    if obj_ptr:
        release = _vtable_call(obj_ptr, 2, ctypes.c_ulong)
        release(obj_ptr)


def _find_comms_mic_index() -> Optional[int]:
    """Internal logic — may raise; caller wraps in try/except."""
    import sounddevice as sd

    device_id = _get_communications_device_id()
    if not device_id:
        logger.debug("Windows AEC: GetDefaultAudioEndpoint returned nothing")
        return None

    friendly_name = _friendly_name_from_registry(device_id)
    if not friendly_name:
        logger.debug("Windows AEC: registry lookup failed for device_id=%s", device_id)
        return None

    logger.debug("Windows AEC: communications mic friendly name = '%s'", friendly_name)

    # Substring match — sounddevice may append " (2- ...)" or similar suffixes.
    fname_lower = friendly_name.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and fname_lower in d["name"].lower():
            logger.info(
                "Windows AEC: mapped to sounddevice [%d] '%s'", i, d["name"]
            )
            return i

    logger.debug(
        "Windows AEC: '%s' not found in sounddevice device list", friendly_name
    )
    return None


def _get_communications_device_id() -> Optional[str]:
    """Call IMMDeviceEnumerator::GetDefaultAudioEndpoint(eCapture, eCommunications)
    and return the raw Windows device ID string, e.g.
    ``\\\\?\\SWD#MMDEVAPI#{0.0.1.00000000}.{GUID}#{...}``.
    """
    ole32 = ctypes.windll.ole32

    # CoInitializeEx is safe to call multiple times; ignore RPC_E_CHANGED_MODE
    ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)

    clsid = _guid_bytes(_CLSID_STR_MMDeviceEnumerator)
    iid   = _guid_bytes(_IID_STR_IMMDeviceEnumerator)

    enumerator = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        clsid, None, _CLSCTX_INPROC_SERVER, iid, ctypes.byref(enumerator)
    )
    if hr != _S_OK or not enumerator:
        logger.debug("Windows AEC: CoCreateInstance failed (hr=0x%08X)", hr & 0xFFFFFFFF)
        return None

    try:
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint(eCapture, eCommunications, &device)
        GetDefaultAudioEndpoint = _vtable_call(
            enumerator,
            _VIDX_GetDefaultAudioEndpoint,
            ctypes.HRESULT,
            ctypes.c_int,                         # dataFlow
            ctypes.c_int,                         # role
            ctypes.POINTER(ctypes.c_void_p),      # ppEndpoint
        )
        device = ctypes.c_void_p()
        hr = GetDefaultAudioEndpoint(
            enumerator, _eCapture, _eCommunications, ctypes.byref(device)
        )
        if hr != _S_OK or not device:
            logger.debug("Windows AEC: GetDefaultAudioEndpoint failed (hr=0x%08X)", hr & 0xFFFFFFFF)
            return None

        try:
            # IMMDevice::GetId(&ppstrId) — returns CoTaskMem-allocated LPWSTR
            GetId = _vtable_call(
                device,
                _VIDX_GetId,
                ctypes.HRESULT,
                ctypes.POINTER(ctypes.c_void_p),  # ppstrId (LPWSTR*)
            )
            raw_ptr = ctypes.c_void_p()
            hr = GetId(device, ctypes.byref(raw_ptr))
            if hr != _S_OK or not raw_ptr.value:
                return None

            device_id = ctypes.wstring_at(raw_ptr.value)
            ole32.CoTaskMemFree(raw_ptr)           # release COM-allocated string
            return device_id
        finally:
            _com_release(device)
    finally:
        _com_release(enumerator)


def _friendly_name_from_registry(device_id: str) -> Optional[str]:
    """Read PKEY_Device_FriendlyName from the Windows MMDevices registry.

    The device ID looks like:
        ``\\\\?\\SWD#MMDEVAPI#{0.0.1.00000000}.{GUID}#{endpoint-guid}``

    The MMDevices subkey is the ``{0.0.1.00000000}.{GUID}`` portion.
    """
    m = re.search(r"\{[0-9.]+\}\.\{[0-9a-fA-F-]+\}", device_id)
    if not m:
        return None

    subkey_guid = m.group(0)
    props_path  = (
        r"SOFTWARE\Microsoft\Windows\CurrentVersion"
        rf"\MMDevices\Audio\Capture\{subkey_guid}\Properties"
    )

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, props_path) as key:
            val, _ = winreg.QueryValueEx(key, _PKEY_FriendlyName)

            if isinstance(val, str):
                # REG_SZ — plain string, common on older drivers
                return val

            if isinstance(val, bytes) and len(val) >= 10:
                # REG_BINARY — PROPVARIANT blob stored by the Windows Audio engine.
                # Layout: vt(2B) + reserved(6B) + payload
                vt = int.from_bytes(val[:2], "little")
                if vt == _VT_LPWSTR:
                    # Payload at offset 8 is an inline UTF-16LE string
                    try:
                        return val[8:].decode("utf-16-le").split("\x00")[0]
                    except UnicodeDecodeError:
                        pass

    except FileNotFoundError:
        pass
    except Exception:
        logger.debug(
            "Windows AEC: registry read error for subkey '%s'", subkey_guid, exc_info=True
        )

    return None
