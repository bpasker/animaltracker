"""Simple ONVIF helper utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from onvif import ONVIFCamera  # type: ignore
except ImportError:  # pragma: no cover
    ONVIFCamera = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class OnvifProfile:
    uri: str
    snapshot_uri: Optional[str]
    metadata: Dict[str, Any]


class OnvifClient:
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        if ONVIFCamera is None:
            LOGGER.warning("onvif-zeep library not installed; ONVIF features disabled")
        else:
            self._camera = ONVIFCamera(host, port, username, password)

    def get_profiles(self) -> list[OnvifProfile]:
        if ONVIFCamera is None:
            return []
        media_service = self._camera.create_media_service()
        profiles = media_service.GetProfiles()
        results = []
        for profile in profiles:
            token = profile.token
            stream_uri = media_service.GetStreamUri({"StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}, "ProfileToken": token}).Uri
            try:
                snapshot_uri = media_service.GetSnapshotUri({"ProfileToken": token}).Uri
            except Exception:  # noqa: BLE001
                snapshot_uri = None
            results.append(OnvifProfile(uri=stream_uri, snapshot_uri=snapshot_uri, metadata={"token": token}))
        return results

    def get_status(self) -> Dict[str, Any]:
        if ONVIFCamera is None:
            return {"status": "unknown", "reason": "library-missing"}
        dev_service = self._camera.create_devicemgmt_service()
        info = dev_service.GetDeviceInformation()
        return {
            "manufacturer": getattr(info, "Manufacturer", "unknown"),
            "model": getattr(info, "Model", "unknown"),
            "firmware": getattr(info, "FirmwareVersion", "unknown"),
        }

    def ptz_move(self, profile_token: str, pan: float, tilt: float, zoom: float = 0.0) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("ContinuousMove")
        request.ProfileToken = profile_token
        request.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        ptz_service.ContinuousMove(request)

    def ptz_stop(self, profile_token: str) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("Stop")
        request.ProfileToken = profile_token
        request.PanTilt = True
        request.Zoom = True
        ptz_service.Stop(request)
