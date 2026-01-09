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

    def ptz_move_absolute(self, profile_token: str, pan: float, tilt: float, zoom: float = 0.0) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        
        # Get status to find range (optional, but good for debugging)
        # status = ptz_service.GetStatus({'ProfileToken': profile_token})
        
        request = ptz_service.create_type("AbsoluteMove")
        request.ProfileToken = profile_token
        request.Position = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        ptz_service.AbsoluteMove(request)

    def ptz_move_relative(self, profile_token: str, pan: float, tilt: float, zoom: float = 0.0) -> None:
        """Move PTZ by a relative amount from current position.
        
        Args:
            profile_token: ONVIF profile token
            pan: Relative pan movement (-1.0 to 1.0, negative=left, positive=right)
            tilt: Relative tilt movement (-1.0 to 1.0, negative=down, positive=up)
            zoom: Relative zoom movement (-1.0 to 1.0, negative=out, positive=in)
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("RelativeMove")
        request.ProfileToken = profile_token
        request.Translation = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        ptz_service.RelativeMove(request)

    def ptz_stop(self, profile_token: str) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("Stop")
        request.ProfileToken = profile_token
        request.PanTilt = True
        request.Zoom = True
        ptz_service.Stop(request)

    def ptz_get_position(self, profile_token: str) -> Dict[str, float]:
        """Get current PTZ position for the given profile.
        
        Returns dict with keys: pan, tilt, zoom (normalized -1.0 to 1.0 range).
        Useful for mapping zoom camera position to wide angle field of view.
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        status = ptz_service.GetStatus({"ProfileToken": profile_token})
        
        position = status.Position
        result = {
            "pan": 0.0,
            "tilt": 0.0,
            "zoom": 0.0,
        }
        
        if position is not None:
            if hasattr(position, "PanTilt") and position.PanTilt is not None:
                result["pan"] = float(position.PanTilt.x)
                result["tilt"] = float(position.PanTilt.y)
            if hasattr(position, "Zoom") and position.Zoom is not None:
                result["zoom"] = float(position.Zoom.x)
        
        return result

    def ptz_get_all_positions(self) -> Dict[str, Dict[str, float]]:
        """Get PTZ position for all profiles (useful for TrackMix with multiple streams).
        
        Returns dict mapping profile token -> position dict.
        """
        if ONVIFCamera is None:
            return {}
        
        positions = {}
        for profile in self.get_profiles():
            token = profile.metadata.get("token")
            if token:
                try:
                    positions[token] = self.ptz_get_position(token)
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("Failed to get PTZ position for %s: %s", token, e)
        
        return positions
