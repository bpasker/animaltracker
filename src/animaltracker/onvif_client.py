"""Simple ONVIF helper utilities."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from onvif import ONVIFCamera  # type: ignore
except ImportError:  # pragma: no cover
    ONVIFCamera = None  # type: ignore

LOGGER = logging.getLogger(__name__)
PTZ_LOGGER = logging.getLogger('ptz.decisions')


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
        start_time = time.time()
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("ContinuousMove")
        request.ProfileToken = profile_token
        request.Velocity = {
            "PanTilt": {"x": pan, "y": tilt},
            "Zoom": {"x": zoom},
        }
        ptz_service.ContinuousMove(request)
        elapsed = (time.time() - start_time) * 1000
        PTZ_LOGGER.debug(
            "[ONVIF] ContinuousMove sent: pan=%.3f, tilt=%.3f, zoom=%.3f (%.1fms)",
            pan, tilt, zoom, elapsed
        )

    def ptz_move_absolute(self, profile_token: str, pan: float, tilt: float, zoom: float = 0.0) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        start_time = time.time()
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
        elapsed = (time.time() - start_time) * 1000
        PTZ_LOGGER.debug(
            "[ONVIF] AbsoluteMove sent: pan=%.3f, tilt=%.3f, zoom=%.3f (%.1fms)",
            pan, tilt, zoom, elapsed
        )

    def ptz_set_zoom(self, profile_token: str, zoom: float) -> None:
        """Set only the zoom level without changing pan/tilt.

        Some cameras don't support combined pan/tilt/zoom absolute moves,
        so this uses zoom-only absolute move or falls back to continuous zoom.
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")

        start_time = time.time()
        ptz_service = self._camera.create_ptz_service()

        # Try absolute zoom-only first
        try:
            request = ptz_service.create_type("AbsoluteMove")
            request.ProfileToken = profile_token
            # Only set Zoom, not PanTilt
            request.Position = {"Zoom": {"x": zoom}}
            ptz_service.AbsoluteMove(request)
            elapsed = (time.time() - start_time) * 1000
            PTZ_LOGGER.debug("[ONVIF] AbsoluteZoom sent: zoom=%.3f (%.1fms)", zoom, elapsed)
            return
        except Exception as e:
            PTZ_LOGGER.debug("[ONVIF] AbsoluteZoom failed, trying continuous: %s", e)

        # Fallback: use continuous zoom with timing
        try:
            # Get current zoom
            current_pos = self.ptz_get_position(profile_token)
            current_zoom = 0.0
            if current_pos:
                z = current_pos.get('zoom')
                current_zoom = z if z is not None else 0.0

            diff = zoom - current_zoom
            if abs(diff) < 0.02:
                return  # Close enough to target

            # Use continuous zoom
            zoom_speed = 0.5 if diff > 0 else -0.5
            duration = abs(diff) * 3.0  # Rough timing

            request = ptz_service.create_type("ContinuousMove")
            request.ProfileToken = profile_token
            request.Velocity = {"Zoom": {"x": zoom_speed}}
            ptz_service.ContinuousMove(request)

            time.sleep(min(duration, 4.0))  # Cap at 4 seconds
            self.ptz_stop(profile_token)

            elapsed = (time.time() - start_time) * 1000
            PTZ_LOGGER.debug("[ONVIF] ContinuousZoom used: target=%.3f (%.1fms)", zoom, elapsed)
        except Exception as e:
            LOGGER.warning("Failed to set zoom: %s", e)

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

    def ptz_get_position(self, profile_token: str) -> Dict[str, Any]:
        """Get current PTZ position for the given profile.
        
        Returns dict with keys: pan, tilt, zoom (normalized -1.0 to 1.0 range),
        and 'available' (bool) indicating if camera reports position.
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        ptz_service = self._camera.create_ptz_service()
        status = ptz_service.GetStatus({"ProfileToken": profile_token})
        
        result: Dict[str, Any] = {
            "pan": None,
            "tilt": None,
            "zoom": None,
            "available": False,
        }
        
        # Try to extract position from various response formats
        position = getattr(status, 'Position', None)
        
        if position is not None:
            # Standard ONVIF format
            pan_tilt = getattr(position, 'PanTilt', None)
            zoom = getattr(position, 'Zoom', None)
            
            if pan_tilt is not None:
                result["pan"] = float(getattr(pan_tilt, 'x', 0) or 0)
                result["tilt"] = float(getattr(pan_tilt, 'y', 0) or 0)
                result["available"] = True
            if zoom is not None:
                result["zoom"] = float(getattr(zoom, 'x', 0) or 0)
                result["available"] = True
        
        # Debug: Log raw status for troubleshooting
        LOGGER.debug("PTZ GetStatus for %s: %s", profile_token, status)
        
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

    def ptz_get_configurations(self) -> list:
        """Get all PTZ configurations to find which profiles support PTZ."""
        if ONVIFCamera is None:
            return []
        
        try:
            ptz_service = self._camera.create_ptz_service()
            configs = ptz_service.GetConfigurations()
            result = []
            for cfg in configs:
                result.append({
                    'token': getattr(cfg, 'token', None),
                    'name': getattr(cfg, 'Name', None),
                    'node_token': getattr(cfg, 'NodeToken', None),
                })
            return result
        except Exception as e:
            LOGGER.error("Failed to get PTZ configurations: %s", e)
            return []

    def ptz_find_working_profile(self) -> Optional[str]:
        """Find a profile token that actually supports PTZ movement.
        
        Tests each profile by attempting a small move and checking if position changes.
        Returns the first working profile token, or None if none work.
        """
        if ONVIFCamera is None:
            return None
        
        import time
        
        profiles = self.get_profiles()
        LOGGER.info("Testing %d profiles for PTZ support...", len(profiles))
        
        for profile in profiles:
            token = profile.metadata.get("token")
            if not token:
                continue
            
            try:
                # Get initial position
                initial = self.ptz_get_position(token)
                LOGGER.info("Profile '%s' initial position: pan=%.3f, tilt=%.3f", 
                           token, initial['pan'], initial['tilt'])
                
                # Try continuous move (most compatible)
                self.ptz_move(token, 0.3, 0.0, 0.0)
                time.sleep(0.5)
                self.ptz_stop(token)
                time.sleep(0.5)
                
                # Check if position changed
                after = self.ptz_get_position(token)
                LOGGER.info("Profile '%s' after move: pan=%.3f, tilt=%.3f", 
                           token, after['pan'], after['tilt'])
                
                if abs(after['pan'] - initial['pan']) > 0.001 or abs(after['tilt'] - initial['tilt']) > 0.001:
                    LOGGER.info("Profile '%s' supports PTZ!", token)
                    # Move back to original position
                    self.ptz_move_absolute(token, initial['pan'], initial['tilt'], initial['zoom'])
                    return token
                    
            except Exception as e:
                LOGGER.debug("Profile '%s' PTZ test failed: %s", token, e)
                continue
        
        LOGGER.warning("No PTZ-capable profile found!")
        return None

    def ptz_get_presets(self, profile_token: str) -> list[Dict[str, Any]]:
        """Get list of PTZ presets for a profile.
        
        Returns list of dicts with 'token', 'name', and optionally position info.
        """
        if ONVIFCamera is None:
            return []
        
        try:
            ptz_service = self._camera.create_ptz_service()
            presets = ptz_service.GetPresets({"ProfileToken": profile_token})
            
            result = []
            for preset in presets:
                preset_info = {
                    'token': getattr(preset, 'token', None),
                    'name': getattr(preset, 'Name', None),
                }
                
                # Try to get position if available
                position = getattr(preset, 'PTZPosition', None)
                if position:
                    pan_tilt = getattr(position, 'PanTilt', None)
                    zoom = getattr(position, 'Zoom', None)
                    if pan_tilt:
                        preset_info['pan'] = float(getattr(pan_tilt, 'x', 0) or 0)
                        preset_info['tilt'] = float(getattr(pan_tilt, 'y', 0) or 0)
                    if zoom:
                        preset_info['zoom'] = float(getattr(zoom, 'x', 0) or 0)
                
                result.append(preset_info)
            
            LOGGER.info("Found %d PTZ presets for profile %s", len(result), profile_token)
            return result
            
        except Exception as e:
            LOGGER.error("Failed to get PTZ presets: %s", e)
            return []

    def ptz_goto_preset(self, profile_token: str, preset_token: str, speed: float = 0.5) -> None:
        """Move PTZ to a preset position.
        
        Args:
            profile_token: ONVIF profile token
            preset_token: Preset token (from ptz_get_presets)
            speed: Movement speed (0.0 to 1.0)
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("GotoPreset")
        request.ProfileToken = profile_token
        request.PresetToken = preset_token
        request.Speed = {
            "PanTilt": {"x": speed, "y": speed},
            "Zoom": {"x": speed},
        }
        ptz_service.GotoPreset(request)
        LOGGER.debug("Moving to preset %s", preset_token)

    def ptz_set_preset(self, profile_token: str, preset_name: str, preset_token: Optional[str] = None) -> str:
        """Save current position as a preset.
        
        Args:
            profile_token: ONVIF profile token
            preset_name: Name for the preset
            preset_token: Optional existing preset token to overwrite
            
        Returns:
            The preset token
        """
        if ONVIFCamera is None:
            raise RuntimeError("ONVIF PTZ not available; install onvif-zeep")
        
        ptz_service = self._camera.create_ptz_service()
        request = ptz_service.create_type("SetPreset")
        request.ProfileToken = profile_token
        request.PresetName = preset_name
        if preset_token:
            request.PresetToken = preset_token
        
        result = ptz_service.SetPreset(request)
        LOGGER.info("Saved preset '%s' with token %s", preset_name, result)
        return result
