#!/usr/bin/env python3
"""Test PTZ position reporting from camera."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load secrets
load_dotenv(Path(__file__).parent.parent / "config" / "secrets.env")

from animaltracker.onvif_client import OnvifClient

# Camera settings - adjust as needed
HOST = "10.0.1.198"
PORT = 8000
USERNAME = os.environ.get("ONVIF_USER", "admin")
PASSWORD = os.environ.get("ONVIF_PASSWORD", "")

def main():
    print(f"Connecting to {HOST}:{PORT} as {USERNAME}...")
    client = OnvifClient(HOST, PORT, USERNAME, PASSWORD)
    
    # Get all profiles
    profiles = client.get_profiles()
    print(f"\nFound {len(profiles)} profiles:")
    for p in profiles:
        token = p.metadata.get('token', 'unknown')
        name = p.metadata.get('name', 'unknown')
        print(f"  - Token: {token}, Name: {name}")
    
    # Get PTZ service directly for raw inspection
    ptz = client._camera.create_ptz_service()
    
    print("\n" + "="*60)
    print("RAW PTZ STATUS FOR EACH PROFILE:")
    print("="*60)
    
    for p in profiles:
        token = p.metadata.get('token')
        if not token:
            continue
            
        print(f"\n--- Profile: {token} ---")
        try:
            status = ptz.GetStatus({"ProfileToken": token})
            print(f"Raw status object: {status}")
            print(f"  type: {type(status)}")
            
            # Inspect all attributes
            for attr in dir(status):
                if not attr.startswith('_'):
                    val = getattr(status, attr, None)
                    if val is not None and not callable(val):
                        print(f"  .{attr} = {val}")
                        # Go deeper for Position
                        if attr == 'Position' and val is not None:
                            for sub_attr in dir(val):
                                if not sub_attr.startswith('_'):
                                    sub_val = getattr(val, sub_attr, None)
                                    if sub_val is not None and not callable(sub_val):
                                        print(f"    .{sub_attr} = {sub_val}")
                                        # Go even deeper for PanTilt/Zoom
                                        if hasattr(sub_val, 'x'):
                                            print(f"      .x = {sub_val.x}")
                                        if hasattr(sub_val, 'y'):
                                            print(f"      .y = {sub_val.y}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Also check PTZ configurations
    print("\n" + "="*60)
    print("PTZ CONFIGURATIONS:")
    print("="*60)
    try:
        configs = ptz.GetConfigurations()
        for cfg in configs:
            print(f"\n  Token: {cfg.token}")
            print(f"  Name: {cfg.Name}")
            print(f"  NodeToken: {cfg.NodeToken}")
            if hasattr(cfg, 'DefaultAbsolutePantable'):
                print(f"  DefaultAbsolutePantable: {cfg.DefaultAbsolutePantable}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    main()
