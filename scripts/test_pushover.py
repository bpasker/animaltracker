"""Send a test Pushover alert the way the pipeline does.

    python scripts/test_pushover.py                 # every destination in config/cameras.yml
    python scripts/test_pushover.py jane            # only these destination ids
    python scripts/test_pushover.py --config config/cameras.mac.yml brandon

Reads config/secrets.env for the variables the destinations name. With no
destinations defined the fallback PUSHOVER_USER_KEY variable is used, as at
runtime.
"""
import argparse
import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from animaltracker.config import load_runtime_config  # noqa: E402
from animaltracker.notification import NotificationContext, PushoverNotifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_secrets(env_path: str = "config/secrets.env") -> None:
    """Simple .env loader since python-dotenv might not be installed."""
    if not os.path.exists(env_path):
        print(f"Warning: {env_path} not found. Relying on existing environment variables.")
        return
    print(f"Loading secrets from {env_path}...")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("destinations", nargs="*", help="destination ids to send to (default: every destination)")
    ap.add_argument("--config", default="config/cameras.yml")
    args = ap.parse_args()

    load_secrets()
    settings = load_runtime_config(args.config).general.notification
    names = [d.label for d in settings.resolved_destinations()]
    print(f"Destinations in {args.config}: {', '.join(names) or 'none'}")

    ctx = NotificationContext(
        species="TEST_SQUIRREL",
        confidence=0.99,
        camera_id="test_cam_01",
        camera_name="Debug Camera",
        clip_path="/tmp/placeholder.mp4",
        event_started_at=time.time(),
        event_duration=10.5,
        thumbnail_path=None,  # Set to an actual .jpg path to test image attachment
        storage_root=None,
        web_base_url=settings.web_base_url,
    )
    print("Sending test notification to Pushover...")
    sent = PushoverNotifier(settings).send(ctx, priority=1, sound="cosmic",
                                           destinations=args.destinations or None)
    if sent:
        print(f"Sent to {sent} user key(s). Check your phone.")
        return 0
    print("Nothing was sent. The log lines above say which variable or destination is missing;"
          " double check config/secrets.env.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
