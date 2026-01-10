import os
import sys
import time
import logging

# Add src to path so we can import animaltracker
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from animaltracker.notification import PushoverNotifier, NotificationContext

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

def load_secrets():
    """Simple .env loader since python-dotenv might not be installed."""
    env_path = "config/secrets.env"
    if not os.path.exists(env_path):
        print(f"Warning: {env_path} not found. Relying on existing environment variables.")
        return

    print(f"Loading secrets from {env_path}...")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    key, val = line.split("=", 1)
                    # Only set if not already set, or just overwrite? Overwrite is safer for testing.
                    os.environ[key] = val
                except ValueError:
                    pass

if __name__ == "__main__":
    load_secrets()

    try:
        # Initialize notifier with the environment variable NAMES, not values
        notifier = PushoverNotifier(
            app_token_env="PUSHOVER_APP_TOKEN", 
            user_key_env="PUSHOVER_USER_KEY"
        )

        # Create a dummy context
        ctx = NotificationContext(
            species="TEST_SQUIRREL",
            confidence=0.99,
            camera_id="test_cam_01",
            camera_name="Debug Camera",
            clip_path="/tmp/placeholder.mp4",  # Won't actually be uploaded unless implemented
            event_started_at=time.time(),
            event_duration=10.5,
            thumbnail_path=None,  # Set to an actual .jpg path to test image attachment
            storage_root=None,
            web_base_url=None,  # Set to test URL link, e.g., "http://192.168.1.195:8080"
        )

        print("Sending test notification to Pushover...")
        notifier.send(ctx, priority=1, sound="cosmic")
        print("Notification sent successfully! Check your phone.")

    except Exception as e:
        print(f"\n❌ Error sending notification: {e}")
        print("Double check your config/secrets.env file.")
