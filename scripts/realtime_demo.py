"""Demo for real-time streaming helpers.

Usage:
    python -m scripts.realtime_demo

This demo shows a polling streamer for a ticker and (optionally) a websocket
connector example. The websocket example is commented out because provider URLs
are provider-specific.
"""
import sys
import time
from pathlib import Path

# Runs against the source tree even when the package is not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_risk_core.data.realtime import PollingStreamer  # noqa: E402


def price_callback(price):
    print(f"Latest price: {price}")


def main():
    # Polling demo for AAPL every 5 seconds
    poller = PollingStreamer("AAPL", interval=5.0, callback=price_callback)
    poller.start()
    try:
        # run for 30 seconds
        time.sleep(30)
    finally:
        poller.stop()


if __name__ == "__main__":
    main()
