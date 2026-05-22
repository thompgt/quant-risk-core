"""Real-time market data helpers.

Provides a generic WebSocketConnector wrapper (uses `websocket-client`) and a
simple PollingStreamer that polls `YahooFinanceConnector.fetch_latest_price`.

The WebSocketConnector is intentionally generic so it can be used with any
provider that exposes a websocket URL. Parsing of provider-specific messages
should be provided by the `on_message` callback.
"""
import threading
import time
from typing import Callable, Optional

try:
    import websocket
except Exception:  # pragma: no cover - optional dependency
    websocket = None

from .data_connectors import YahooFinanceConnector


class WebSocketConnector:
    """Generic websocket client wrapper.

    Example:
        ws = WebSocketConnector(url, on_message=handle_msg)
        ws.start()
        ws.stop()
    """

    def __init__(self, url: str, on_message: Callable[[str], None], on_error: Optional[Callable[[Exception], None]] = None):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed; install websocket-client in your environment")
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self._ws = None
        self._thread = None
        self._running = False

    def _run(self):
        def _on_message(wsapp, message):
            try:
                self.on_message(message)
            except Exception:
                # swallow exceptions from callback to keep the loop running
                pass

        def _on_error(wsapp, error):
            if self.on_error:
                try:
                    self.on_error(error)
                except Exception:
                    pass

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(self.url, on_message=_on_message, on_error=_on_error)
                self._ws.run_forever()
            except Exception as e:
                if self.on_error:
                    try:
                        self.on_error(e)
                    except Exception:
                        pass
            # simple reconnect delay
            time.sleep(1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)


class PollingStreamer:
    """Poll `YahooFinanceConnector.fetch_latest_price` periodically.

    This is useful as a lightweight alternative when a websocket feed is not
    available. It runs callbacks on each new poll in a background thread.
    """

    def __init__(self, ticker: str, interval: float = 5.0, callback: Optional[Callable[[Optional[float]], None]] = None):
        self.ticker = ticker
        self.interval = max(0.1, float(interval))
        self.callback = callback
        self._thread = None
        self._running = False

    def _run(self):
        while self._running:
            try:
                price = YahooFinanceConnector.fetch_latest_price(self.ticker)
                if self.callback:
                    try:
                        self.callback(price)
                    except Exception:
                        pass
            except Exception:
                # swallow errors to keep polling
                pass
            time.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
