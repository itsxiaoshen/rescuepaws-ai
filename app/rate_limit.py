"""In-memory rate limits for the public demo, so nobody can run up the LLM bill.

Kept in memory, so limits reset when the server restarts and only work for one server
process. That's fine for a demo; a production service would use a shared store like Redis.
"""
import time
from collections import defaultdict, deque

DAY_SECONDS = 24 * 60 * 60


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, global_per_day: int, clock=time.time):
        self.per_minute = per_minute          # per client
        self.per_day = per_day                # per client
        self.global_per_day = global_per_day  # all clients together: caps the daily cost
        self.clock = clock
        self.history: dict[str, deque] = defaultdict(deque)  # client -> request times in the last day
        self.all_requests: deque = deque()

    def check(self, client: str) -> str | None:
        """Record a request from this client. Returns an error message if a limit is hit, else None."""
        now = self.clock()
        history = self.history[client]
        for times in (history, self.all_requests):
            while times and times[0] <= now - DAY_SECONDS:
                times.popleft()

        if len(self.all_requests) >= self.global_per_day:
            return "The demo has reached its daily limit. Please try again tomorrow."
        if len(history) >= self.per_day:
            return "You've reached today's message limit for this demo. Please try again tomorrow."
        if sum(1 for t in history if t > now - 60) >= self.per_minute:
            return "Too many messages. Please wait a minute and try again."

        history.append(now)
        self.all_requests.append(now)
        return None
