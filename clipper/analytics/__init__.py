from __future__ import annotations


class Analytics:
    """Stub. Enabled by ANALYTICS_ENABLED; post-performance tracking is planned for later."""

    def __init__(self, settings):
        self.settings = settings

    def enabled(self) -> bool:
        return bool(self.settings.analytics)

    def compute(self, job):
        raise NotImplementedError("analytics is not implemented yet")
