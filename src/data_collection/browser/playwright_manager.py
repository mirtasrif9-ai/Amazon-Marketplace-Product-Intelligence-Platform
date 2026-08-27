import logging

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

from src.common.config import REQUEST_TIMEOUT


logger = logging.getLogger(__name__)


class PlaywrightManager:
    """Manage Playwright browser lifecycle."""

    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    def start(self) -> Page:
        """Start Chromium and return a new page."""

        logger.info(
            "Starting Playwright browser. headless=%s",
            self.headless,
        )

        try:
            self._playwright = sync_playwright().start()

            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
            )

            self._context = self._browser.new_context()

            page = self._context.new_page()

            page.set_default_timeout(REQUEST_TIMEOUT * 1000)

            logger.info("Playwright browser started successfully.")

            return page

        except Exception:
            logger.exception(
                "Failed to start Playwright browser."
            )
            self.close()
            raise

    def close(self) -> None:
        """Close browser resources safely."""

        logger.info("Closing Playwright browser.")

        if self._context is not None:
            self._context.close()
            self._context = None

        if self._browser is not None:
            self._browser.close()
            self._browser = None

        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

        logger.info("Playwright browser closed.")