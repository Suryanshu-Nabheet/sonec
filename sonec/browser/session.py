"""Browser automation facade (Playwright optional)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sonec.core.errors import SonecError


class BrowserError(SonecError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="browser_error")


@dataclass
class BrowserPage:
    url: str
    title: str
    content: str


class BrowserSession:
    """Thin wrapper around Playwright when installed.

    Importing Playwright is deferred so the core package remains usable without
    the optional ``browser`` extra.
    """

    def __init__(self, *, headless: bool = True) -> None:
        self.headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise BrowserError(
                "Playwright is not installed. Install with: pip install 'sonec[browser]' "
                "&& playwright install chromium"
            ) from exc
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()

    async def aclose(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    async def goto(self, url: str) -> BrowserPage:
        if self._page is None:
            await self.start()
        assert self._page is not None
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith("file:")):
            raise BrowserError(f"Unsupported URL scheme: {url}")
        await self._page.goto(url)
        title = await self._page.title()
        content = await self._page.content()
        return BrowserPage(url=url, title=title, content=content)

    async def click(self, selector: str) -> None:
        if self._page is None:
            raise BrowserError("Browser session not started")
        await self._page.click(selector)

    async def fill(self, selector: str, value: str) -> None:
        if self._page is None:
            raise BrowserError("Browser session not started")
        await self._page.fill(selector, value)

    async def text(self, selector: str = "body") -> str:
        if self._page is None:
            raise BrowserError("Browser session not started")
        locator = self._page.locator(selector)
        return await locator.inner_text()

    async def screenshot(self, path: str) -> str:
        if self._page is None:
            raise BrowserError("Browser session not started")
        await self._page.screenshot(path=path)
        return path
