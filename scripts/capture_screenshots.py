"""Capture README screenshots from a running Nimbus Streamlit server."""
from pathlib import Path
from playwright.sync_api import Error, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "screenshots"
OUTPUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    try:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
    except Error:
        browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded")
    page.get_by_role("heading", name="New York, United States").wait_for(timeout=60_000)
    page.wait_for_timeout(2_000)
    page.screenshot(path=OUTPUT / "dashboard-overview.png")

    page.get_by_role("tab", name="Forecast lab").click()
    page.get_by_text("Tomorrow temperature estimator").wait_for(timeout=10_000)
    page.wait_for_timeout(1_000)
    page.screenshot(path=OUTPUT / "forecast-lab.png")
    browser.close()
