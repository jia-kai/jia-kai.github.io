import argparse
import re
import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


PROFILE_DIR = Path.cwd() / "chromium-profile"
MEDIA_CHECKBOXES = (
    '[role="main"] [role="checkbox"][aria-label]'
    ':not([aria-label^="Select all"])'
)
DATE_CHECKBOXES = '[role="checkbox"][aria-label^="Select all photos from"]'
MEDIA_LINKS = 'a[href*="/photo/"]'
LIBRARIES = {
    "Photos": "https://photos.google.com/",
    "Archive": "https://photos.google.com/archive",
}
MAX_BATCH_SIZE = 2_000


def log(message: str) -> None:
    print(message, flush=True)


def wait_for_library(page: Page, name: str, url: str) -> None:
    log(f"{name}: loading {url}")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    if not page.url.startswith("https://photos.google.com/"):
        raise RuntimeError(f"Google Photos login was lost; opened {page.url}")
    log(f"{name}: page ready")


def wait_for_selectable_media(page: Page, name: str) -> bool:
    saw_media_links = False
    for wait_number in range(1, 31):
        if page.locator(MEDIA_CHECKBOXES).count() > 0:
            return True
        saw_media_links = saw_media_links or page.locator(MEDIA_LINKS).count() > 0
        if wait_number == 1 or wait_number % 10 == 0:
            log(f"{name}: waiting for media controls ({wait_number}s)")
        page.wait_for_timeout(1000)

    if saw_media_links:
        raise RuntimeError(
            f"{name}: media links loaded but selection controls did not; refusing to "
            "treat the library as empty."
        )
    return False


def confirm_library_empty(page: Page, name: str, url: str) -> bool:
    for confirmation in range(1, 4):
        if confirmation > 1:
            log(f"{name}: empty check {confirmation}/3; waiting and reloading")
            page.wait_for_timeout(10_000)
            wait_for_library(page, name, url)
        if wait_for_selectable_media(page, name):
            if confirmation > 1:
                log(f"{name}: media appeared during empty verification")
            return False
        log(f"{name}: empty check {confirmation}/3 passed")
    return True


def selected_count(page: Page) -> int:
    body_text = page.locator("body").inner_text()
    selected = re.search(r"([\d,]+) selected", body_text, re.IGNORECASE)
    return int(selected.group(1).replace(",", "")) if selected else 0


def select_batch(page: Page) -> int:
    main = page.locator('[role="main"]').first
    previous_top = -1
    stalled = 0

    for viewport_number in range(1, 1000):
        date_checkboxes = page.locator(DATE_CHECKBOXES)
        dates_clicked = date_checkboxes.evaluate_all(
            """
            elements => {
                let clicked = 0;
                for (const element of elements) {
                    if (element.getAttribute('aria-checked') !== 'true') {
                        element.click();
                        clicked += 1;
                    }
                }
                return clicked;
            }
            """
        )
        page.wait_for_timeout(100)

        checkboxes = page.locator(MEDIA_CHECKBOXES)
        items_clicked = checkboxes.evaluate_all(
            """
            elements => {
                let clicked = 0;
                for (const element of elements) {
                    if (element.getAttribute('aria-checked') !== 'true') {
                        element.click();
                        clicked += 1;
                    }
                }
                return clicked;
            }
            """
        )
        page.wait_for_timeout(100)
        count = selected_count(page)
        if viewport_number == 1 or viewport_number % 10 == 0 or count >= MAX_BATCH_SIZE:
            log(
                f"  viewport {viewport_number}: selected {count} total "
                f"({dates_clicked} date groups, {items_clicked} loose items clicked)"
            )
        if count >= MAX_BATCH_SIZE:
            return count

        state = main.evaluate(
            """
            element => {
                const previousTop = element.scrollTop;
                element.scrollTop += element.clientHeight * 0.9;
                return {
                    previousTop,
                    height: element.scrollHeight,
                    top: element.scrollTop,
                    client: element.clientHeight,
                };
            }
            """
        )
        page.wait_for_timeout(200)
        at_end = state["top"] + state["client"] >= state["height"] - 5
        stalled = stalled + 1 if state["top"] == previous_top and at_end else 0
        previous_top = state["top"]
        if stalled >= 2:
            return count

    raise RuntimeError("Stopped after scanning 1000 viewports for one batch.")


def move_current_library_to_trash(page: Page, name: str, url: str) -> int:
    batches = 0

    while True:
        batch = batches + 1
        wait_for_library(page, name, url)
        checkboxes = page.locator(MEDIA_CHECKBOXES)
        if not wait_for_selectable_media(page, name):
            if not confirm_library_empty(page, name, url):
                continue
            log(f"{name}: empty")
            return batches

        first_label = checkboxes.first.get_attribute("aria-label")
        log(f"{name} batch {batch}: starting at {first_label!r}")
        count = select_batch(page)
        if count == 0:
            raise RuntimeError(f"{name} batch {batch}: no items could be selected.")
        log(f"{name} batch {batch}: moving {count} items to trash")

        page.get_by_label("Move to trash", exact=True).click()
        log(f"{name} batch {batch}: confirming move to trash")
        dialog = page.get_by_role("dialog")
        dialog.get_by_role("button", name="Move to trash", exact=True).click()
        for wait_number in range(1, 61):
            page.wait_for_timeout(10_000)
            if dialog.count() == 0 or not dialog.last.is_visible():
                break
            if wait_number == 1 or wait_number % 3 == 0:
                dialog_text = " ".join(dialog.last.inner_text().split())
                log(
                    f"{name} batch {batch}: still processing after "
                    f"{wait_number * 10}s; dialog={dialog_text[:200]!r}"
                )
        else:
            raise RuntimeError(
                f"{name} batch {batch}: confirmation remained open for 10 minutes."
            )
        page.wait_for_timeout(3000)
        batches += 1
        log(f"{name} batch {batch}: completed")

        if batches >= 1000:
            raise RuntimeError(f"Stopped after 1000 {name} batches.")


def empty_trash(page: Page) -> None:
    wait_for_library(page, "Trash", "https://photos.google.com/trash")
    trash_items = page.locator('a[href*="/trash/"]')
    if trash_items.count() == 0:
        log("Trash: already empty")
        return

    log("Trash: permanently deleting all items")
    page.get_by_role("button", name="Empty trash", exact=True).first.click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("button", name="Empty Trash", exact=True).click()
    for wait_number in range(1, 181):
        page.wait_for_timeout(10_000)
        if dialog.count() == 0 or not dialog.last.is_visible():
            break
        if wait_number == 1 or wait_number % 3 == 0:
            dialog_text = " ".join(dialog.last.inner_text().split())
            log(
                f"Trash: still emptying after {wait_number * 10}s; "
                f"dialog={dialog_text[:200]!r}"
            )
    else:
        raise RuntimeError("Trash confirmation remained open for 30 minutes.")
    page.wait_for_timeout(5000)
    wait_for_library(page, "Trash verification", "https://photos.google.com/trash")
    if page.locator('a[href*="/trash/"]').count() != 0:
        raise RuntimeError("Trash still contains rendered items after emptying.")
    log("Trash: empty")


def main() -> None:
    parser = argparse.ArgumentParser(description="Permanently remove Google Photos media.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform deletion; without this flag the script exits",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to delete without --execute.")

    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        raise SystemExit("Chromium was not found on PATH.")
    if not PROFILE_DIR.exists():
        raise SystemExit("Profile not found. Run login.py first.")

    log("Starting resumable Google Photos deletion")

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            executable_path=chromium,
            headless=True,
            viewport={"width": 1920, "height": 4000},
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            verification_pass = 1
            while True:
                moved_batches = 0
                log(f"Library verification pass {verification_pass}")
                for name, url in LIBRARIES.items():
                    moved_batches += move_current_library_to_trash(page, name, url)
                if moved_batches == 0:
                    break
                verification_pass += 1
                log("Media was removed; repeating all library checks before Trash")
            empty_trash(page)
        finally:
            context.close()


if __name__ == "__main__":
    main()
