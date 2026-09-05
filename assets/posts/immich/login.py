import shutil
import subprocess
from pathlib import Path

PROFILE_DIR = Path.cwd() / "chromium-profile"
GOOGLE_PHOTOS_URL = "https://photos.google.com/"


def main() -> None:
    PROFILE_DIR.mkdir(exist_ok=True)
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        raise SystemExit("Chromium was not found on PATH.")

    print("Log in to Google Photos, then close the Chromium window.")
    subprocess.run(
        [
            chromium,
            f"--user-data-dir={PROFILE_DIR}",
            "--password-store=basic",
            "--no-first-run",
            "--start-maximized",
            GOOGLE_PHOTOS_URL,
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
