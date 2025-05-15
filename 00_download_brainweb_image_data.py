from pathlib import Path
import urllib.request
import zipfile

DATA_DIR = Path("data")
ZIP_URL = "https://zenodo.org/records/8067595/files/brainweb_petmr_v2.zip?download=1"
EXTRACTED_FOLDER = DATA_DIR / "brainweb_petmr_v2"
ZIP_PATH = EXTRACTED_FOLDER / "brainweb_petmr_v2.zip"


def main():

    # Check if data already exists
    if EXTRACTED_FOLDER.exists():
        print(f"Dataset already exists in {EXTRACTED_FOLDER}. Skipping download.")
        return

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

    # Download the zip file if it doesn't exist
    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip file already downloaded.")

    # Unzip the file
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    print("Extraction complete.")

    # Optionally, remove the zip file after extraction
    ZIP_PATH.unlink()


if __name__ == "__main__":
    main()
