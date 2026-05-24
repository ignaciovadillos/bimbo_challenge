from pathlib import Path
import zipfile


def unzip_file(zip_path: Path, extract_to: Path, remove_zip: bool = False) -> None:
    """
    Unzip a file into the target directory.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file.
    extract_to : Path
        Directory where files should be extracted.
    remove_zip : bool
        Whether to delete the zip file after extraction.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {zip_path}")
    print(f"Destination: {extract_to}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if remove_zip:
        zip_path.unlink()
        print(f"Removed zip file: {zip_path}")

    print("Extraction complete.")


def unzip_nested_csv_zips(directory: Path, remove_nested_zips: bool = False) -> None:
    """
    Unzip any .csv.zip files found inside the extracted folder.

    Example files:
    - train.csv.zip
    - test.csv.zip
    - cliente_tabla.csv.zip
    """
    nested_zips = list(directory.glob("*.csv.zip"))

    if not nested_zips:
        print("No nested .csv.zip files found.")
        return

    for nested_zip in nested_zips:
        print(f"Extracting nested file: {nested_zip.name}")

        with zipfile.ZipFile(nested_zip, "r") as zip_ref:
            zip_ref.extractall(directory)

        if remove_nested_zips:
            nested_zip.unlink()
            print(f"Removed nested zip: {nested_zip.name}")

    print("Nested CSV extraction complete.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    raw_dir = project_root / "data" / "raw"
    csv_dir = raw_dir / "csv"

    main_zip = raw_dir / "grupo-bimbo-inventory-demand.zip"

    unzip_file(
        zip_path=main_zip,
        extract_to=csv_dir,
        remove_zip=False,
    )

    unzip_nested_csv_zips(
        directory=csv_dir,
        remove_nested_zips=False,
    )


if __name__ == "__main__":
    main()