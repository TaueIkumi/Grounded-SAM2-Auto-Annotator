import argparse
import os
import zipfile
import shutil
from pathlib import Path
from cvat_sdk import make_client

def make_client_with_auth(url: str, username: str, password: str):
    try:
        client = make_client(host=url, credentials=(username, password))
    except Exception as e:
        raise ConnectionError(f"Failed to connect to CVAT server: {e}")
    return client

def prepare_zip_for_upload(format: str, dataset_dir: Path, zip_file: str = None) -> str:
    dataset_dir = Path(dataset_dir)
    
    if format == "coco":
        candidates = [
            dataset_dir / "annotations" / "instances.json",
            dataset_dir / "instances.json"
        ]
        if not any(c.exists() for c in candidates):
            found = list(dataset_dir.rglob("*.json"))
            if found:
                candidates.append(found[0])

        for path in candidates:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(f"COCO annotations (instances.json) not found in {dataset_dir}")

    elif format == "pascal_voc":
        # If a ZIP file path was provided explicitly, use it (and validate)
        if zip_file:
            p = Path(zip_file)
            if not p.exists():
                raise FileNotFoundError(f"ZIP file not found: {p}")
            if p.suffix.lower() != ".zip":
                raise ValueError(f"Provided file is not a ZIP: {p}")
            print(f"Using provided ZIP file: {p}")
            return str(p.resolve())

        zip_filename = "temp_pascal_voc.zip"
        print(f"Creating Pascal VOC ZIP (entire directory): {zip_filename} ...")
        base_name = zip_filename.replace('.zip', '')
        
        shutil.make_archive(
            base_name=base_name,
            format='zip',
            root_dir=dataset_dir
        )

        return str(Path(f"{base_name}.zip").resolve())

    elif format == "cvat_xml":
        xml_files = list(dataset_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {dataset_dir} for CVAT format")
        return str(xml_files[0])


    else:
        raise ValueError(f"Unsupported format: {format}")

def main():
    parser = argparse.ArgumentParser(description="Upload Local Dataset to CVAT")
    
    parser.add_argument("-u", "--cvat-url", type=str, required=True, help="URL of the CVAT server")
    parser.add_argument("-U", "--username", type=str, required=True, help="Username for CVAT server")
    parser.add_argument("-P", "--password", type=str, required=True, help="Password for CVAT server")
    
    parser.add_argument("--format", type=str, required=True, choices=["coco", "pascal_voc", "cvat_xml"], help="Format of the dataset")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory path of the dataset (images and annotations)")
    
    parser.add_argument("--task-name", type=str, required=True, help="Name of the task to create")
    
    parser.add_argument("--zip-file", type=str, required=False, help="Path to a ZIP file containing the dataset (optional)")

    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    print(f"Scanning images in {dataset_dir}...")
    image_paths = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for path in dataset_dir.rglob("*"):
        if path.suffix.lower() in supported_extensions:
            image_paths.append(str(path))
    
    image_paths.sort()
    
    if not image_paths:
        print(f"Error: No images found in {dataset_dir}")
        return

    try:
        upload_file_path = prepare_zip_for_upload(args.format, dataset_dir, args.zip_file)
    except Exception as e:
        print(f"Error preparing annotations: {e}")
        return

    print(f"Connecting to {args.cvat_url} as {args.username}...")
    
    with make_client_with_auth(args.cvat_url, args.username, args.password) as client:
        try:
            print(f"Creating Task: '{args.task_name}' with {len(image_paths)} images...")
            task = client.tasks.create(
                spec={
                    "name": args.task_name,
                },
            )
            print(f" -> Task Created! ID: {task.id}")
            print("Uploading Images...")
            task.upload_data(resources=image_paths)

            cvat_format_map = {
                "coco": "COCO 1.0",
                "pascal_voc": "PASCAL VOC 1.1",
                "cvat_xml": "CVAT for Images 1.1"
            }
            target_format = cvat_format_map[args.format]

            print(f"Uploading Annotations ({target_format})...")
            task.import_annotations(
                format_name=target_format,
                filename=upload_file_path,
            )
            print(" -> Annotations Uploaded Successfully!")
            print(f"\nFinished. Access here: {args.cvat_url}/tasks/{task.id}")

        except Exception as e:
            print(f"\nError occurred during upload: {e}")
        finally:
            if upload_file_path.startswith("temp_") or "temp_" in str(Path(upload_file_path).name):
                p = Path(upload_file_path)
                if args.zip_file is None and p.exists() and p.suffix == ".zip":
                    os.remove(p)
                    print(f"Cleaned up temporary file: {p.name}")

if __name__ == "__main__":
    main()