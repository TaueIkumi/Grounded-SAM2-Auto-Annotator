import argparse
import os
import zipfile
import shutil
from pathlib import Path
from cvat_sdk import make_client

def make_client_with_auth(url: str, username: str, password: str):
    """Create a client connection to the CVAT server."""
    try:
        client = make_client(host=url, credentials=(username, password))
    except Exception as e:
        raise ConnectionError(f"Failed to connect to CVAT server: {e}")
    return client

def get_image_paths(format_name: str, dataset_dir: Path):
    supported_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []

    if format_name == "pascal_voc":
        jpeg_dir = dataset_dir / "JPEGImages"
        search_root = jpeg_dir if jpeg_dir.exists() else dataset_dir
    else:
        search_root = dataset_dir

    print(f"[{format_name}] Scanning images: {search_root}")

    for path in search_root.rglob("*"):
        if path.suffix.lower() in supported_extensions:
            image_paths.append(str(path))
    
    image_paths.sort()
    return image_paths

def prepare_zip_for_upload(format_name: str, dataset_dir: Path) -> str:
    """
    Prepare annotation files according to the format and return the path.
    """
    dataset_dir = Path(dataset_dir)
    
    if format_name == "coco":
        # COCO format returns a single JSON file
        candidates = [
            dataset_dir / "annotations" / "instances.json",
            dataset_dir / "instances.json"
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        
        # Recursively search if not found
        found = list(dataset_dir.rglob("*.json"))
        if found:
            return str(found[0])
        raise FileNotFoundError(f"COCO annotation (instances.json) not found: {dataset_dir}")

    elif format_name == "pascal_voc":
        zip_filename = "temp_pascal_voc_upload.zip"        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    if file.lower().endswith(('.xml', '.txt')):
                        file_path = Path(root) / file
                        archive_name = file_path.relative_to(dataset_dir)
                        zf.write(file_path, archive_name)
        return str(Path(zip_filename).resolve())

    elif format_name == "cvat_xml":
        xml_files = list(dataset_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"CVAT XML files not found: {dataset_dir}")
        return str(xml_files[0])

    else:
        raise ValueError(f"Unsupported format: {format_name}")

def main():
    parser = argparse.ArgumentParser(description="CVAT Dataset Uploader (Multi-Format Support)")
    
    parser.add_argument("-u", "--cvat-url", type=str, required=True, help="CVAT server URL")
    parser.add_argument("-U", "--username", type=str, required=True, help="Username")
    parser.add_argument("-P", "--password", type=str, required=True, help="Password")
    
    parser.add_argument("--format", type=str, required=True, choices=["coco", "pascal_voc", "cvat_xml"], help="Dataset format")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset directory path")
    
    parser.add_argument("--task-name", type=str, required=True, help="Task name to create")
    parser.add_argument("--labels", type=str, nargs='+', help="List of label names (e.g., --labels car person)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Directory not found: {dataset_dir}")
        return

    # 1. Get image list according to format
    image_paths = get_image_paths(args.format, dataset_dir)
    if not image_paths:
        print(f"Error: No images found.")
        return

    # 2. Prepare annotation files
    try:
        upload_file_path = prepare_zip_for_upload(args.format, dataset_dir)
    except Exception as e:
        print(f"Error: Failed to prepare annotations: {e}")
        return

    # 3. CVAT operations
    print(f"Connecting to CVAT: {args.cvat_url}...")
    with make_client_with_auth(args.cvat_url, args.username, args.password) as client:
        try:
            if not args.labels:
                print("Error: --labels is required for creating a new task.")
                return

            # Configure default detection type (Bbox)
            labels_spec = [
                {"name": name, "type": "rectangle"} for name in args.labels
            ]

            print(f"Creating task: '{args.task_name}' (Number of images: {len(image_paths)})")
            task = client.tasks.create(
                spec={
                    "name": args.task_name,
                    "labels": labels_spec,
                },
            )
            print(f" -> Task created successfully! ID: {task.id}")

            # Upload images
            print("Uploading images...")
            task.upload_data(resources=image_paths)

            # Import annotations
            cvat_format_map = {
                "coco": "COCO 1.0",
                "pascal_voc": "PASCAL VOC 1.1",
                "cvat_xml": "CVAT for Images 1.1"
            }
            target_format = cvat_format_map[args.format]

            print(f"Importing annotations ({target_format})...")
            task.import_annotations(
                format_name=target_format,
                filename=upload_file_path,
            )
            
            print("Success!")
            print(f"Access URL: {args.cvat_url}/tasks/{task.id}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()