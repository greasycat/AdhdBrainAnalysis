import boto3
import argparse
import os
from botocore import UNSIGNED
from botocore.config import Config

def list_s3_folders(bucket_name, prefix="", region='us-east-1'):
    """
    List folders in an S3 bucket without authentication (for public buckets)
    """
    
    # Create S3 client without credentials (for public buckets)
    s3_client = boto3.client(
        's3',
        region_name=region,
        config=Config(signature_version=UNSIGNED)
    )
    
    try:
        # List objects with the given prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )
        
        # Extract folder names from CommonPrefixes
        folders = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                folder_name = prefix_info['Prefix']
                folders.append(folder_name)
        
        return folders
        
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return []

def list_s3_objects(bucket_name, prefix="", region='us-east-1'):
    """
    List all objects in an S3 bucket without authentication
    """
    
    s3_client = boto3.client(
        's3',
        region_name=region,
        config=Config(signature_version=UNSIGNED)
    )
    
    try:
        objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append(obj['Key'])
        
        return objects
        
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return []

def parse_s3_uri(s3_uri):
    """
    Parse S3 URI to extract bucket name and prefix
    """
    if not s3_uri.startswith('s3://'):
        raise ValueError("S3 URI must start with 's3://'")
    
    # Remove s3:// prefix
    path = s3_uri[5:]
    
    # Split into bucket and prefix
    parts = path.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    return bucket_name, prefix

def read_id_file(file_path):
    """
    Read IDs from a file, one ID per line
    """
    try:
        with open(file_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        return ids
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

def download_s3_folder(s3_client, bucket_name, s3_prefix, local_path):
    """
    Download an entire S3 folder to local directory
    
    Parameters:
    - s3_client: Boto3 S3 client
    - bucket_name: S3 bucket name
    - s3_prefix: S3 prefix (folder path)
    - local_path: Local directory to download to
    """
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip if it's just a folder marker
                    if s3_key.endswith('/'):
                        continue
                    
                    # Calculate local file path
                    relative_path = s3_key[len(s3_prefix):]
                    local_file_path = os.path.join(local_path, relative_path)
                    
                    # Check if file already exists
                    if os.path.exists(local_file_path):
                        print(f"Skipping (exists): {s3_key} -> {local_file_path}")
                        continue
                    
                    # Create subdirectories if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download the file
                    s3_client.download_file(bucket_name, s3_key, local_file_path)
                    print(f"Downloaded: {s3_key} -> {local_file_path}")
        
        return True
    except Exception as e:
        print(f"Error downloading folder {s3_prefix}: {e}")
        return False

def download_s3_files(s3_client, bucket_name, s3_prefix, local_path, file_pattern):
    """
    Download specific files from S3 folder matching a pattern
    
    Parameters:
    - s3_client: Boto3 S3 client
    - bucket_name: S3 bucket name
    - s3_prefix: S3 prefix (folder path)
    - local_path: Local directory to download to
    - file_pattern: Pattern to match files (e.g., "sub-123_task-rest")
    """
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)
        
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        downloaded_count = 0
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    filename = os.path.basename(s3_key)
                    
                    # Check if file matches the pattern
                    if filename.startswith(file_pattern):
                        # Calculate local file path
                        local_file_path = os.path.join(local_path, filename)
                        
                        # Check if file already exists
                        if os.path.exists(local_file_path):
                            print(f"Skipping (exists): {s3_key} -> {local_file_path}")
                            continue
                        
                        # Download the file
                        s3_client.download_file(bucket_name, s3_key, local_file_path)
                        print(f"Downloaded: {s3_key} -> {local_file_path}")
                        downloaded_count += 1
        
        print(f"Downloaded {downloaded_count} files matching pattern '{file_pattern}'")
        return downloaded_count > 0
        
    except Exception as e:
        print(f"Error downloading files with pattern {file_pattern}: {e}")
        return False

def download_qc_html(s3_client, bucket_name, prefix, subject_id, output_dir, derivative_type):
    """
    Download quality check HTML file for fmriprep
    
    Parameters:
    - s3_client: Boto3 S3 client
    - bucket_name: S3 bucket name
    - prefix: S3 prefix
    - subject_id: Subject ID
    - output_dir: Local output directory
    - derivative_type: Type of derivatives (only downloads if "fmriprep")
    
    Returns:
    - bool: True if downloaded successfully, False otherwise
    """
    if derivative_type != "fmriprep":
        return False
    
    try:
        print("Downloading quality check HTML...")
        qc_s3_key = f"{prefix}{derivative_type}/{subject_id}.html"
        qc_local_path = os.path.join(output_dir, f"{subject_id}", f"{subject_id}.html")
        
        # Check if file already exists
        if os.path.exists(qc_local_path):
            print(f"Skipping (exists): {qc_s3_key} -> {qc_local_path}")
            return True
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(qc_local_path), exist_ok=True)
        
        # Download the HTML file
        s3_client.download_file(bucket_name, qc_s3_key, qc_local_path)
        print(f"Downloaded: {qc_s3_key} -> {qc_local_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading quality check HTML: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='List S3 folders and process subject IDs')
    parser.add_argument('--ids', nargs='+', required=True,
                       help='One or more paths to files containing subject IDs (one per line)')
    parser.add_argument('--type', required=True,
                       help='Type of derivatives (e.g., "fmriprep", "freesurfer", "task")')
    parser.add_argument('--tasks', nargs='+', default=['rest'],
                       help='List of tasks to process (e.g., "rest", "scap", "stopsignal")')
    parser.add_argument('--s3-uri', default="s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/",
                       help='S3 URI to list folders from')
    parser.add_argument('--output-dir', default="./downloads",
                       help='Local directory to download files to')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: download only the first subject ID')
    
    args = parser.parse_args()
    
    # Read ID files
    print("Reading subject ID files...")
    all_ids = []
    file_counts = []
    
    for i, id_file in enumerate(args.ids):
        ids = read_id_file(id_file)
        all_ids.extend(ids)
        file_counts.append(len(ids))
        print(f"File {i+1} ({id_file}): {len(ids)} subjects")
    
    print(f"Total subjects across all files: {len(all_ids)}")
    print(f"Data type: {args.type}")
    print(f"Tasks: {args.tasks}")
    print(f"Output directory: {args.output_dir}")
    
    # Create S3 client
    s3_client = boto3.client(
        's3',
        region_name='us-east-1',
        config=Config(signature_version=UNSIGNED)
    )
    
    bucket_name, prefix = parse_s3_uri(args.s3_uri)
    prefix_derivatives = f"{prefix}derivatives/"
    

    print(f"S3 prefix: {prefix_derivatives}")
    
    # Test mode: only download first subject
    if args.test:
        if all_ids:
            all_ids = [all_ids[0]]
            print(f"\nTEST MODE: Downloading only the first subject: {all_ids[0]}")
        else:
            print("ERROR: No subject IDs found for test mode")
            return
    
    # Download data for each participant
    print(f"\nStarting downloads for {len(all_ids)} participants...")
    
    for i, subject_id in enumerate(all_ids, 1):
        print(f"\n--- Processing participant {i}/{len(all_ids)}: {subject_id} ---")
        
        anat_s3_prefix = f"{prefix_derivatives}{args.type}/{subject_id}/anat/"
        anat_local_path = os.path.join(args.output_dir, f"{subject_id}", "anat")
        
        print("Downloading anatomical data...")
        download_s3_folder(s3_client, bucket_name, anat_s3_prefix, anat_local_path)
        
        func_s3_prefix = f"{prefix_derivatives}{args.type}/{subject_id}/func/"
        func_local_path = os.path.join(args.output_dir, f"{subject_id}", "func")
        
        for task in args.tasks:
            print(f"Downloading functional data for task: {task}")
            file_pattern = f"{subject_id}_task-{task}"
            download_s3_files(s3_client, bucket_name, func_s3_prefix, func_local_path, file_pattern)

            task_analysis_s3_prefix = f"{prefix_derivatives}{'task'}/{subject_id}/{task}.feat/"
            task_analysis_local_path = os.path.join(args.output_dir, "task", f"{subject_id}",  f"{task}.feat")
            if not os.path.exists(task_analysis_local_path):
                # mkdir
                os.makedirs(task_analysis_local_path, exist_ok=True)
            print(f"Downloading task analysis data for task: {task_analysis_s3_prefix} to {task_analysis_local_path}")
            download_s3_folder(s3_client, bucket_name, task_analysis_s3_prefix, task_analysis_local_path)
        
        download_qc_html(s3_client, bucket_name, prefix_derivatives, subject_id, args.output_dir, args.type)

    
    print(f"\nDownload completed! Files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
