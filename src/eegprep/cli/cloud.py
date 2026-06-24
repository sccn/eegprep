import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Iterator
import contextlib

class CloudSyncError(Exception):
    pass

class CloudCache:
    def __init__(self):
        self.tmpdir = None
        self._outputs = []

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="eegprep-cloud-")
        self.tmp_path = Path(self.tmpdir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Only upload outputs if no error occurred in the CLI processing
            if exc_type is None:
                self._sync_outputs()
        finally:
            if self.tmpdir:
                shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sync_outputs(self):
        for local_path, remote_uri in self._outputs:
            if local_path.exists():
                print(f"Syncing cloud output {remote_uri}...", file=sys.stderr)
                try:
                    self._sync_up(local_path, remote_uri)
                    # Sync metadata/associated files
                    if local_path.suffix == ".set":
                        fdt_path = local_path.with_suffix(".fdt")
                        if fdt_path.exists():
                            fdt_uri = remote_uri.rsplit(".", 1)[0] + ".fdt"
                            self._sync_up(fdt_path, fdt_uri)
                except subprocess.CalledProcessError as e:
                    raise CloudSyncError(f"Failed to upload {remote_uri}") from e

    def _sync_up(self, local_path: Path, remote_uri: str):
        if local_path.is_dir():
            if remote_uri.startswith("s3://"):
                subprocess.run(["aws", "s3", "sync", str(local_path), remote_uri], check=True, stdout=subprocess.DEVNULL)
            elif remote_uri.startswith("gs://"):
                subprocess.run(["gsutil", "-m", "rsync", "-r", str(local_path), remote_uri], check=True, stdout=subprocess.DEVNULL)
        else:
            if remote_uri.startswith("s3://"):
                subprocess.run(["aws", "s3", "cp", str(local_path), remote_uri], check=True, stdout=subprocess.DEVNULL)
            elif remote_uri.startswith("gs://"):
                subprocess.run(["gsutil", "cp", str(local_path), remote_uri], check=True, stdout=subprocess.DEVNULL)

    def resolve_input(self, uri: str) -> Path:
        if not (uri.startswith("s3://") or uri.startswith("gs://")):
            return Path(uri)
        
        name = uri.split("/")[-1]
        local_dir = self.tmp_path / "in" / str(hash(uri) % 100000)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / name
        
        print(f"Syncing cloud input {uri}...", file=sys.stderr)
        try:
            if uri.startswith("s3://"):
                # Try downloading as file
                res = subprocess.run(["aws", "s3", "cp", uri, str(local_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode != 0:
                    # If it failed, it might be a directory
                    subprocess.run(["aws", "s3", "sync", uri, str(local_path)], check=True, stdout=subprocess.DEVNULL)
                else:
                    # It was a file, check for .fdt
                    if uri.endswith(".set"):
                        fdt_uri = uri[:-4] + ".fdt"
                        fdt_local = local_dir / (name[:-4] + ".fdt")
                        subprocess.run(["aws", "s3", "cp", fdt_uri, str(fdt_local)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif uri.startswith("gs://"):
                res = subprocess.run(["gsutil", "cp", uri, str(local_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode != 0:
                    # If it failed, try as directory
                    subprocess.run(["gsutil", "-m", "rsync", "-r", uri, str(local_path)], check=True, stdout=subprocess.DEVNULL)
                else:
                    if uri.endswith(".set"):
                        fdt_uri = uri[:-4] + ".fdt"
                        fdt_local = local_dir / (name[:-4] + ".fdt")
                        subprocess.run(["gsutil", "cp", fdt_uri, str(fdt_local)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise CloudSyncError(f"Failed to download {uri}") from e
            
        return local_path

    def resolve_output(self, uri: str) -> Path:
        if not (uri.startswith("s3://") or uri.startswith("gs://")):
            return Path(uri)
            
        name = uri.split("/")[-1]
        local_dir = self.tmp_path / "out" / str(hash(uri) % 100000)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / name
        
        self._outputs.append((local_path, uri))
        return local_path

# Global cache instance
_cache = None

@contextlib.contextmanager
def managed_cache():
    global _cache
    with CloudCache() as c:
        _cache = c
        yield c

def resolve_input(path: str | Path) -> Path:
    if isinstance(path, str) and (path.startswith("s3://") or path.startswith("gs://")):
        if _cache is None:
            raise RuntimeError("CloudCache not initialized")
        return _cache.resolve_input(path)
    return Path(path)

def resolve_output(path: str | Path) -> Path:
    if isinstance(path, str) and (path.startswith("s3://") or path.startswith("gs://")):
        if _cache is None:
            raise RuntimeError("CloudCache not initialized")
        return _cache.resolve_output(path)
    return Path(path)

