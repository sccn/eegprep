import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from eegprep.cli.cloud import CloudCache, CloudSyncError, managed_cache, resolve_input, resolve_output

def test_cloud_cache_lifecycle():
    with managed_cache() as cache:
        assert cache.tmpdir is not None
        assert Path(cache.tmpdir).exists()
        tmpdir = cache.tmpdir
    # Check cleanup
    assert not Path(tmpdir).exists()

@patch("subprocess.run")
@patch("shutil.which")
def test_resolve_input_s3(mock_which, mock_run):
    mock_which.return_value = "/usr/bin/aws"
    
    # Mock successful download
    mock_run.return_value = MagicMock(returncode=0)
    
    with managed_cache() as cache:
        res = cache.resolve_input("s3://bucket/test.set")
        assert res.name == "test.set"
        
        # Should have called aws s3 cp for .set
        mock_run.assert_any_call(
            ["aws", "s3", "cp", "s3://bucket/test.set", str(res)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # And for .fdt
        mock_run.assert_any_call(
            ["aws", "s3", "cp", "s3://bucket/test.fdt", str(res.with_suffix(".fdt"))],
            capture_output=True, text=True
        )

@patch("subprocess.run")
@patch("shutil.which")
def test_resolve_input_failure(mock_which, mock_run):
    mock_which.return_value = "/usr/bin/aws"
    
    # Mock failure on download
    mock_run.side_effect = subprocess.CalledProcessError(1, "aws")
    
    with managed_cache() as cache:
        with pytest.raises(CloudSyncError, match="Failed to download"):
            cache.resolve_input("s3://bucket/fail.set")

@patch("shutil.which")
def test_cli_not_installed(mock_which):
    mock_which.return_value = None
    
    with managed_cache() as cache:
        with pytest.raises(CloudSyncError, match="aws-cli is not installed"):
            cache.resolve_input("s3://bucket/test.set")

@patch("subprocess.run")
@patch("shutil.which")
def test_resolve_input_missing_fdt_ignored(mock_which, mock_run):
    mock_which.return_value = "/usr/bin/aws"
    
    def side_effect(args, **kwargs):
        if args[0] == "aws" and args[2] == "cp" and args[3].endswith(".fdt"):
            # Return failure with 404
            return MagicMock(returncode=1, stderr="404 Not Found")
        return MagicMock(returncode=0)
    
    mock_run.side_effect = side_effect
    
    with managed_cache() as cache:
        # Should not raise exception
        cache.resolve_input("s3://bucket/test.set")

def test_resolve_input_local():
    with managed_cache() as cache:
        res = cache.resolve_input("/local/path/file.set")
        assert str(res) == "/local/path/file.set"

@patch("subprocess.run")
@patch("shutil.which")
def test_sync_output(mock_which, mock_run):
    mock_which.return_value = "/usr/bin/aws"
    
    with managed_cache() as cache:
        out = cache.resolve_output("s3://bucket/out.set")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        out.with_suffix(".fdt").touch()
        
    # Exiting context should trigger upload
    # Two uploads: .set and .fdt
    assert mock_run.call_count == 2
