"""Git utilities."""

import subprocess
import os


__all__ = ['get_git_commit_id']


def get_git_commit_id(repo_path: str = None, shorten: int = 8) -> str | None:
    """Get the current commit ID (hash) of a Git repository.

    Args:
        repo_path: The path to the Git repository. If None, it uses the
                   current working directory.
        shorten: The length to which the commit ID should be shortened.
                 If set to 0, the full commit ID is returned.

    Returns:
        The commit ID as a string, or None if it's not a Git repository
        or an error occurs.
    """
    if repo_path is None:
        # If no path is specified, use the current working directory.
        repo_path = os.getcwd()

    try:
        # The command to get the full commit hash of the current HEAD
        command = ['git', 'rev-parse', 'HEAD']

        # Execute the command
        commit_result = subprocess.run(
            command,
            cwd=repo_path,  # Run command in the specified directory
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text
            check=True  # Raise CalledProcessError if command fails
        )


        # 2. Check for dirty status using the porcelain format
        # This command outputs nothing if the working tree is clean.
        status_cmd = ['git', 'status', '--porcelain']
        status_result = subprocess.run(
            status_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        # The output is the commit hash, with a trailing newline
        hash = commit_result.stdout.strip()
        if shorten > 0:
            # Shorten the hash to the specified length
            hash = hash[:shorten]

        # If there is any status output, the directory is dirty
        if status_result.stdout:
            hash += " (dirty)"

        return hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError: 'git' command returned a non-zero exit code
        # (e.g., not a git repo).
        # FileNotFoundError: 'git' command was not found.
        return None
