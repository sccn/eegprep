#!/usr/bin/env python3
"""
Release script for eegprep package.

This script helps maintainers create test and production releases with
appropriate checks and git tagging.

Version source:
    `pyproject.toml` declares `dynamic = ["version"]`, so the single source of
    truth is `__version__` in src/eegprep/__init__.py. That file and the pinned
    image tag in tools/hpc/main.pbs are what this script rewrites.

Build tool:
    Builds run through `uv build`. `python -m build` cannot work from this repo,
    because the `build/` output directory shadows the `build` package on sys.path.

TestPyPI Package Naming:
    TestPyPI releases use the package name 'eegprep_test' to avoid conflicts
    with the existing package owned by a previous maintainer. The production
    PyPI releases use the regular 'eegprep' package name.

Authentication:
    You can provide PyPI credentials in three ways:

    1. ~/.pypirc file (recommended for interactive use):
        [testpypi]
        repository = https://test.pypi.org/legacy/
        username = __token__
        password = pypi-...your-token...

        [pypi]
        username = __token__
        password = pypi-...your-token...

    2. Environment variables (recommended for CI/CD):
        TESTPYPI_TOKEN or TWINE_PASSWORD_TESTPYPI - TestPyPI API token
        PYPI_TOKEN or TWINE_PASSWORD - PyPI API token

    3. Enter them when prompted.
"""

import os
import sys
import subprocess
import shutil
import platform
import re
from pathlib import Path

# Use colorama for colored output (already a dependency)
try:
    from colorama import init, Fore, Style

    init(autoreset=True)
except ImportError:
    # Fallback if colorama not available
    class Fore:
        RED = GREEN = YELLOW = CYAN = BLUE = MAGENTA = ""

    class Style:
        BRIGHT = RESET_ALL = ""


# Find project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
# pyproject declares `dynamic = ["version"]` and reads it from this attribute, so
# __init__.py is the single source of truth for the version.
VERSION_PATH = PROJECT_ROOT / "src" / "eegprep" / "__init__.py"
MAIN_PATH = PROJECT_ROOT / "tools" / "hpc" / "main.pbs"
DIST_DIR = PROJECT_ROOT / "dist"
DOCKERFILE_NAME = "Dockerfile"

# Test package name for TestPyPI (to avoid conflicts with existing package)
TESTPYPI_PACKAGE_NAME = "eegprep_test"

# Detect if this is a uv-managed project
IS_UV_PROJECT = (PROJECT_ROOT / "uv.lock").exists()
UV_AVAILABLE = shutil.which("uv") is not None


def print_header(text):
    """Print a section header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'*' * 40}")
    print(f"{text}")
    print(f"{'*' * 40}{Style.RESET_ALL}\n")


def print_step(step_num, text):
    """Print a step header with clear delineation."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'*' * 40}")
    print(f"Step {step_num}: {text}")
    print(f"{'*' * 40}{Style.RESET_ALL}\n")


def print_success(text):
    """Print a success message."""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text):
    """Print a warning message."""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


def print_error(text):
    """Print an error message."""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text):
    """Print an info message."""
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")


def get_version():
    """Extract __version__ from src/eegprep/__init__.py."""
    try:
        content = VERSION_PATH.read_text()
    except Exception as e:
        print_error(f"Failed to read version from {VERSION_PATH}: {e}")
        sys.exit(1)

    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)

    print_error(f"Could not find __version__ in {VERSION_PATH}")
    sys.exit(1)


def get_package_name():
    """Extract package name from pyproject.toml."""
    try:
        with open(PYPROJECT_PATH, 'r') as f:
            content = f.read()
            match = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception as e:
        print_error(f"Failed to read package name from pyproject.toml: {e}")
        sys.exit(1)

    print_error("Could not find package name in pyproject.toml")
    sys.exit(1)


def set_package_name(new_name):
    """Temporarily change the package name in pyproject.toml."""
    try:
        with open(PYPROJECT_PATH, 'r') as f:
            content = f.read()

        # Replace the name field
        modified_content = re.sub(
            r'^name\s*=\s*["\']([^"\']+)["\']', f'name = "{new_name}"', content, count=1, flags=re.MULTILINE
        )

        with open(PYPROJECT_PATH, 'w') as f:
            f.write(modified_content)

        print_success(f"Temporarily set package name to: {new_name}")
        return True
    except Exception as e:
        print_error(f"Failed to modify package name in pyproject.toml: {e}")
        return False


def get_install_command(package_name):
    """Determine the appropriate install command based on the environment."""
    if IS_UV_PROJECT and UV_AVAILABLE:
        return "uv sync --group release"
    return f"pip install {package_name}"


def check_prerequisites():
    """Check that required tools are available."""
    print_header("Pre-flight Checks")

    # Show environment info
    print_info(f"Python executable: {sys.executable}")
    if IS_UV_PROJECT:
        print_info("Detected uv-managed project (uv.lock present)")
        if UV_AVAILABLE:
            print_success("uv is available")
        else:
            print_warning("uv is not available in PATH but project uses uv")

    # Check if running on Windows
    if platform.system() == "Windows":
        print_warning("Running on Windows. This script is primarily tested on Linux/Mac.")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Exiting.")
            sys.exit(0)

    # Builds go through `uv build`. `python -m build` cannot work from this repo:
    # PROJECT_ROOT holds a `build/` output directory that shadows the `build`
    # package on sys.path, so the module resolves to the directory and fails with
    # "'build' is a package and cannot be directly executed" no matter what is
    # installed. That shadowing also makes find_spec("build") a false positive.
    if not UV_AVAILABLE:
        print_error("uv is required to build the package.")
        print(f"Install it from: {Fore.CYAN}https://docs.astral.sh/uv/{Style.RESET_ALL}")
        sys.exit(1)
    print_success("uv is available for building")

    # Check for twine. Resolve it as a runnable module rather than with find_spec,
    # so a stale directory or partial install cannot pass the check.
    twine_probe = subprocess.run(
        [sys.executable, "-m", "twine", "--version"], cwd=SCRIPT_DIR, capture_output=True, text=True
    )
    if twine_probe.returncode != 0:
        print_error("Package 'twine' is not installed (or not runnable).")
        print(f"Install with: {Fore.CYAN}{get_install_command('twine')}{Style.RESET_ALL}")
        sys.exit(1)
    print_success(f"twine is available ({twine_probe.stdout.strip().splitlines()[0]})")

    # Remind about tests
    print_info("Remember to run tests before releasing!")
    print_info("  uv run pytest tests")


def get_new_version(current_version):
    """Ask user for new version number."""
    print_step(2, "Version Update")
    print(f"Current version in pyproject.toml: {Fore.GREEN}{Style.BRIGHT}{current_version}{Style.RESET_ALL}")
    new_version = input("Enter new version number: ").strip()
    if not new_version:
        print_error("Version cannot be empty")
        sys.exit(1)
    return new_version


def replace_once(file_path, old_text, new_text):
    """Replace the first occurrence of old_text in file_path; report if absent."""
    try:
        content = Path(file_path).read_text()
    except Exception as e:
        print_error(f"Failed to read {file_path}: {e}")
        return False

    if old_text not in content:
        print_error(f"Expected to find {old_text!r} in {file_path}")
        return False

    try:
        Path(file_path).write_text(content.replace(old_text, new_text, 1))
    except Exception as e:
        print_error(f"Failed to write {file_path}: {e}")
        return False
    return True


def update_version_files(old_version, new_version):
    """Update __version__ and the HPC wrapper's pinned image tag."""
    print_step(3, f"Updating version from {old_version} to {new_version}")

    print_info(f"Updating {VERSION_PATH.relative_to(PROJECT_ROOT)}...")
    if not replace_once(VERSION_PATH, f'__version__ = "{old_version}"', f'__version__ = "{new_version}"'):
        return False
    print_success("Updated __version__")

    # The HPC wrapper pins the published image tag, so it must track the version.
    print_info(f"Updating {MAIN_PATH.relative_to(PROJECT_ROOT)}...")
    main_text = MAIN_PATH.read_text()
    if f"eegprep:{old_version}" not in main_text:
        print_warning(f"No 'eegprep:{old_version}' pin found in {MAIN_PATH.name}; leaving it unchanged")
    else:
        MAIN_PATH.write_text(main_text.replace(f"eegprep:{old_version}", f"eegprep:{new_version}"))
        print_success("Updated HPC wrapper")

    return True


def commit_version_changes(version):
    """Commit version changes."""
    print_step(4, "Committing version changes")

    cmd = f"git add {VERSION_PATH} {MAIN_PATH}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(["git", "add", str(VERSION_PATH), str(MAIN_PATH)], cwd=PROJECT_ROOT, check=True)
        print_success("Staged version files")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to stage files: {e}")
        return False

    commit_msg = f"Release version {version}"
    cmd = f'git commit -m "{commit_msg}"'
    print(f"Running: {cmd}")
    try:
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_ROOT, check=True)
        print_success(f"Committed version changes: {commit_msg}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to commit: {e}")
        return False


def choose_release_type():
    """Let user choose what type of release to make."""
    print_header("Release Type")
    print("Choose release type:")
    print(f"  a) Test/staging release (TestPyPI only, as '{TESTPYPI_PACKAGE_NAME}')")
    print("  b) Production release (PyPI + git tag)")
    print("  c) Both (test first, then production)")
    print("  q) Quit")
    print()
    print_info(f"TestPyPI will use package name '{TESTPYPI_PACKAGE_NAME}' to avoid conflicts")
    print_info("Production PyPI will use the regular package name 'eegprep'")

    while True:
        choice = input("\nYour choice [a/b/c/q]: ").strip().lower()
        if choice in ['a', 'b', 'c', 'q']:
            return choice
        print_error("Invalid choice. Please enter a, b, c, or q.")


def clean_dist():
    """Remove old dist directory."""
    if DIST_DIR.exists():
        print_info("Removing old dist directory...")
        shutil.rmtree(DIST_DIR)
        print_success("Old dist directory removed")


def build_package(package_name=None):
    """Build the package.

    Args:
        package_name: Optional package name to use. If provided, temporarily
                      modifies pyproject.toml before building.
    """
    print_step(5, "Building Package")
    clean_dist()

    original_name = None
    if package_name:
        original_name = get_package_name()
        if original_name != package_name:
            print_info(f"Building with package name: {package_name}")
            if not set_package_name(package_name):
                return False

    build_cmd = ["uv", "build"]
    print(f"Running: {' '.join(build_cmd)}")
    try:
        subprocess.run(build_cmd, cwd=PROJECT_ROOT, check=True)
        print_success("Package built successfully")

        # Show what was built
        if DIST_DIR.exists():
            files = list(DIST_DIR.glob("*"))
            if files:
                print_info("Built files:")
                for f in files:
                    print(f"  - {f.name}")

        # Restore original name if it was changed
        if original_name and original_name != package_name:
            set_package_name(original_name)
            print_info(f"Restored package name to: {original_name}")

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Build failed: {e}")

        # Restore original name on error too
        if original_name and original_name != package_name:
            set_package_name(original_name)

        return False


def dist_files():
    """Return the built artifacts as explicit paths.

    The shell is not involved when running through subprocess, so a literal
    "dist/*" argument would not be expanded here.
    """
    files = sorted(str(p) for p in DIST_DIR.glob("*") if p.suffix in {".whl", ".gz"})
    if not files:
        print_error(f"No artifacts found in {DIST_DIR}. Build first.")
        sys.exit(1)
    return files


def upload_to_testpypi():
    """Upload to TestPyPI using the test package name."""
    print_header("Uploading to TestPyPI")

    print_info(f"Using test package name: {TESTPYPI_PACKAGE_NAME}")
    print_info("This avoids conflicts with existing packages on TestPyPI")

    # Build command with optional token
    cmd = [sys.executable, "-m", "twine", "upload", "--repository", "testpypi", *dist_files()]

    # Check if token is provided via environment variable
    token = os.environ.get("TWINE_PASSWORD_TESTPYPI") or os.environ.get("TESTPYPI_TOKEN")
    if token:
        print_info("Using API token from environment variable")
        # Set environment for subprocess
        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = token
    else:
        print_info("Using credentials from ~/.pypirc or will prompt")
        env = None

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        print_success(f"Uploaded to TestPyPI successfully as '{TESTPYPI_PACKAGE_NAME}'")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to TestPyPI failed: {e}")
        print_info("Tip: Set TESTPYPI_TOKEN environment variable or configure ~/.pypirc")
        return False


def upload_to_pypi():
    """Upload to PyPI."""
    print_step(6, "Uploading to PyPI")

    cmd = [sys.executable, "-m", "twine", "upload", *dist_files()]
    print(f"Running: {' '.join(cmd)}")

    # Check if token is provided via environment variable
    token = os.environ.get("TWINE_PASSWORD") or os.environ.get("PYPI_TOKEN")
    if token:
        print_info("Using API token from environment variable")
        # Set environment for subprocess
        env = os.environ.copy()
        env["TWINE_USERNAME"] = "__token__"
        env["TWINE_PASSWORD"] = token
    else:
        print_info("Using credentials from ~/.pypirc or will prompt")
        env = None

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        print_success("Uploaded to PyPI successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to PyPI failed: {e}")
        print_info("Tip: Set PYPI_TOKEN environment variable or configure ~/.pypirc")
        return False


def push_git_changes():
    """Push all committed changes to remote."""
    print_step(7, "Pushing git changes")

    cmd = "git push"
    print(f"Running: {cmd}")
    try:
        subprocess.run(["git", "push"], cwd=PROJECT_ROOT, check=True)
        print_success("Pushed git changes successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to push git changes: {e}")
        print_info("This might be due to credentials or network issues.")
        print_info(f"To push manually later, run: {Fore.CYAN}git push{Style.RESET_ALL}")
        return False


def create_and_push_tag(version):
    """Create and push git tag for production release."""
    print_step(8, "Creating and pushing git tag")

    # Match the newest published tags, which carry the `v` prefix (v0.2.23).
    tag_name = f"v{version}"

    # Create tag
    cmd = f'git tag -a {tag_name} -m "Release version {version}"'
    print(f"Running: {cmd}")
    try:
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release version {version}"], cwd=PROJECT_ROOT, check=True)
        print_success(f"Created git tag: {tag_name}")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create tag: {e}")
        return False

    # Push tag
    cmd = f"git push origin {tag_name}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(["git", "push", "origin", tag_name], cwd=PROJECT_ROOT, check=True)
        print_success(f"Pushed tag {tag_name} to origin")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to push tag: {e}")
        print_info("This might be due to credentials or network issues.")
        print_info(f"To push manually later, run: {Fore.CYAN}git push origin {tag_name}{Style.RESET_ALL}")
        return True  # Continue anyway


def build_and_push_docker(version):
    """Build and push Docker image."""
    print_step(9, "Building and pushing Docker image")

    # Build Docker image. Use the tracked filename's exact casing: macOS is
    # case-insensitive so "DOCKERFILE" resolves locally, but it fails on Linux.
    cmd = f"docker build -t eegprep:{version} -f {DOCKERFILE_NAME} ."
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["docker", "build", "-t", f"eegprep:{version}", "-f", DOCKERFILE_NAME, "."], cwd=PROJECT_ROOT, check=True
        )
        print_success(f"Built Docker image: eegprep:{version}")
    except subprocess.CalledProcessError as e:
        print_error(f"Docker build failed: {e}")
        return False

    # Tag Docker image
    cmd = f"docker tag eegprep:{version} arnodelorme/eegprep:{version}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["docker", "tag", f"eegprep:{version}", f"arnodelorme/eegprep:{version}"], cwd=PROJECT_ROOT, check=True
        )
        print_success(f"Tagged Docker image: arnodelorme/eegprep:{version}")
    except subprocess.CalledProcessError as e:
        print_error(f"Docker tag failed: {e}")
        return False

    # Push Docker image
    cmd = f"docker push arnodelorme/eegprep:{version}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(["docker", "push", f"arnodelorme/eegprep:{version}"], cwd=PROJECT_ROOT, check=True)
        print_success(f"Pushed Docker image: arnodelorme/eegprep:{version}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Docker push failed: {e}")
        print_info("Make sure you're logged in to Docker Hub: docker login")
        return False


def print_test_instructions(version, release_type):
    """Print instructions for testing the release."""
    print_header("Testing the Release")

    if release_type in ['test', 'both']:
        print(f"{Fore.MAGENTA}To test the TestPyPI release:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  NOTE: The test package is named '{TESTPYPI_PACKAGE_NAME}' on TestPyPI{Style.RESET_ALL}")
        print(
            f"  uv pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ {TESTPYPI_PACKAGE_NAME}=={version}"
        )
        print()
        print(f"{Fore.CYAN}  After installing, you can still import it as 'eegprep':{Style.RESET_ALL}")
        print("  python -c 'import eegprep; print(eegprep.__version__)'")
        print()

    if release_type in ['prod', 'both']:
        print(f"{Fore.MAGENTA}To test the PyPI release:{Style.RESET_ALL}")
        print(f"  uv pip install eegprep=={version}")
        print()
        print(f"{Fore.MAGENTA}Or with all optional dependencies:{Style.RESET_ALL}")
        print(f"  uv pip install 'eegprep[all]=={version}'")
        print()


def main():
    """Main release workflow."""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                  EEGPrep Release Script                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    # Step 1: Check for uncommitted changes
    print_step(1, "Checking for uncommitted changes")
    cmd = "git status | grep modified"
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )

        if result.stdout.strip():
            # Filter out src/eegprep/eeglab changes
            modified_lines = []
            for line in result.stdout.strip().split('\n'):
                if 'src/eegprep/eeglab' not in line:
                    modified_lines.append(line)

            if modified_lines:
                print_warning("Found uncommitted changes (excluding src/eegprep/eeglab):")
                for line in modified_lines:
                    print(f"  {line}")
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Exiting. Commit or stash changes before releasing.")
                    sys.exit(0)
            else:
                print_success("No uncommitted changes (ignoring src/eegprep/eeglab)")
        else:
            print_success("No uncommitted changes")
    except subprocess.CalledProcessError as e:
        print_warning(f"Could not check git status: {e}")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            sys.exit(0)

    # Run other checks
    check_prerequisites()

    choice = choose_release_type()
    if choice == 'q':
        print("Exiting.")
        sys.exit(0)
    do_test = choice in ('a', 'c')
    do_prod = choice in ('b', 'c')

    # Step 2: Get current version and ask for new version
    current_version = get_version()
    new_version = get_new_version(current_version)

    # Step 3: Update version files
    if not update_version_files(current_version, new_version):
        sys.exit(1)

    # Step 4: Commit version changes. A TestPyPI-only run is a throwaway upload, so
    # it does not get a release commit; the bumped files are left in the worktree.
    if do_prod:
        if not commit_version_changes(new_version):
            sys.exit(1)
    else:
        print_warning("Test-only release: version files were bumped but NOT committed.")

    # Step 5: TestPyPI leg, built under the test package name
    if do_test:
        if not build_package(TESTPYPI_PACKAGE_NAME):
            sys.exit(1)
        if not upload_to_testpypi():
            sys.exit(1)
        if do_prod:
            response = input("\nTestPyPI upload done. Continue to production PyPI? [y/N]: ").strip().lower()
            if response != 'y':
                print("Stopping before production release.")
                print_test_instructions(new_version, 'test')
                return

    # Step 5-9: Build, upload to PyPI, push changes, tag, and Docker
    if do_prod:
        if not build_package():
            sys.exit(1)

        if not upload_to_pypi():
            sys.exit(1)

        if not push_git_changes():
            sys.exit(1)

        if not create_and_push_tag(new_version):
            sys.exit(1)

        if not build_and_push_docker(new_version):
            print_warning("Docker build/push failed, but continuing...")

    # Print summary
    print_header("Release Summary")
    print_success(f"Release {new_version} completed successfully!")

    # Reminder about brainlife online
    print_step(10, "Next Steps")
    if do_prod:
        print_warning("REMINDER: Update the default app option on brainlife online")
    print_test_instructions(new_version, 'both' if (do_test and do_prod) else ('prod' if do_prod else 'test'))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Release cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)
