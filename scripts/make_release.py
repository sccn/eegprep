#!/usr/bin/env python3
"""
Release script for eegprep package.

This script helps maintainers create test and production releases with
appropriate checks and git tagging.

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
from importlib.util import find_spec

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
MAIN_PATH = PROJECT_ROOT / "main"
DIST_DIR = PROJECT_ROOT / "dist"
DOCKERFILE_PATH = PROJECT_ROOT / "DOCKERFILE"

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
    """Extract version from pyproject.toml."""
    try:
        with open(PYPROJECT_PATH, 'r') as f:
            content = f.read()
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception as e:
        print_error(f"Failed to read version from pyproject.toml: {e}")
        sys.exit(1)
    
    print_error("Could not find version in pyproject.toml")
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
            r'^name\s*=\s*["\']([^"\']+)["\']',
            f'name = "{new_name}"',
            content,
            count=1,
            flags=re.MULTILINE
        )
        
        with open(PYPROJECT_PATH, 'w') as f:
            f.write(modified_content)
        
        print_success(f"Temporarily set package name to: {new_name}")
        return True
    except Exception as e:
        print_error(f"Failed to modify package name in pyproject.toml: {e}")
        return False


def get_install_command():
    """Determine the appropriate install command based on the environment."""
    if IS_UV_PROJECT and UV_AVAILABLE:
        return "uv pip install"
    return "pip install"


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

    install_cmd = get_install_command()
    
    # Check if running on Windows
    if platform.system() == "Windows":
        print_warning("Running on Windows. This script is primarily tested on Linux/Mac.")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Exiting.")
            sys.exit(0)
    
    # Check for build package
    if find_spec("build") is None:
        print_error("Package 'build' is not installed.")
        print(f"Install with: {Fore.CYAN}{install_cmd} build{Style.RESET_ALL}")
        sys.exit(1)
    print_success("Package 'build' is installed")
    
    # Check for twine
    if find_spec("twine") is None:
        print_error("Package 'twine' is not installed.")
        print(f"Install with: {Fore.CYAN}{install_cmd} twine{Style.RESET_ALL}")
        sys.exit(1)
    print_success("Package 'twine' is installed")
    
    # Remind about tests
    print_info("Remember to run tests before releasing!")
    print_info("  python -m unittest discover -s tests")


def get_new_version(current_version):
    """Ask user for new version number."""
    print_step(2, "Version Update")
    print(f"Current version in pyproject.toml: {Fore.GREEN}{Style.BRIGHT}{current_version}{Style.RESET_ALL}")
    new_version = input("Enter new version number: ").strip()
    if not new_version:
        print_error("Version cannot be empty")
        sys.exit(1)
    return new_version


def update_version_in_file(file_path, old_version, new_version):
    """Update version in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace version
        updated_content = content.replace(old_version, new_version)
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        print_error(f"Failed to update version in {file_path}: {e}")
        return False


def update_version_files(old_version, new_version):
    """Update version in pyproject.toml and main file."""
    print_step(3, f"Updating version from {old_version} to {new_version}")
    
    # Update pyproject.toml
    print_info(f"Updating pyproject.toml...")
    cmd = f"sed -i '' 's/version = \"{old_version}\"/version = \"{new_version}\"/' {PYPROJECT_PATH}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["sed", "-i", "", f's/version = "{old_version}"/version = "{new_version}"/', str(PYPROJECT_PATH)],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success(f"Updated pyproject.toml")
    except subprocess.CalledProcessError:
        # Fallback to Python method
        if not update_version_in_file(PYPROJECT_PATH, f'version = "{old_version}"', f'version = "{new_version}"'):
            return False
        print_success(f"Updated pyproject.toml")
    
    # Update main file
    print_info(f"Updating main file...")
    cmd = f"sed -i '' 's/eegprep:{old_version}/eegprep:{new_version}/g' {MAIN_PATH}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["sed", "-i", "", f's/eegprep:{old_version}/eegprep:{new_version}/g', str(MAIN_PATH)],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success(f"Updated main file")
    except subprocess.CalledProcessError:
        # Fallback to Python method
        if not update_version_in_file(MAIN_PATH, f'eegprep:{old_version}', f'eegprep:{new_version}'):
            return False
        print_success(f"Updated main file")
    
    return True


def commit_version_changes(version):
    """Commit version changes."""
    print_step(4, f"Committing version changes")
    
    cmd = f"git add {PYPROJECT_PATH} {MAIN_PATH}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["git", "add", str(PYPROJECT_PATH), str(MAIN_PATH)],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success("Staged version files")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to stage files: {e}")
        return False
    
    commit_msg = f"Release version {version}"
    cmd = f'git commit -m "{commit_msg}"'
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=PROJECT_ROOT,
            check=True
        )
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
        print_info(f"Removing old dist directory...")
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
    
    cmd = f"{sys.executable} -m build"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=PROJECT_ROOT,
            check=True
        )
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


def upload_to_testpypi():
    """Upload to TestPyPI using the test package name."""
    print_header("Uploading to TestPyPI")
    
    print_info(f"Using test package name: {TESTPYPI_PACKAGE_NAME}")
    print_info("This avoids conflicts with existing packages on TestPyPI")
    
    # Build command with optional token
    cmd = [sys.executable, "-m", "twine", "upload", "--repository", "testpypi", "dist/*"]
    
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
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            env=env
        )
        print_success(f"Uploaded to TestPyPI successfully as '{TESTPYPI_PACKAGE_NAME}'")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to TestPyPI failed: {e}")
        print_info("Tip: Set TESTPYPI_TOKEN environment variable or configure ~/.pypirc")
        return False


def upload_to_pypi():
    """Upload to PyPI."""
    print_step(6, "Uploading to PyPI")
    
    # Build command with optional token
    cmd = f"{sys.executable} -m twine upload dist/*"
    print(f"Running: {cmd}")
    
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
        subprocess.run(
            [sys.executable, "-m", "twine", "upload", "dist/*"],
            cwd=PROJECT_ROOT,
            check=True,
            env=env
        )
        print_success("Uploaded to PyPI successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to PyPI failed: {e}")
        print_info("Tip: Set PYPI_TOKEN environment variable or configure ~/.pypirc")
        return False


def create_and_push_tag(version):
    """Create and push git tag for production release."""
    print_step(7, "Creating and pushing git tag")
    
    tag_name = f"{version}"
    
    # Create tag
    cmd = f'git tag -a {tag_name} -m "Release version {version}"'
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release version {version}"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success(f"Created git tag: {tag_name}")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create tag: {e}")
        return False
    
    # Push tag
    cmd = f"git push origin {tag_name}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["git", "push", "origin", tag_name],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success(f"Pushed tag {tag_name} to origin")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to push tag: {e}")
        print_info("This might be due to credentials or network issues.")
        print_info(f"To push manually later, run: {Fore.CYAN}git push origin {tag_name}{Style.RESET_ALL}")
        return True  # Continue anyway


def build_and_push_docker(version):
    """Build and push Docker image."""
    print_step(8, f"Building and pushing Docker image")
    
    # Build Docker image
    cmd = f"docker build -t eegprep:{version} -f DOCKERFILE ."
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["docker", "build", "-t", f"eegprep:{version}", "-f", "DOCKERFILE", "."],
            cwd=PROJECT_ROOT,
            check=True
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
            ["docker", "tag", f"eegprep:{version}", f"arnodelorme/eegprep:{version}"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success(f"Tagged Docker image: arnodelorme/eegprep:{version}")
    except subprocess.CalledProcessError as e:
        print_error(f"Docker tag failed: {e}")
        return False
    
    # Push Docker image
    cmd = f"docker push arnodelorme/eegprep:{version}"
    print(f"Running: {cmd}")
    try:
        subprocess.run(
            ["docker", "push", f"arnodelorme/eegprep:{version}"],
            cwd=PROJECT_ROOT,
            check=True
        )
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
        print(f"  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ {TESTPYPI_PACKAGE_NAME}=={version}")
        print()
        print(f"{Fore.CYAN}  After installing, you can still import it as 'eegprep':{Style.RESET_ALL}")
        print(f"  python -c 'import eegprep; print(eegprep.__version__)'")
        print()
    
    if release_type in ['prod', 'both']:
        print(f"{Fore.MAGENTA}To test the PyPI release:{Style.RESET_ALL}")
        print(f"  pip install eegprep=={version}")
        print()
        print(f"{Fore.MAGENTA}Or with all optional dependencies:{Style.RESET_ALL}")
        print(f"  pip install eegprep[all]=={version}")
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
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
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
    
    # Step 2: Get current version and ask for new version
    current_version = get_version()
    new_version = get_new_version(current_version)
    
    # Step 3: Update version files
    if not update_version_files(current_version, new_version):
        sys.exit(1)
    
    # Step 4: Commit version changes
    if not commit_version_changes(new_version):
        sys.exit(1)
    
    # Step 5-8: Build, upload to PyPI, tag, and Docker
    if not build_package():
        sys.exit(1)
    
    if not upload_to_pypi():
        sys.exit(1)
    
    if not create_and_push_tag(new_version):
        sys.exit(1)
    
    if not build_and_push_docker(new_version):
        print_warning("Docker build/push failed, but continuing...")
    
    # Print summary
    print_header("Release Summary")
    print_success(f"Release {new_version} completed successfully!")
    
    # Reminder about brainlife online
    print_step(9, "Next Steps")
    print_warning("REMINDER: Update the default app option on brainlife online")
    print_test_instructions(new_version, 'prod')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Release cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)

