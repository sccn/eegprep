#!/usr/bin/env python3
"""
Release script for eegprep package.

This script helps maintainers create test and production releases with
appropriate checks and git tagging.
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
DIST_DIR = PROJECT_ROOT / "dist"


def print_header(text):
    """Print a section header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}{Style.RESET_ALL}\n")


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


def check_prerequisites():
    """Check that required tools are available."""
    print_header("Pre-flight Checks")
    
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
        print("Install with: pip install build")
        sys.exit(1)
    print_success("Package 'build' is installed")
    
    # Check for twine
    if find_spec("twine") is None:
        print_error("Package 'twine' is not installed.")
        print("Install with: pip install twine")
        sys.exit(1)
    print_success("Package 'twine' is installed")
    
    # Check git status
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            print_warning("Git working directory has uncommitted changes or untracked files:")
            print(result.stdout)
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("Exiting. Commit or stash changes before releasing.")
                sys.exit(0)
        else:
            print_success("Git working directory is clean")
    except subprocess.CalledProcessError as e:
        print_warning(f"Could not check git status: {e}")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    # Remind about tests
    print_info("Remember to run tests before releasing!")
    print_info("  python -m unittest discover -s tests")


def confirm_version(version):
    """Confirm the version number with the user."""
    print_header("Version Confirmation")
    print(f"Current version in pyproject.toml: {Fore.GREEN}{Style.BRIGHT}{version}{Style.RESET_ALL}")
    response = input("Is this the correct version to release? [y/N]: ").strip().lower()
    if response != 'y':
        print("\nPlease update the version in pyproject.toml before releasing.")
        sys.exit(0)


def choose_release_type():
    """Let user choose what type of release to make."""
    print_header("Release Type")
    print("Choose release type:")
    print("  a) Test/staging release (TestPyPI only)")
    print("  b) Production release (PyPI + git tag)")
    print("  c) Both (test first, then production)")
    print("  q) Quit")
    
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


def build_package():
    """Build the package."""
    print_header("Building Package")
    clean_dist()
    
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
        
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Build failed: {e}")
        return False


def upload_to_testpypi():
    """Upload to TestPyPI."""
    print_header("Uploading to TestPyPI")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "upload", "--repository", "testpypi", "dist/*"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success("Uploaded to TestPyPI successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to TestPyPI failed: {e}")
        return False


def upload_to_pypi():
    """Upload to PyPI."""
    print_header("Uploading to PyPI")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "twine", "upload", "dist/*"],
            cwd=PROJECT_ROOT,
            check=True
        )
        print_success("Uploaded to PyPI successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Upload to PyPI failed: {e}")
        return False


def create_and_push_tag(version):
    """Create and push git tag for production release."""
    print_header("Git Tagging")
    
    tag_name = f"v{version}"
    
    # Create tag
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
    
    # Ask for confirmation before pushing
    print_info(f"Ready to push tag {tag_name} to origin")
    response = input("Push tag to remote? [y/N]: ").strip().lower()
    
    if response != 'y':
        print_warning(f"Tag {tag_name} created locally but not pushed.")
        print_info(f"To push later, run: git push origin {tag_name}")
        return True
    
    # Try to push tag
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


def print_test_instructions(version, release_type):
    """Print instructions for testing the release."""
    print_header("Testing the Release")
    
    if release_type in ['test', 'both']:
        print(f"{Fore.MAGENTA}To test the TestPyPI release:{Style.RESET_ALL}")
        print(f"  pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ eegprep=={version}")
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
    
    # Get version
    version = get_version()
    
    # Run checks
    check_prerequisites()
    
    # Confirm version
    confirm_version(version)
    
    # Choose release type
    choice = choose_release_type()
    
    if choice == 'q':
        print("Exiting.")
        sys.exit(0)
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Execute based on choice
    success = True
    release_type = None
    
    if choice == 'a':  # Test only
        success = upload_to_testpypi()
        release_type = 'test'
    
    elif choice == 'b':  # Production only
        success = upload_to_pypi()
        if success:
            create_and_push_tag(version)
        release_type = 'prod'
    
    elif choice == 'c':  # Both
        # First test
        if upload_to_testpypi():
            print_success("Test release completed successfully!")
            print_info("Proceeding to production release...")
            input("Press Enter to continue or Ctrl+C to abort...")
            
            # Then production
            if upload_to_pypi():
                create_and_push_tag(version)
                release_type = 'both'
            else:
                success = False
        else:
            success = False
            print_error("Test release failed. Aborting production release.")
    
    # Print summary
    print_header("Release Summary")
    if success:
        print_success(f"Release {version} completed successfully!")
        print_test_instructions(version, release_type)
    else:
        print_error("Release process encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Release cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)

