"""Upload the app to the Hugging Face Space, which then builds and runs the Dockerfile.

Run by CI after tests and the Docker build pass on main. Needs HF_TOKEN (a write token).
Only files the app needs are uploaded: no tests, evals, docs, or intake photos.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "itsxiaoshen/rescuepaws-ai"
ROOT = Path(__file__).resolve().parent.parent
APP_FILES = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "src/**",
    "app/**",
    "data/animals_demo.json",
    "data/record_conflicts.json",
    "data/shelter_policies/**",
]


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    commit = os.getenv("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        repo_id=SPACE_ID,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=APP_FILES,
        ignore_patterns=["**/__pycache__/**"],
        delete_patterns=["src/**", "app/**", "data/**"],  # remove files deleted from the repo
        commit_message=f"Deploy {commit} from GitHub",
    )
    api.upload_file(
        repo_id=SPACE_ID,
        repo_type="space",
        path_or_fileobj=ROOT / "deploy" / "huggingface_README.md",
        path_in_repo="README.md",
        commit_message=f"Update Space README ({commit})",
    )
    print(f"Deployed {commit} to https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
