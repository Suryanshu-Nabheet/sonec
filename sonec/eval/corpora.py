"""Open-source corpus sync — shallow clones for hard live agent work.

Default targets are small-to-medium public trees used as real workspaces
(not full VS Code/Bun monorepos, which are too large for laptop rollouts).
Full giant trees stay optional via YAML overrides.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CORPORA: list[dict[str, Any]] = [
    {
        "id": "vscode-extension-samples",
        "url": "https://github.com/microsoft/vscode-extension-samples.git",
        "ref": "main",
        "depth": 1,
        "notes": "Official VS Code extension samples — IDE agent surface",
    },
    {
        "id": "openai-codex",
        "url": "https://github.com/openai/codex.git",
        "ref": "main",
        "depth": 1,
        "notes": "OpenAI Codex CLI — agent CLI patterns",
    },
    {
        "id": "oven-bun",
        "url": "https://github.com/oven-sh/bun.git",
        "ref": "main",
        "depth": 1,
        "sparse": ["docs", "test", "packages/bun-types"],
        "notes": "Bun runtime (sparse) — package/runtime patterns",
        "optional": True,
    },
]


class CorpusRepo(BaseModel):
    id: str
    url: str
    ref: str = "main"
    depth: int = 1
    sparse: list[str] = Field(default_factory=list)
    notes: str = ""
    optional: bool = False


class CorporaManifest(BaseModel):
    root: Path
    repos: list[CorpusRepo] = Field(default_factory=list)

    def path_for(self, repo_id: str) -> Path:
        return self.root / repo_id


def default_manifest(root: Path) -> CorporaManifest:
    return CorporaManifest(
        root=root,
        repos=[CorpusRepo.model_validate(item) for item in DEFAULT_CORPORA],
    )


def load_manifest(path: Path, *, root: Path) -> CorporaManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    repos = data.get("repos") or DEFAULT_CORPORA
    return CorporaManifest(
        root=Path(data.get("root") or root),
        repos=[CorpusRepo.model_validate(r) for r in repos],
    )


def write_default_yaml(path: Path) -> Path:
    payload = {
        "root": "corpora",
        "repos": DEFAULT_CORPORA,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def sync_repo(repo: CorpusRepo, dest: Path, *, include_optional: bool = False) -> dict[str, Any]:
    if repo.optional and not include_optional:
        return {"id": repo.id, "status": "skipped_optional", "path": str(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (dest / ".git").exists():
        try:
            _run(["git", "fetch", "--depth", str(repo.depth), "origin", repo.ref], cwd=dest)
            _run(["git", "checkout", "FETCH_HEAD"], cwd=dest)
            return {"id": repo.id, "status": "updated", "path": str(dest)}
        except subprocess.CalledProcessError as exc:
            return {
                "id": repo.id,
                "status": "error",
                "path": str(dest),
                "error": (exc.stderr or str(exc))[:500],
            }
    try:
        cmd = [
            "git",
            "clone",
            "--depth",
            str(repo.depth),
            "--branch",
            repo.ref,
            repo.url,
            str(dest),
        ]
        if repo.sparse:
            cmd = [
                "git",
                "clone",
                "--depth",
                str(repo.depth),
                "--filter=blob:none",
                "--sparse",
                "--branch",
                repo.ref,
                repo.url,
                str(dest),
            ]
            _run(cmd)
            _run(["git", "sparse-checkout", "set", *repo.sparse], cwd=dest)
        else:
            _run(cmd)
        return {"id": repo.id, "status": "cloned", "path": str(dest)}
    except subprocess.CalledProcessError as exc:
        return {
            "id": repo.id,
            "status": "error",
            "path": str(dest),
            "error": (exc.stderr or str(exc))[:500],
        }


def sync_all(
    manifest: CorporaManifest,
    *,
    include_optional: bool = False,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for repo in manifest.repos:
        if only and repo.id not in only:
            continue
        results.append(
            sync_repo(
                repo,
                manifest.path_for(repo.id),
                include_optional=include_optional,
            )
        )
    status_path = manifest.root / "sync_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
