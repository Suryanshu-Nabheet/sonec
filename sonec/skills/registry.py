"""Skills registry — progressive disclosure expertise packs for the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


@dataclass
class Skill:
    id: str
    name: str
    description: str
    body: str
    tags: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    always: bool = False
    priority: int = 100
    path: str = ""

    def score(self, goal: str) -> float:
        if self.always:
            return 1_000.0
        g = goal.lower()
        score = 0.0
        for trigger in self.triggers:
            if trigger.lower() in g:
                score += 3.0
        for tag in self.tags:
            if tag.lower() in g:
                score += 1.5
        # token overlap with name/description
        tokens = set(re.findall(r"[a-z0-9]+", g))
        for token in re.findall(r"[a-z0-9]+", f"{self.name} {self.description}".lower()):
            if token in tokens and len(token) > 3:
                score += 0.25
        return score


@dataclass
class SkillActivation:
    skill: Skill
    score: float


class SkillsRegistry:
    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self.skills: dict[str, Skill] = {}
        self._load_packaged()
        for directory in extra_dirs or []:
            self._load_directory(directory)

    def _load_packaged(self) -> None:
        try:
            root = resources.files("sonec.skills")
        except (TypeError, ModuleNotFoundError, AttributeError):
            root = Path(__file__).resolve().parent
        self._load_from_traversable(root)

    def _load_from_traversable(self, root: object) -> None:
        path = Path(str(root))
        if path.is_dir():
            for skill_md in path.rglob("SKILL.md"):
                self._register_file(skill_md)
            return
        # importlib.resources Traversable
        try:
            for entry in root.iterdir():  # type: ignore[attr-defined]
                if entry.is_dir():
                    skill_file = entry.joinpath("SKILL.md")
                    if skill_file.is_file():
                        text = skill_file.read_text(encoding="utf-8")
                        skill = parse_skill_markdown(entry.name, text, path=str(skill_file))
                        self.skills[skill.id] = skill
        except Exception:
            # Fallback to filesystem relative to this module
            fallback = Path(__file__).resolve().parent
            for skill_md in fallback.rglob("SKILL.md"):
                self._register_file(skill_md)

    def _load_directory(self, directory: Path) -> None:
        if not directory.exists():
            return
        for skill_md in directory.rglob("SKILL.md"):
            self._register_file(skill_md)

    def _register_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        skill_id = path.parent.name
        skill = parse_skill_markdown(skill_id, text, path=str(path))
        self.skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        if skill_id not in self.skills:
            raise KeyError(skill_id)
        return self.skills[skill_id]

    def activate(self, goal: str, *, limit: int = 4) -> list[SkillActivation]:
        scored = [
            SkillActivation(skill=s, score=s.score(goal))
            for s in self.skills.values()
        ]
        scored.sort(key=lambda a: (-a.score, a.skill.priority, a.skill.id))
        chosen: dict[str, SkillActivation] = {}
        non_always = 0
        for item in scored:
            if item.skill.always or item.score > 0:
                if item.skill.id not in chosen and not item.skill.always:
                    if non_always >= limit:
                        continue
                    non_always += 1
                chosen[item.skill.id] = item
        for skill in self.skills.values():
            if skill.always and skill.id not in chosen:
                chosen[skill.id] = SkillActivation(skill=skill, score=skill.score(goal))
        return sorted(chosen.values(), key=lambda a: (-a.score, a.skill.id))

    def render(self, goal: str, *, limit: int = 4, max_chars: int = 28_000) -> str:
        activations = self.activate(goal, limit=limit)
        if not activations:
            return "# Skills\n\nNo specialized skills activated."
        parts = ["# Activated skills", ""]
        total = 0
        for act in activations:
            chunk = (
                f"## {act.skill.name} (`{act.skill.id}`, score={act.score:.1f})\n\n"
                f"{act.skill.body.strip()}\n"
            )
            if total + len(chunk) > max_chars:
                parts.append(f"[Skill omitted due to budget: {act.skill.id}]")
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n".join(parts)

    def catalog(self) -> list[dict[str, object]]:
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "tags": list(s.tags),
                "triggers": list(s.triggers),
                "always": s.always,
            }
            for s in sorted(self.skills.values(), key=lambda x: x.id)
        ]


def parse_skill_markdown(skill_id: str, text: str, *, path: str = "") -> Skill:
    meta: dict[str, object] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end].strip()
            meta = yaml.safe_load(front) or {}
            body = text[end + 4 :].lstrip("\n")
    name = str(meta.get("name") or skill_id.replace("-", " ").title())
    description = str(meta.get("description") or "")
    tags = tuple(str(t) for t in (meta.get("tags") or []))
    triggers = tuple(str(t) for t in (meta.get("triggers") or []))
    always = bool(meta.get("always", False))
    priority = int(meta.get("priority", 100))
    return Skill(
        id=str(meta.get("id") or skill_id),
        name=name,
        description=description,
        body=body,
        tags=tags,
        triggers=triggers,
        always=always,
        priority=priority,
        path=path,
    )
