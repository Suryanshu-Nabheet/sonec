"""Real-world WorldBench — IDE/runtime/CLI-shaped SE tasks (not toy smoke).

Tasks mirror engineering patterns from VS Code extensions, Bun packages,
and coding-agent CLIs (Codex-style). Fixtures are self-contained mini repos
so grades stay deterministic offline; optional `sonec corpora sync` pulls
full open-source trees for harder live work.
"""

from __future__ import annotations

import json
from pathlib import Path

from sonec.eval.harness import EvalCheck, EvalTask


def build_worldbench_tasks() -> list[EvalTask]:
    tasks: list[EvalTask] = []

    # --- VS Code extension patterns ---
    for i in range(1, 6):
        pkg = {
            "name": f"sonec-vscode-ext-{i}",
            "displayName": f"Sonec Ext {i}",
            "version": "0.0.1",
            "engines": {"vscode": "^1.85.0"},
            "activationEvents": ["onCommand:sonec.hello"],
            "contributes": {
                "commands": [{"command": "sonec.hello", "title": "Sonec: Hello"}]
            },
            "main": "./out/extension.js",
        }
        tasks.append(
            EvalTask(
                id=f"vscode-ext-activate-{i:02d}",
                name=f"VS Code extension activation {i}",
                prompt=(
                    f"This is a VS Code extension workspace. Fix package.json so "
                    f"activationEvents includes onCommand:sonec.hello{i} and add that "
                    f"command to contributes.commands with title 'Sonec: Hello {i}'."
                ),
                difficulty="medium",
                tags=["vscode", "extension", "typescript", "ide"],
                seed_files={
                    "package.json": json.dumps(pkg, indent=2) + "\n",
                    "src/extension.ts": (
                        "import * as vscode from 'vscode';\n\n"
                        "export function activate(context: vscode.ExtensionContext) {\n"
                        "  const disposable = vscode.commands.registerCommand('sonec.hello', () => {\n"
                        "    vscode.window.showInformationMessage('Hello');\n"
                        "  });\n"
                        "  context.subscriptions.push(disposable);\n"
                        "}\n"
                    ),
                    "tsconfig.json": json.dumps(
                        {"compilerOptions": {"module": "commonjs", "target": "ES2020", "outDir": "out"}},
                        indent=2,
                    )
                    + "\n",
                },
                checks=[
                    EvalCheck(
                        kind="file_contains",
                        path="package.json",
                        contains=f"onCommand:sonec.hello{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path="package.json",
                        contains=f"Sonec: Hello {i}",
                    ),
                ],
            )
        )

    for i in range(1, 5):
        tasks.append(
            EvalTask(
                id=f"vscode-ext-command-{i:02d}",
                name=f"Register VS Code command handler {i}",
                prompt=(
                    f"In src/extension.ts register command sonec.run{i} that shows "
                    f"message 'run-{i}'. Keep activate() export."
                ),
                difficulty="hard",
                tags=["vscode", "typescript", "ide"],
                seed_files={
                    "package.json": json.dumps(
                        {
                            "name": f"ext-{i}",
                            "engines": {"vscode": "^1.85.0"},
                            "activationEvents": [f"onCommand:sonec.run{i}"],
                            "contributes": {
                                "commands": [
                                    {"command": f"sonec.run{i}", "title": f"Run {i}"}
                                ]
                            },
                            "main": "./out/extension.js",
                        },
                        indent=2,
                    )
                    + "\n",
                    "src/extension.ts": (
                        "import * as vscode from 'vscode';\n\n"
                        "export function activate(context: vscode.ExtensionContext) {\n"
                        "  // TODO: register sonec.run command\n"
                        "}\n"
                    ),
                },
                checks=[
                    EvalCheck(
                        kind="file_contains",
                        path="src/extension.ts",
                        contains=f"sonec.run{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path="src/extension.ts",
                        contains=f"run-{i}",
                    ),
                    EvalCheck(kind="file_contains", path="src/extension.ts", contains="activate"),
                ],
            )
        )

    # --- Bun / package runtime patterns ---
    for i in range(1, 6):
        tasks.append(
            EvalTask(
                id=f"bun-script-{i:02d}",
                name=f"Bun package script {i}",
                prompt=(
                    f"Add a package.json script 'check{i}' that runs "
                    f"'bun test tests/check{i}.test.ts'. Create that test file "
                    f"asserting expect(1+1).toBe(2)."
                ),
                difficulty="medium",
                tags=["bun", "javascript", "tests", "runtime"],
                seed_files={
                    "package.json": json.dumps(
                        {"name": f"bun-app-{i}", "type": "module", "scripts": {"dev": "bun run src/index.ts"}},
                        indent=2,
                    )
                    + "\n",
                    "src/index.ts": f" console.log('bun app {i}');\n",
                },
                checks=[
                    EvalCheck(
                        kind="file_contains",
                        path="package.json",
                        contains=f"check{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path="package.json",
                        contains=f"tests/check{i}.test.ts",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path=f"tests/check{i}.test.ts",
                        contains="toBe(2)",
                    ),
                ],
            )
        )

    for i in range(1, 4):
        tasks.append(
            EvalTask(
                id=f"bun-fetch-client-{i:02d}",
                name=f"Bun HTTP client {i}",
                prompt=(
                    f"Create src/client{i}.ts exporting async function fetchUser(id: string) "
                    f"that calls fetch(`/api/users/${{id}}`) and returns response.json()."
                ),
                difficulty="medium",
                tags=["bun", "typescript", "api"],
                seed_files={
                    "package.json": '{"name":"client","type":"module"}\n',
                    "src/index.ts": "export {};\n",
                },
                checks=[
                    EvalCheck(kind="file_exists", path=f"src/client{i}.ts"),
                    EvalCheck(
                        kind="file_contains",
                        path=f"src/client{i}.ts",
                        contains="fetchUser",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path=f"src/client{i}.ts",
                        contains="/api/users/",
                    ),
                ],
            )
        )

    # --- Codex / agent-CLI patterns ---
    for i in range(1, 5):
        tasks.append(
            EvalTask(
                id=f"codex-tool-schema-{i:02d}",
                name=f"Agent tool schema {i}",
                prompt=(
                    f"Create tools/fs_read_{i}.json describing an agent tool with "
                    f'name "fs_read_{i}", description "Read a file", and parameters '
                    f'object requiring "path" string.'
                ),
                difficulty="easy",
                tags=["codex", "agent", "cli", "tools"],
                seed_files={
                    "README.md": "# agent workspace\n",
                    "AGENTS.md": "You are a coding agent. Use tools. Verify before done.\n",
                },
                checks=[
                    EvalCheck(kind="file_exists", path=f"tools/fs_read_{i}.json"),
                    EvalCheck(
                        kind="file_contains",
                        path=f"tools/fs_read_{i}.json",
                        contains=f"fs_read_{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path=f"tools/fs_read_{i}.json",
                        contains="path",
                    ),
                ],
            )
        )

    for i in range(1, 4):
        tasks.append(
            EvalTask(
                id=f"codex-verify-gate-{i:02d}",
                name=f"Agent verify-before-done {i}",
                prompt=(
                    f"Create scripts/verify_{i}.sh that exits 0 and echoes OK_{i}. "
                    f"Create VERIFY.md stating the verification command is "
                    f"./scripts/verify_{i}.sh"
                ),
                difficulty="medium",
                tags=["codex", "agent", "verify", "cli"],
                seed_files={"AGENTS.md": "Always verify with a script before finishing.\n"},
                checks=[
                    EvalCheck(kind="file_exists", path=f"scripts/verify_{i}.sh"),
                    EvalCheck(
                        kind="file_contains",
                        path=f"scripts/verify_{i}.sh",
                        contains=f"OK_{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path="VERIFY.md",
                        contains=f"./scripts/verify_{i}.sh",
                    ),
                ],
            )
        )

    # --- Multi-file refactor / migration (IDE-scale) ---
    for i in range(1, 4):
        tasks.append(
            EvalTask(
                id=f"ide-rename-symbol-{i:02d}",
                name=f"Cross-file rename {i}",
                prompt=(
                    f"Rename exported function oldName{i} to newName{i} in src/lib.ts "
                    f"and update the import/usage in src/app.ts."
                ),
                difficulty="hard",
                tags=["ide", "refactor", "typescript"],
                seed_files={
                    "src/lib.ts": f"export function oldName{i}(): number {{\n  return {i};\n}}\n",
                    "src/app.ts": (
                        f"import {{ oldName{i} }} from './lib';\n"
                        f"export const value = oldName{i}();\n"
                    ),
                },
                checks=[
                    EvalCheck(
                        kind="file_contains",
                        path="src/lib.ts",
                        contains=f"newName{i}",
                    ),
                    EvalCheck(
                        kind="file_not_contains",
                        path="src/lib.ts",
                        contains=f"oldName{i}",
                    ),
                    EvalCheck(
                        kind="file_contains",
                        path="src/app.ts",
                        contains=f"newName{i}",
                    ),
                    EvalCheck(
                        kind="file_not_contains",
                        path="src/app.ts",
                        contains=f"oldName{i}",
                    ),
                ],
            )
        )

    for i in range(1, 4):
        tasks.append(
            EvalTask(
                id=f"restraint-review-{i:02d}",
                name=f"Review-only no edit {i}",
                prompt=(
                    f"Question only ({i}): In one sentence, why should an IDE agent "
                    f"prefer apply_patch over rewriting whole files? Do not create or edit files."
                ),
                difficulty="easy",
                tags=["review", "restraint", "ide"],
                checks=[],
            )
        )

    assert len(tasks) >= 30
    return tasks


def write_worldbench(path: Path) -> Path:
    tasks = build_worldbench_tasks()
    payload = {
        "name": "worldbench-v1",
        "version": "1.0.0",
        "sealed": True,
        "description": (
            "WorldBench v1 — real-world IDE/runtime/CLI agent tasks shaped like "
            "VS Code extensions, Bun packages, and Codex-style agent CLIs. "
            "Do not train on these task ids. Use for live open-weight evaluation."
        ),
        "task_count": len(tasks),
        "tasks": [t.model_dump() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = Path("examples/benchmarks/worldbench_v1.json")
    write_worldbench(out)
    print(f"Wrote {out} with {len(build_worldbench_tasks())} tasks")
