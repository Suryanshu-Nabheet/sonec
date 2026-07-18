# Prebuilt rules

These are **first-class SONEC operating rules** shipped with the package.

They originate from battle-tested operator standards (including the author's
Cursor rule set) and are loaded by `sonec.rules.RulesEngine` as `prebuilt/<id>`.

| Kind | Behavior |
| --- | --- |
| Always-on | Injected into every harness run (constitution, guidelines, git safety) |
| Conditional | Activated when the goal matches rule tags (design, security, animation, …) |

Agents can list them with `rules_list` and load full (untruncated) bodies with `rules_load`.

Do not treat this folder as IDE config — it is part of the SONEC product.
