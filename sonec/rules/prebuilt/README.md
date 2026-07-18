# Prebuilt rules

Authoritative operating rules shipped with sonec.

Loaded by `sonec.rules.RulesEngine` as `prebuilt/<id>`.

| Kind | Behavior |
| --- | --- |
| Always-on | Injected into every harness run |
| Conditional | Activated when the goal matches rule tags |

List with `rules_list`; load full bodies with `rules_load`.

This directory is part of the product, not IDE configuration.
