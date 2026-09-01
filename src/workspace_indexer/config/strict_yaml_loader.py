"""A YAML loader that refuses to silently discard a repeated key."""

from __future__ import annotations

from typing import Any

import yaml


class StrictYamlLoader(yaml.SafeLoader):
    """`yaml.safe_load` keeps the *last* of two identical keys and says nothing.

    Which means a block can be edited, saved, and have no effect whatsoever --
    including its comments, which is how a carefully explained `file:` section
    came to be dead text in a real config. The failure has no symptom: the file
    parses, the program runs, and the setting the author is reading is not the
    setting in force.

    Raising costs one obvious error at startup and removes an entire class of
    silent misconfiguration.
    """


def _no_duplicate_keys(
    loader: StrictYamlLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    # Only scalar keys are checked, and they are read straight off the node
    # rather than through `construct_object`, which PyYAML leaves untyped.
    # Every key a configuration file has is a scalar; a list or mapping used as
    # a key would go unchecked, which costs a warning nobody was going to need
    # and keeps this honestly typed.
    seen: set[str] = set()
    for key_node, _ in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            continue
        key = key_node.value
        if key in seen:
            raise yaml.MarkedYAMLError(
                context="while reading a mapping",
                problem=(
                    f"duplicate key {key!r}. YAML keeps only the last one, so the "
                    "earlier block -- and any comments in it -- would be silently "
                    "ignored. Merge them into one."
                ),
                problem_mark=key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep)


StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)
