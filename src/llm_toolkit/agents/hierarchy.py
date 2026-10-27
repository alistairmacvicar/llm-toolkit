"""Validation for a director tree whose leaves are character names."""


def validate_hierarchy(children: dict[str, list[str]], root: str, characters):
    if not isinstance(children, dict) or not children or root not in children:
        raise ValueError("The hierarchy needs a root director")
    if any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in children):
        raise ValueError("Director names must be non-empty and trimmed")
    cast = set(characters)
    if cast.intersection(children):
        raise ValueError("Directors and characters must have different names")
    known = cast | set(children)
    parents = {}
    for parent, members in children.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"{parent} needs at least one child")
        for member in members:
            if not isinstance(member, str) or member not in known:
                raise ValueError(f"Unknown member in {parent}: {member}")
            if member in parents:
                raise ValueError(f"{member} must have exactly one parent")
            parents[member] = parent
    if root in parents:
        raise ValueError("The root director cannot have a parent")
    visited = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in visited:
            raise ValueError("The director hierarchy contains a cycle")
        visited.add(name)
        pending.extend(children.get(name, []))
    if visited != known:
        raise ValueError("Every director and character must be reachable from the root; check for cycles or unassigned nodes")
