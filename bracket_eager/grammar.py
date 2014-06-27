def get_valid_stacks(rules, max_len=100):
    """Get the set of stacks that are valid prefixes of TOP"""
    assert max_len
    stacks = set()
    queue = [('TOP',)]
    for stack in queue:
        prefix = stack[:-1]
        if prefix and len(prefix) < max_len and prefix not in stacks:
            stacks.add(prefix)
            queue.append(prefix)
        for rule in rules.get(stack[-1], []):
            expansion = prefix + rule
            if len(expansion) < max_len and expansion not in stacks:
                stacks.add(expansion)
                queue.append(expansion)
    return stacks


def rules_from_trees(trees):
    rules = {}
    for top in trees:
        lhs, rhs = top.production
        rules.setdefault(lhs, set()).add(rhs)
        for node in top.iter_nodes():
            if not node.is_leaf:
                lhs, rhs = node.production
                rules.setdefault(lhs, set()).add(rhs)
    return rules
