from collections import defaultdict


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
        # left-hand side, right-hand side, e.g. S --> NP VP
        lhs = top.label
        rhs = _get_rhs(top.children)
        rules.setdefault('TOP', {}).setdefault((lhs,), 0)
        rules['TOP'][(lhs,)] += 1
        rules.setdefault(lhs, {}).setdefault(rhs, 0)
        rules[lhs][rhs] += 1
        for node in top.iter_nodes():
            if not node.is_leaf:
                lhs = node.label
                rhs = _get_rhs(node.children)
                rules.setdefault(lhs, {}).setdefault(rhs, 0)
                rules[lhs][rhs] += 1
    return dict(rules)


def _get_rhs(nodes):
    return tuple(n.label for n in nodes)
