def extract_features(stack, q):
    n0 = _from_queue(q, 0)
    n1 = _from_queue(q, 1)
    n2 = _from_queue(q, 2)
    s0, s0L, s0lhs = _from_stack(stack, -1)
    s1, s1L, s1lhs = _from_stack(stack, -2)
    s2, s2L, s2lhs = _from_stack(stack, -3)

    f = {'Bias': 1}
    
    add_uni(f, 'n0', n0, '', [])
    add_uni(f, 'n1', n1, '', [])
    add_uni(f, 'n2', n2, '', [])
    add_uni(f, 's0', s0, s0L, s0lhs)
    add_uni(f, 's1', s1, s1L, s1lhs)
    add_uni(f, 's2', s2, s2L, s2lhs)

    add_bi(f, 's0/n0', s0, n0, s0L, s0lhs, '', [])
    add_bi(f, 's1/s0', s1, s0, s1L, s1lhs, s0L, s0lhs)
    add_bi(f, 's2/s0', s2, s0, s2L, s2lhs, s0L, s0lhs)
    
    #add_tri(f, 's1/s0/n0', s2, s1, s0, s2lhs, s2L, s1lhs, s1L, s0lhs, s0L)
    #add_tri(f, 's1/s0/n0', s1, s0, n0, s1lhs, s1L, s0lhs, s0L, [], '')
    return f


def add_uni(f, pre, tok, label, lhs):
    if not tok:
        return None
    f[pre + 'w=' + tok.lex] = 1
    f[pre + 'p=' + tok.label] = 1
    if label:
        f[pre + 'L=' + label] = 1
    if lhs:
        f[pre + 'exp=' + ','.join(lhs)] = 1
        f[pre + 'exp_len=%d' % len(lhs)] = 1
        f[pre + 'prod=%s --> %s .. %s' % (label, lhs[0], lhs[-1])] = 1


def add_bi(f, pre, t1, t2, lab1, lhs1, lab2, lhs2):
    if not t1 or not t2:
        return None
    f[pre + 'wp=' + t1.lex + '_' + t2.label] = 1
    f[pre + 'pw=' + t1.label + '_' + t2.lex] = 1
    f[pre + 'pp=' + t1.label + '_' + t2.label] = 1
    f[pre + 'LL=' + lab1 + '_' + lab2] = 1
    f[pre + 'wLL=' + t1.lex + '_' + t1.label + t2.label] = 1
    f[pre + 'wLL=' + t2.lex + '_' + t1.label + t2.label] = 1
    f[pre + 'pLL=' + t1.label + '_' + t1.label + t2.label] = 1
    f[pre + 'pLL=' + t2.label + '_' + t1.label + t2.label] = 1

def _from_queue(tokens, i):
    if i < 0:
        return None
    if i >= len(tokens):
        return None
    else:
        return tokens[i]


def _from_stack(nodes, i):
    if -i > len(nodes):
        return '', [], None
    else:
        return nodes[i].head, nodes[i].label, [n.label for n in nodes[i].children]
