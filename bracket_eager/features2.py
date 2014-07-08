def extract_features(stack, q):
    n0 = _from_queue(q, 0)
    n1 = _from_queue(q, 1)
    n2 = _from_queue(q, 2)
    s0, s0L, s0lhs = _from_stack(stack, -1)
    s1, s1L, s1lhs = _from_stack(stack, -2)
    s2, s2L, s2lhs = _from_stack(stack, -3)
    ps0 = _prod(stack[-1]) if stack else "_"

    feats = ['Bias']
    f = feats.append

    f("a" + ps0 + "_" + n0.label + "_" + n1.label + "_" + s1L)
    #f("b" + ps0 + "_" + n0.label + "_" + n1.label)
    #f("c" + ps0 + "_" + n0.label + "_" + s1L)

    f("seq" + "_".join([s2L, s1L, s0L, n0.label]))
    
    add_uni(f, 'n0', n0, '', [])
    add_uni(f, 'n1', n1, '', [])
    add_uni(f, 'n2', n2, '', [])
    add_uni(f, 's0', s0, s0L, s0lhs)
    add_uni(f, 's1', s1, s1L, s1lhs) # no need for LHS
    add_uni(f, 's2', s2, s2L, []) # no need for LHS

    add_bi(f, 's0/n0', s0, n0, s0L, s0lhs, '', [])
    add_bi(f, 's1/s0', s1, s0, s1L, s1lhs, s0L, s0lhs)
    add_bi(f, 's2/s0', s2, s0, s2L, s2lhs, s0L, s0lhs)
    
    #add_tri(f, 's1/s0/n0', s2, s1, s0, s2lhs, s2L, s1lhs, s1L, s0lhs, s0L)
    #add_tri(f, 's1/s0/n0', s1, s0, n0, s1lhs, s1L, s0lhs, s0L, [], '')
    return feats


def add_uni(f, pre, tok, label, lhs):
    if not tok:
        return None
    f(pre + 'w=' + tok.lex)
    f(pre + 'p=' + tok.label)
    if label:
        f(pre + 'L=' + label)
    if lhs:
        f(pre + 'exp=' + ','.join(lhs))
        f(pre + 'exp_len=%d' % len(lhs))
        f(pre + 'prod=%s->%s..%s' % (label, lhs[0], lhs[-1]))


def add_bi(f, pre, t1, t2, lab1, lhs1, lab2, lhs2):
    if not t1 or not t2:
        return None
    f(pre + 'wp=' + t1.lex + '_' + t2.label)
    f(pre + 'pw=' + t1.label + '_' + t2.lex)
    f(pre + 'pp=' + t1.label + '_' + t2.label)
    f(pre + 'LL=' + lab1 + '_' + lab2)
    #f[pre + 'w1LL=' + t1.lex + '_' + lab1 + '_' + lab2)
    #f[pre + 'w2LL=' + t2.lex + '_' + lab1 + '_' + lab2)
    f(pre + 'p1LL=' + t1.label + '_' + lab1 + '_' + lab2)
    f(pre + 'p2LL=' + t2.label + '_' + lab1 + '_' + lab2)
    f(pre + 'ppLL=' + t1.label + '_' + t2.label + '_' + lab1 + '_' + lab2)

import tree
_end = tree.Word(-1,'-EOS-','-EOS-')

def _prod(tok):
   lhs = tok.children
   if lhs:
      return "%s->%s_%s" % (tok.label, lhs[0].label, lhs[-1].label)
   else:
      return "%s" % tok.label

def _from_queue(tokens, i):
    if i < 0:
        return None
    if i >= len(tokens):
        #return None
        return _end
    else:
        return tokens[i]


def _from_stack(nodes, i):
    if -i > len(nodes):
        #return '', [], None
        return _end, '-EOS-', []
    else:
        return nodes[i].head, nodes[i].label, [n.label for n in nodes[i].children]
