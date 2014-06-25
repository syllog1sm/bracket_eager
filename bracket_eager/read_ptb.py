import re

def get_brackets(sent_text):
    open_brackets = []
    brackets = []
    bracketsRE = re.compile(r'(\()([^\s\)\(]+)|([^\s\)\(]+)?(\))')
    word_i = 0
    # Remove outermost bracket
    for match in bracketsRE.finditer(sent_text[2:-1]):
        open_, label, text, close = match.groups()
        if open_:
            assert not close
            assert label
            open_brackets.append((label, word_i))
        else:
            assert close
            label, start = open_brackets.pop()
            if text:
                word_i += 1
            brackets.append((label, start, word_i))
    return brackets


def split_sentences(text):
    sentences = []
    current = []
    for line in text.strip().split('\n'):
        line = line.rstrip()
        if not line:
            continue
        # Detect the start of sentences by line starting with (
        # This is messy, but it keeps bracket parsing at the sentence level
        if line.startswith('(') and current:
            sentences.append('\n'.join(current))
            current = []
        current.append(line)
    if current:
        sentences.append('\n'.join(current))
    return sentences
