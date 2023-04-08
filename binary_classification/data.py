import collections

def tokenize(s):
    return [t.rstrip('.') for t in s.split(' ')]


def vectorize(tokens):
    return collections.Counter(tokens)


def readiter(fi):
    for line in fi:
        fields = line.strip('\n').split('\t')
        x = vectorize(tokenize(fields[1]))
        y = fields[0]
        yield x, y


def get_data(file_name):
    with open(file_name, encoding="utf8") as fi:
        D = [d for d in readiter(fi)]
        return D