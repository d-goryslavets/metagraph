#!/usr/bin/env python

import sys
import numpy as np
from graphviz import Digraph


"""
Script for plotting internal Multi-BRWT tree
"""


def human_readable(value):
    value = int(value)
    if value >= 10**15:
        return '{:.1f}P'.format(value / 10**15)
    if value >= 10**12:
        return '{:.1f}T'.format(value / 10**12)
    if value >= 10**9:
        return '{:.1f}G'.format(value / 10**9)
    if value >= 10**6:
        return '{:.1f}M'.format(value / 10**6)
    if value >= 10**3:
        return '{:.1f}K'.format(value / 10**3)
    return '{}'.format(value)


def render_tree(text):
    assert(text[14] == '==================== Multi-BRWT TREE ===================\n')
    assert(text[-1] == '========================================================\n')

    print("Rendering Multi-BRWT structure for:", text[4].strip().split()[-1])
    d = Digraph(name=text[4].strip().split()[-1], graph_attr={'rankdir': 'LR'})

    text = text[15:-1]

    max_size = max([int(line.split(',')[1]) for line in text])

    for line in text:
        line = line.strip()
        tokens = line.split(',')
        node, size, num_set_bits = tokens[0:3]
        children = tokens[4:]
        d.node(node, label='{}, {:.1f}%'.format(human_readable(size), float(num_set_bits) / int(size) * 100),
                     style='filled',
                     fillcolor='0.000 0.000 {:1.3f}'.format(1 - float(num_set_bits) / 1.8 / int(size)),
                     penwidth=str(15 * np.sqrt(1 + int(size)) / np.sqrt(1 + max_size)))
        for child in tokens[3:]:
            d.edge(node, child)

    d.render()


if __name__ == '__main__':
    render_tree(sys.stdin.readlines())
