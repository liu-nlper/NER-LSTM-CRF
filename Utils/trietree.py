#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Trie tree
"""
from collections import defaultdict
from time import time


class Node(object):

    def __init__(self, value, children_value=None):
        self._value = value
        self._children = defaultdict()
        self._end_flag = ''

    @property
    def children(self):
        return self._children

    @property
    def value(self):
        return self._value

    @property
    def end_flag(self):
        return self._end_flag

    def set_end_flag(self, value):
        self._end_flag = value
		
    def set_children(self, children):
        self._children = children

    def __str__(self):
        return self._value


class TrieTree(object):

    def __init__(self, entity_dict=None):
        """
        Args:
            entity_dict: dict
        """
        self._root = Node('root')
        if entity_dict:
            self.update_tree_batch(entity_dict)

    def update_tree_batch(self, entity_dict):
        """
        Args:
            entity_dict: dict
        """
        for entity in entity_dict:
            entity_type = entity_dict[entity]
            self.update_tree(entity, entity_type)

    def update_tree(self, entity, entity_type):
        """
        Args:
            entity: str
            entity_type: str
        """
        name_len = len(entity)
        node_pre = self.root
        for i in range(name_len):
            c = entity[i]
            if c not in node_pre.children:
                node = Node(c)
                node_pre.children[c] = node
                node_pre = node
                if i == name_len-1:
                    node.set_end_flag(entity_type)
            else:
                if i == name_len-1:
                    node_pre.children[c].set_end_flag(entity_type)
                node_pre = node_pre.children[c]

    @property
    def root(self):
        return self._root

    def show(self, node, level=0):
        """
        输出tree
        """
        if not node:
            return
        if level == 0:
            print(node)
        else:
            print('└%s%s %s' % ('─' * level*2, node, node.end_flag))
        for node_value in node.children:
            end_flag = node.children[node_value].end_flag
            print('└%s%s %s' % ('─' * (level+1)*2, node_value, end_flag))
            if not node.children[node_value]:
                continue
            for child in node.children[node_value].children:
                self.show(node.children[node_value].children[child], level+2)


def match_sentence(sentence, root, start):
    """
    Args:
        sentence: str
        root: Node, root node
        start: int, 句子的起始位置
    Return:
        end, entity_len, entity_type
    """
    current_node = root
    index, entity_type = -1, ''
    for i in range(start, len(sentence)):
        if sentence[i] not in current_node.children:
            if index == -1:
                return start + 1, 0, ''
            else:
                return index+1, index+1-start, entity_type
        current_node = current_node.children[sentence[i]]
        if current_node.end_flag:
            index = i
            entity_type = current_node.end_flag
    if index != -1:
        return index+1, index+1-start, entity_type
    return start+1, 0, ''


def demo():
    # 构建树
    print('Building tree...', end='')
    entity_dict = {'苏州': 'GPE', '苏大': 'ORG', '苏州大学': 'ORG', '小明': 'PER', '江苏': 'GPE',
                   '苏有朋': 'PER', '江苏大学': 'ORG', '中华人民共和国': 'GPE'}
    tree = TrieTree(entity_dict)
    print('done!')
    tree.show(tree.root)

    # 测试
    sentence = '我住在中华人民共和国江苏省苏州苏州大学，邻居是苏州大的小明。苏有朋'
    print('\nsentence:', sentence)
    end, sent_len = 0, len(sentence)
    print('\nresult:\nstart\tend\tentity\tentity_type')
    while end < sent_len:
        end, entity_len, entity_type = match_sentence(sentence, tree.root, end)
        if entity_type:
            print('%d\t%d\t%s\t%s' %
                  (end-entity_len, end-1, sentence[end-entity_len:end], entity_type))
    print('Done!')


def demo_2():
    # 构建树
    print('Building tree...', end='')
    entity_dict = {'苏州': 'GPE', '苏大': 'ORG', '苏州大学': 'ORG', '小明': 'PER', '江苏': 'GPE',
                   '苏有朋': 'PER', '江苏大学': 'ORG', '中华人民共和国': 'GPE'}
    tree = TrieTree(entity_dict)
    print('done!')
    # tree.show(tree.root)

    t0 = time()
    # 测试
    sentence_count = int(1e6)
    sentences = ['我住在中华人民共和国江苏省苏州苏州大学，邻居是苏州大的小明。苏有朋'] * sentence_count
    for sentence in sentences:
        end, sent_len = 0, len(sentence)
        while end < sent_len:
            end, entity_len, entity_type = match_sentence(sentence, tree.root, end)
    duration = time() - t0
    print('Done in %.1fs, %.1f sentences/s' % (duration, sentence_count / duration))


if __name__ == '__main__':
    demo()
