def first_n_words(node,n):
    # return first n words from a list which is generated from a string called node with space splitted
    node_ls = []+str(node).split(' ')
    return ' '.join([i for i in node_ls if i][:n])