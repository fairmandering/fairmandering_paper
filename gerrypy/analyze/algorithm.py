import numpy as np


def analyze_shc(leaf_nodes, interior_nodes):
    print('%d total leaf nodes' % len(leaf_nodes))
    print('%d total interior nodes' % len(interior_nodes))
    n_failures = [i.n_sample_failures for i in interior_nodes]
    n_disconnected = [i.n_disconnected_samples for i in interior_nodes]
    print('%d total sample failures' % sum(n_failures))
    print('%d total disconnected samples' % sum(n_disconnected))


def number_of_districtings(leaf_nodes, interior_nodes):
    nodes = leaf_nodes + interior_nodes
    id_to_node = {node.id: node for node in nodes}
    root = nodes[np.argmax(np.array([n.n_districts for n in nodes]))]
    
    def recursive_compute(current_node, all_nodes):
        if not current_node.children_ids:
            return 1
        
        total_districtings = 0
        for sample in current_node.children_ids:
            sample_districtings = 1
            for child_id in sample:
                child_node = id_to_node[child_id]
                sample_districtings *= recursive_compute(child_node, all_nodes)

            total_districtings += sample_districtings
        return total_districtings
    
    return recursive_compute(root, nodes)


if __name__ == '__main__':
    import pickle
    leaves, interior = pickle.load(open('test_nodes.p', 'rb'))
    print(number_of_districtings(leaves, interior))