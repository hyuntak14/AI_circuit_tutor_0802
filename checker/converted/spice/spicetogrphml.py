import networkx as nx
import matplotlib.pyplot as plt
import os


def parse_spice(netlist_path):
    """Parse a SPICE netlist and return a NetworkX graph."""
    G = nx.Graph()
    with open(netlist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.lower().startswith((
                    '.include', '.subckt', '.ends', '.model', '.option')):
                continue
            tokens = line.split()
            comp_id = tokens[0]
            nets = tokens[1:-1]
            G.add_node(comp_id, type='component')
            for net in nets:
                if not G.has_node(net):
                    G.add_node(net, type='net')
                G.add_edge(comp_id, net)
    return G


def convert_all_spice_in_folder(folder_path='.'):
    """Convert all .spice files in the given folder to GraphML and PNG images."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.spice'):
            base = os.path.splitext(filename)[0]
            spice_path = os.path.join(folder_path, filename)
            graphml_path = os.path.join(folder_path, f"{base}.graphml")
            image_path = os.path.join(folder_path, f"{base}.png")

            print(f"Processing {filename}...")
            G = parse_spice(spice_path)

            # Write GraphML file
            nx.write_graphml(G, graphml_path)
            print(f"  -> GraphML saved: {graphml_path}")

            # Visualize and save image
            pos = nx.spring_layout(G)
            comp_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'component']
            net_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'net']
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, nodelist=comp_nodes, node_shape='s', label='Components')
            nx.draw_networkx_nodes(G, pos, nodelist=net_nodes, node_shape='o', label='Nets')
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_labels(G, pos)
            plt.axis('off')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(image_path, format='png')
            plt.close()
            print(f"  -> Visualization saved: {image_path}\n")


if __name__ == '__main__':
    convert_all_spice_in_folder()
