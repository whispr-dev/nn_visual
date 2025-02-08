import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

class AnalogNeuralNetwork:
    def __init__(self):
        # Weight matrices from the document
        self.weights = {
            'layer1': np.array([
                [0.5, 1.0, 0, 0],      # Neuron A weights
                [0, 0, -1.5, 0.75]     # Neuron B weights
            ]),
            'layer2': np.array([[1.0, -0.5]])  # Output neuron weights
        }
        
        self.thresholds = {
            'layer1': np.array([15, 50]),
            'layer2': np.array([-5])
        }
        
        # Initialize values
        self.values = {
            'inputs': np.array([20, 10, -30, 40]),
            'layer1': np.zeros(2),
            'output': np.zeros(1)
        }

    def analog_activation(self, x, theta=0):
        """Analog activation function (-100 to +100)"""
        shifted = x - theta
        return np.clip(shifted, -100, 100)

    def forward(self, time):
        """Forward propagation with time-varying inputs"""
        # Generate oscillating inputs
        self.values['inputs'] = np.array([
            20 * np.sin(time/10 + i) + val 
            for i, val in enumerate(self.values['inputs'])
        ])
        
        # Layer 1
        layer1_sums = np.dot(self.weights['layer1'], self.values['inputs'])
        self.values['layer1'] = self.analog_activation(
            layer1_sums, 
            self.thresholds['layer1']
        )
        
        # Output layer
        output_sum = np.dot(self.weights['layer2'], self.values['layer1'])
        self.values['output'] = self.analog_activation(
            output_sum, 
            self.thresholds['layer2']
        )
        
        return self.values

class NeuralNetworkVisualizer:
    def __init__(self):
        self.network = AnalogNeuralNetwork()
        
        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.axis('off')
        
        # Create network graph
        self.G = nx.DiGraph()
        
        # Add nodes
        self.input_pos = {f'i{i}': (0.1, 0.2 + i*0.15) for i in range(4)}
        self.hidden_pos = {f'h{i}': (0.5, 0.3 + i*0.3) for i in range(2)}
        self.output_pos = {'o0': (0.9, 0.5)}
        
        self.pos = {**self.input_pos, **self.hidden_pos, **self.output_pos}
        self.G.add_nodes_from(self.pos.keys())
        
        # Add edges
        self.edges = []
        for i in range(4):
            for h in range(2):
                self.G.add_edge(f'i{i}', f'h{h}')
        for h in range(2):
            self.G.add_edge(f'h{h}', 'o0')
        
        # Initialize node colors
        self.nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='lightblue',
            node_size=1000
        )
        
        # Draw edges with varying widths and colors
        self.edges = nx.draw_networkx_edges(
            self.G, self.pos,
            edge_color='gray',
            width=1,
            arrows=True,
            arrowsize=20
        )
        
        # Add value labels
        self.value_labels = {}
        for node in self.G.nodes():
            self.value_labels[node] = self.ax.text(
                self.pos[node][0], self.pos[node][1],
                '0',
                horizontalalignment='center',
                verticalalignment='center'
            )
        
        self.time = 0

    def update(self, frame):
        # Update network values
        values = self.network.forward(self.time)
        self.time += 0.1
        
        # Update node colors and labels
        colors = []
        for node in self.G.nodes():
            if node.startswith('i'):
                val = values['inputs'][int(node[1])]
            elif node.startswith('h'):
                val = values['layer1'][int(node[1])]
            else:
                val = values['output'][0]
            
            # Update value labels
            self.value_labels[node].set_text(f'{val:.1f}')
            
            # Calculate color based on value
            normalized_val = (val + 100) / 200  # Convert -100 to +100 to 0 to 1
            colors.append(plt.cm.Blues(normalized_val))
        
        self.nodes.set_color(colors)
        
        # Update edge colors and widths based on weights
        edge_colors = []
        edge_widths = []
        for edge in self.G.edges():
            if edge[0].startswith('i') and edge[1].startswith('h'):
                weight = self.network.weights['layer1'][
                    int(edge[1][1]), 
                    int(edge[0][1])
                ]
            else:  # hidden to output
                weight = self.network.weights['layer2'][
                    0, 
                    int(edge[0][1])
                ]
            
            edge_colors.append('blue' if weight > 0 else 'red')
            edge_widths.append(abs(weight) * 2)
        
        self.edges.set_color(edge_colors)
        self.edges.set_linewidths(edge_widths)
        
        return self.nodes, self.edges, *self.value_labels.values()

    def animate(self):
        ani = FuncAnimation(
            self.fig, 
            self.update, 
            frames=None,
            interval=50,
            blit=True
        )
        plt.show()

if __name__ == "__main__":
    visualizer = NeuralNetworkVisualizer()
    visualizer.animate()