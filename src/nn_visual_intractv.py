import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

class AnalogNeuralNetwork:
    def __init__(self):
        # Previous network code remains the same
        self.weights = {
            'layer1': np.array([
                [0.5, 1.0, 0, 0],
                [0, 0, -1.5, 0.75]
            ]),
            'layer2': np.array([[1.0, -0.5]])
        }
        
        self.thresholds = {
            'layer1': np.array([15, 50]),
            'layer2': np.array([-5])
        }
        
        self.values = {
            'inputs': np.array([20, 10, -30, 40]),
            'layer1': np.zeros(2),
            'output': np.zeros(1)
        }
        
        self.injected = {
            'inputs': np.zeros(4),
            'layer1': np.zeros(2),
            'output': np.zeros(1)
        }
        
        self.decay_rate = 0.95

    def analog_activation(self, x, theta=0):
        shifted = x - theta
        return np.clip(shifted, -100, 100)

    def forward(self, time):
        # Previous forward propagation code remains the same
        for layer in self.injected:
            self.injected[layer] *= self.decay_rate
        
        self.values['inputs'] = np.array([
            20 * np.sin(time/10 + i) + val + inj
            for i, (val, inj) in enumerate(zip(
                self.values['inputs'], 
                self.injected['inputs']
            ))
        ])
        
        layer1_sums = np.dot(self.weights['layer1'], self.values['inputs'])
        self.values['layer1'] = self.analog_activation(
            layer1_sums + self.injected['layer1'], 
            self.thresholds['layer1']
        )
        
        output_sum = np.dot(self.weights['layer2'], self.values['layer1'])
        self.values['output'] = self.analog_activation(
            output_sum + self.injected['output'], 
            self.thresholds['layer2']
        )
        
        return self.values
    
    def inject_signal(self, layer, index, strength=50):
        self.injected[layer][index] = strength

class NeuralNetworkVisualizer:
    def __init__(self):
        self.network = AnalogNeuralNetwork()
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.axis('off')
        
        # Create network graph
        self.G = nx.DiGraph()
        
        # Add nodes with positions
        self.input_pos = {f'i{i}': (0.1, 0.2 + i*0.15) for i in range(4)}
        self.hidden_pos = {f'h{i}': (0.5, 0.3 + i*0.3) for i in range(2)}
        self.output_pos = {'o0': (0.9, 0.5)}
        
        self.pos = {**self.input_pos, **self.hidden_pos, **self.output_pos}
        self.G.add_nodes_from(self.pos.keys())
        
        # Add edges
        for i in range(4):
            for h in range(2):
                self.G.add_edge(f'i{i}', f'h{h}')
        for h in range(2):
            self.G.add_edge(f'h{h}', 'o0')
        
        # Initialize nodes
        self.nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='lightblue',
            node_size=1000
        )
        
        # Create edge collections
        edge_pos = []
        for (u, v) in self.G.edges():
            edge_pos.append([self.pos[u], self.pos[v]])
        self.edge_collection = LineCollection(
            edge_pos,
            colors=['gray'] * len(edge_pos),
            linewidths=1,
            zorder=1
        )
        self.ax.add_collection(self.edge_collection)
        
        # Add value labels
        self.value_labels = {}
        for node in self.G.nodes():
            self.value_labels[node] = self.ax.text(
                self.pos[node][0], self.pos[node][1],
                '0',
                horizontalalignment='center',
                verticalalignment='center'
            )
        
        # Initialize click feedback effects
        self.click_effects = []
        self.click_times = {}
        
        self.time = 0
        
        # Add click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add title with instructions
        self.ax.set_title('Click on neurons to inject signals!', pad=20)

    def add_click_effect(self, x, y):
        """Add visual feedback effects for clicks"""
        # Add ripple circles
        for size in [0.05, 0.07, 0.09]:
            circle = patches.Circle(
                (x, y), size,
                fill=False,
                color='yellow',
                alpha=1.0,
                linewidth=2
            )
            self.ax.add_patch(circle)
            self.click_effects.append({
                'patch': circle,
                'start_time': self.time,
                'type': 'ripple'
            })
        
        # Add flash effect
        flash = patches.Circle(
            (x, y), 0.04,
            color='yellow',
            alpha=0.8,
            zorder=3
        )
        self.ax.add_patch(flash)
        self.click_effects.append({
            'patch': flash,
            'start_time': self.time,
            'type': 'flash'
        })

    def find_closest_node(self, click_x, click_y):
        min_dist = float('inf')
        closest_node = None
        
        for node, (x, y) in self.pos.items():
            dist = np.sqrt((click_x - x)**2 + (click_y - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        return closest_node if min_dist < 0.1 else None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        node = self.find_closest_node(event.xdata, event.ydata)
        if node is None:
            return
            
        # Add visual feedback
        self.add_click_effect(self.pos[node][0], self.pos[node][1])
            
        # Inject signal based on node type
        if node.startswith('i'):
            self.network.inject_signal('inputs', int(node[1]))
        elif node.startswith('h'):
            self.network.inject_signal('layer1', int(node[1]))
        elif node.startswith('o'):
            self.network.inject_signal('output', int(node[1]))

    def update_effects(self):
        """Update visual feedback effects"""
        remaining_effects = []
        for effect in self.click_effects:
            age = self.time - effect['start_time']
            
            if age > 1.0:  # Remove old effects
                effect['patch'].remove()
                continue
                
            if effect['type'] == 'ripple':
                # Expand and fade ripples
                effect['patch'].set_alpha(1 - age)
                effect['patch'].set_radius(0.05 + age * 0.1)
            else:  # flash
                # Fade flash
                effect['patch'].set_alpha(0.8 * (1 - age))
            
            remaining_effects.append(effect)
            
        self.click_effects = remaining_effects

    def update(self, frame):
        values = self.network.forward(self.time)
        
        # Update visual effects
        self.update_effects()
        
        # Update node colors and labels
        colors = []
        for node in self.G.nodes():
            if node.startswith('i'):
                val = values['inputs'][int(node[1])]
            elif node.startswith('h'):
                val = values['layer1'][int(node[1])]
            else:
                val = values['output'][0]
            
            self.value_labels[node].set_text(f'{val:.1f}')
            
            normalized_val = (val + 100) / 200
            colors.append(plt.cm.Blues(normalized_val))
        
        self.nodes.set_color(colors)
        
        # Update edge colors and widths
        edge_colors = []
        edge_widths = []
        for edge in self.G.edges():
            if edge[0].startswith('i') and edge[1].startswith('h'):
                weight = self.network.weights['layer1'][
                    int(edge[1][1]), 
                    int(edge[0][1])
                ]
            else:
                weight = self.network.weights['layer2'][
                    0, 
                    int(edge[0][1])
                ]
            
            edge_colors.append('blue' if weight > 0 else 'red')
            edge_widths.append(abs(weight) * 2)
        
        self.edge_collection.set_color(edge_colors)
        self.edge_collection.set_linewidths(edge_widths)
        
        self.time += 0.1
        
        return (self.nodes, self.edge_collection, 
                *self.value_labels.values(),
                *[effect['patch'] for effect in self.click_effects])

    def animate(self):
        ani = FuncAnimation(
            self.fig, 
            self.update, 
            frames=None,
            interval=50,
            blit=True,
            cache_frame_data=False
        )
        plt.show()

if __name__ == "__main__":
    visualizer = NeuralNetworkVisualizer()
    visualizer.animate()