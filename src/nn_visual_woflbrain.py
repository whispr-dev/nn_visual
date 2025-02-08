#  Welcome to woflbrain interactive: a neural network that can be seen and played with!
#  it's a fascinating neural network architecture! It has several interesting components:
#
#  A central "ganglion" that includes:
#
#  A fuzzy logic layer
#  A crosstalk mechanism
#  An addition layer
#
#  Three parallel processing paths:
#
#  An RNN (Recurrent Neural Network)
#  A Kohonen SOM (Self-Organizing Map)
#  An LSTM (Long Short-Term Memory)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

class ComplexNeuralNetwork:
    def __init__(self):
        # Initialize different pathway weights with corrected dimensions
        self.weights = {
            'rnn': np.random.uniform(-0.5, 0.5, (20, 10)),
            'som': np.random.uniform(-0.5, 0.5, (10, 10)),
            'lstm': np.random.uniform(-0.5, 0.5, (20, 10)),
            'ganglion': {
                'fuzzy': np.random.uniform(-0.5, 0.5, (30, 30)),
                'crosstalk': np.random.uniform(-0.5, 0.5, (10, 30))
            },
            'output': np.random.uniform(-0.5, 0.5, (10, 10))
        }
        
        # Initialize values for each component
        self.values = {
            'input': np.zeros(10),
            'rnn': np.zeros(20),
            'som': np.zeros(10),
            'lstm': np.zeros(20),
            'ganglion': {
                'fuzzy': np.zeros(30),
                'crosstalk': np.zeros(10)
            },
            'merged': np.zeros(10),
            'output': np.zeros(10)
        }
        
        # Initialize injection points
        self.injected = {
            'input': np.zeros(10),
            'rnn': np.zeros(20),
            'som': np.zeros(10),
            'lstm': np.zeros(20),
            'ganglion': np.zeros(30),
            'output': np.zeros(10)
        }
        
        self.decay_rate = 0.95

    def analog_activation(self, x):
        """Analog activation keeping values between -100 and +100"""
        return np.clip(x, -100, 100)

    def forward(self, time):
        # Decay injected signals
        for key in self.injected:
            self.injected[key] *= self.decay_rate
        
        # Input with oscillation and injection
        self.values['input'] = np.array([
            20 * np.sin(time/10 + i) + self.injected['input'][i]
            for i in range(10)
        ])
        
        # Process through RNN pathway
        self.values['rnn'] = self.analog_activation(
            np.dot(self.values['input'], self.weights['rnn'].T) +
            self.injected['rnn']
        )
        
        # Process through SOM pathway
        self.values['som'] = self.analog_activation(
            np.dot(self.values['input'], self.weights['som'].T) +
            self.injected['som']
        )
        
        # Process through LSTM pathway
        self.values['lstm'] = self.analog_activation(
            np.dot(self.values['input'], self.weights['lstm'].T) +
            self.injected['lstm']
        )
        
        # Process through ganglion
        combined_input = np.concatenate([
            self.values['rnn'][:10],
            self.values['som'],
            self.values['lstm'][:10]
        ])
        
        self.values['ganglion']['fuzzy'] = self.analog_activation(
            np.dot(combined_input, self.weights['ganglion']['fuzzy'].T) +
            self.injected['ganglion']
        )
        
        self.values['ganglion']['crosstalk'] = self.analog_activation(
            np.dot(self.values['ganglion']['fuzzy'], self.weights['ganglion']['crosstalk'].T)
        )
        
        # Merge pathways
        self.values['merged'] = self.analog_activation(
            self.values['ganglion']['fuzzy'][:10] + 
            self.values['ganglion']['crosstalk']
        )
        
        # Final output
        self.values['output'] = self.analog_activation(
            np.dot(self.values['merged'], self.weights['output'].T) +
            self.injected['output']
        )
        
        return self.values

    def inject_signal(self, component, index, strength=50):
        """Inject a signal into a specific component"""
        if component in self.injected:
            if index < len(self.injected[component]):
                self.injected[component][index] = strength

class NetworkVisualizer:
    def __init__(self):
        self.network = ComplexNeuralNetwork()
        
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.axis('off')
        
        # Create network graph
        self.G = nx.DiGraph()
        
        # Define node positions for each component
        self.pos = {}
        
        # Input layer
        self.input_pos = {f'i{i}': (0.1, 0.1 + i*0.08) for i in range(10)}
        
        # Three parallel pathways
        self.rnn_pos = {f'rnn{i}': (0.3, 0.7 + i*0.04) for i in range(20)}
        self.som_pos = {f'som{i}': (0.3, 0.4 + i*0.04) for i in range(10)}
        self.lstm_pos = {f'lstm{i}': (0.3, 0.1 + i*0.04) for i in range(20)}
        
        # Ganglion
        self.ganglion_pos = {
            f'g{i}': (0.6, 0.2 + i*0.02) for i in range(30)
        }
        
        # Output layer
        self.output_pos = {f'o{i}': (0.9, 0.3 + i*0.05) for i in range(10)}
        
        # Combine all positions
        self.pos.update(self.input_pos)
        self.pos.update(self.rnn_pos)
        self.pos.update(self.som_pos)
        self.pos.update(self.lstm_pos)
        self.pos.update(self.ganglion_pos)
        self.pos.update(self.output_pos)
        
        # Add nodes
        self.G.add_nodes_from(self.pos.keys())
        
        # Add edges (connections)
        # Input to pathways
        for i in range(10):
            for r in range(20):
                self.G.add_edge(f'i{i}', f'rnn{r}')
            for s in range(10):
                self.G.add_edge(f'i{i}', f'som{s}')
            for l in range(20):
                self.G.add_edge(f'i{i}', f'lstm{l}')
        
        # Pathways to ganglion
        for r in range(20):
            for g in range(30):
                self.G.add_edge(f'rnn{r}', f'g{g}')
        for s in range(10):
            for g in range(30):
                self.G.add_edge(f'som{s}', f'g{g}')
        for l in range(20):
            for g in range(30):
                self.G.add_edge(f'lstm{l}', f'g{g}')
        
        # Ganglion to output
        for g in range(30):
            for o in range(10):
                self.G.add_edge(f'g{g}', f'o{o}')
        
        # Initialize nodes
        self.nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='lightblue',
            node_size=300
        )
        
        # Create edge collection
        edge_pos = []
        for (u, v) in self.G.edges():
            edge_pos.append([self.pos[u], self.pos[v]])
        self.edge_collection = LineCollection(
            edge_pos,
            colors=['gray'] * len(edge_pos),
            linewidths=0.5,
            alpha=0.3,
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
                verticalalignment='center',
                fontsize=6
            )
        
        # Add component labels
        self.ax.text(0.1, 0.05, 'Input Layer', ha='center')
        self.ax.text(0.3, 0.95, 'RNN', ha='center')
        self.ax.text(0.3, 0.35, 'SOM', ha='center')
        self.ax.text(0.3, 0.05, 'LSTM', ha='center')
        self.ax.text(0.6, 0.05, 'Ganglion', ha='center')
        self.ax.text(0.9, 0.05, 'Output', ha='center')
        
        self.time = 0
        
        # Add click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add title with instructions
        self.ax.set_title('Click on neurons to inject signals!')
        
        # Initialize click effects
        self.click_effects = []

    def add_click_effect(self, x, y):
        """Add visual feedback effects for clicks"""
        # Add ripple circles
        for size in [0.02, 0.03, 0.04]:
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
            (x, y), 0.015,
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
        """Find the closest node to the click position"""
        min_dist = float('inf')
        closest_node = None
        
        for node, (x, y) in self.pos.items():
            dist = np.sqrt((click_x - x)**2 + (click_y - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        return closest_node if min_dist < 0.05 else None

    def on_click(self, event):
        """Handle click events"""
        if event.inaxes != self.ax:
            return
        
        node = self.find_closest_node(event.xdata, event.ydata)
        if node is None:
            return
            
        # Add visual feedback
        self.add_click_effect(self.pos[node][0], self.pos[node][1])
            
        # Inject signal based on node type
        if node.startswith('i'):
            self.network.inject_signal('input', int(node[1:]))
        elif node.startswith('rnn'):
            self.network.inject_signal('rnn', int(node[3:]))
        elif node.startswith('som'):
            self.network.inject_signal('som', int(node[3:]))
        elif node.startswith('lstm'):
            self.network.inject_signal('lstm', int(node[4:]))
        elif node.startswith('g'):
            self.network.inject_signal('ganglion', int(node[1:]))
        elif node.startswith('o'):
            self.network.inject_signal('output', int(node[1:]))

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
                effect['patch'].set_radius(0.02 + age * 0.04)
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
                val = values['input'][int(node[1:])]
            elif node.startswith('rnn'):
                val = values['rnn'][int(node[3:])]
            elif node.startswith('som'):
                val = values['som'][int(node[3:])]
            elif node.startswith('lstm'):
                val = values['lstm'][int(node[4:])]
            elif node.startswith('g'):
                idx = int(node[1:])
                if idx < len(values['ganglion']['fuzzy']):
                    val = values['ganglion']['fuzzy'][idx]
                else:
                    val = 0
            else:  # output
                val = values['output'][int(node[1:])]
            
            self.value_labels[node].set_text(f'{val:.1f}')
            
            normalized_val = (val + 100) / 200
            colors.append(plt.cm.Blues(normalized_val))
        
        self.nodes.set_color(colors)
        
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
        plt.show(block=True)

if __name__ == '__main__':
    visualizer = NetworkVisualizer()
    visualizer.animate()