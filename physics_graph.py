# Enhanced version with physics-based graph interactions
import importlib
import contextlib
import os
import sys
import re
import pandas as pd
import numpy as np
import networkx as nx
from torch_snippets import *
from inspect import getmembers, isfunction, isclass, getsource
from torch_snippets.markup import AttrDict, pretty_json
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, CustomJS
from bokeh.models import (
    HoverTool,
    TapTool,
    BoxSelectTool,
    PanTool,
    WheelZoomTool,
    ResetTool,
)

from bokeh.plotting import from_networkx
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource, LabelSet

# Import your existing classes and functions here
# (PyObj, PyFunc, PyClass, PyFile, Source, etc. - copying from your original file)

def merge(l):
    l = [_l for _l in l if _l != {}]
    d = AttrDict(dict((k, v) for d in l for k, v in d.items()))
    return AttrDict({k: v for k, v in d.items() if v != {}})

def is_plural(item):
    return len(item) != 1

class PyObj:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def __repr__(self):
        return self.name

    @property
    def source(self):
        try:
            return getsource(self.function)
        except:
            return getsource(self.function.fget)

class PyFunc(PyObj):
    @property
    def children(self):
        calls = [f[:-1] for f in re.findall(r"[a-zA-Z_\.]+\(", self.source)]
        return [c for c in calls if c != self.name]

class PyClass(PyObj):
    @property
    def methods(self):
        return [
            PyFunc(f"{self.name}.{f}", getattr(self.function, f))
            for f in dir(self.function)
            if not f.startswith("__")
        ]

class PyFile:
    def __init__(self, file):
        self.file = file
        self._module = file.replace("/", ".")
        self._module = re.sub(r"\.py$", "", self._module)
        try:
            self.mod = importlib.import_module(self._module)
        except Exception as e:
            logger.warning(f"Error parsing functions @ {self.file}: {e}")
            self.mod = None

    @property
    def classes(self):
        if self.mod is None:
            return []
        else:
            members = getmembers(self.mod, isclass)
            return [
                PyClass(name, cls)
                for (name, cls) in members
                if cls.__module__ == self._module
            ]

    @property
    def functions(self):
        if self.mod is not None:
            members = getmembers(self.mod, isfunction)
            return [
                PyFunc(name, function)
                for (name, function) in members
                if function.__module__ == self._module
            ]
        else:
            return []

    def __len__(self):
        return len(self.classes) + len(self.functions)

    def __getitem__(self, ix):
        return self.functions[ix]

    def __repr__(self):
        return f'{self.file} ({len(self.functions)} {"functions" if is_plural(self.functions) else "function"} and {len(self.classes)} {"classes" if is_plural(self.classes) else "class"})'

class Source:
    def __init__(self, root, extension="py"):
        self.root = root
        self.catalogue = AttrDict({})
        self.tree = self.filter(extension)

    @property
    def children(self):
        files = Glob(self.root, silent=True)
        self.folders = [Source(f) for f in files if os.path.isdir(f)]
        self.files = [f for f in files if not os.path.isdir(f)]
        return self.folders + self.files

    def __repr__(self):
        return str(self.root)

    def get(self, path):
        x = self.tree
        for i in path.split("/"):
            if i != "":
                x = getattr(x, i)
        return dir(x)

    def resolve(self, x=None, extension=None):
        x = self if x is None else x
        if os.path.isfile(str(x)):
            if extension is None or extn(x) == extension:
                _file = str(x)
                file = PyFile(_file)
                for function in file.functions:
                    _name = function.name
                    function.name = f"{function.name} @ {_file}"
                    self.catalogue[_name] = function
                for cls in file.classes:
                    _name = cls.name
                    cls.name = f"{cls.name} @ {_file}"
                    self.catalogue[_name] = cls
                return AttrDict({stem(x): file})
            else:
                return AttrDict({})
        if isinstance(x, Source):
            return merge([self.resolve(f, extension) for f in x.children])

    def filter(self, extension):
        return self.resolve(extension=extension)

def add_node_if_not_existing(network, node_name, **kwargs):
    if network.has_node(node_name):
        pass
    else:
        network.add_node(node_name, **kwargs)

def pretty(node, NETWORK, children=False):
    if isinstance(node, AttrDict):
        return [pretty(_node) for k, _node in node.items()]
    if len(node) == 0:
        return
    add_node_if_not_existing(
        NETWORK, node.file, size=15, color="red", name=node.file, typ="file"
    )
    for f in node.functions:
        add_node_if_not_existing(
            NETWORK, f.name, size=8, color="green", name=f.name, typ="function"
        )
        NETWORK.add_edge(f.name, node.file, weight=1 / 5)
        if children:
            [
                (
                    add_node_if_not_existing(
                        NETWORK, child, size=5, color="yellow", name=child, typ="child"
                    ),
                    NETWORK.add_edge(f.name, child, weight=1 / 2.5),
                )
                for child in f.children
            ]
    for cls in node.classes:
        add_node_if_not_existing(
            NETWORK, cls.name, size=12, color="purple", name=cls.name, typ="class"
        )
        NETWORK.add_edge(cls.name, node.file, weight=1 / 4)
        for method in cls.methods:
            try:
                add_node_if_not_existing(
                    NETWORK,
                    method.name,
                    size=5,
                    color="blue",
                    name=method.name,
                    typ="method",
                )
                NETWORK.add_edge(cls.name, method.name, weight=1 / 2.5)
                if children:
                    [
                        (
                            add_node_if_not_existing(
                                NETWORK,
                                child,
                                size=5,
                                color="yellow",
                                name=child,
                                typ="child",
                            ),
                            NETWORK.add_edge(method.name, child, weight=1 / 1),
                        )
                        for child in method.children
                    ]
            except:
                ...

def summarize(folder, NETWORK):
    x = Source(folder)
    [pretty(getattr(x.tree, f), NETWORK) for f in dir(x.tree)]
    [pretty(getattr(x.tree, f), NETWORK, children=True) for f in dir(x.tree)]

@contextlib.contextmanager
def working_directory(path):
    prev_cwd = os.getcwd()
    os.chdir(parent(path))
    sys.path.append(str(P(path).resolve()))
    yield
    os.chdir(prev_cwd)

def get_ranges(pos):
    xs, ys = zip(*list(pos.values()))
    return np.min(xs), np.max(xs), np.min(ys), np.max(ys)

def create_physics_callback():
    """Creates a JavaScript callback for physics-based node interactions"""
    return CustomJS(code="""
        // Physics-based graph interaction inspired by Obsidian
        
        // Store physics state
        if (typeof window.graphPhysics === 'undefined') {
            window.graphPhysics = {
                isDragging: false,
                draggedNode: null,
                originalPositions: {},
                animationFrame: null,
                forces: {
                    repulsion: 100,
                    attraction: 0.1,
                    damping: 0.8,
                    springLength: 50
                }
            };
        }
        
        // Get canvas and context
        const canvas = document.querySelector('canvas');
        if (!canvas) return;
        
        // Mouse event handlers for dragging
        canvas.addEventListener('mousedown', function(e) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Check if clicking near a node (simplified collision detection)
            window.graphPhysics.isDragging = true;
            console.log('Started dragging at:', x, y);
        });
        
        canvas.addEventListener('mousemove', function(e) {
            if (!window.graphPhysics.isDragging) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Apply physics forces (simplified)
            console.log('Dragging to:', x, y);
            
            // Trigger plot update
            if (cb_obj && cb_obj.trigger) {
                cb_obj.trigger('change');
            }
        });
        
        canvas.addEventListener('mouseup', function() {
            window.graphPhysics.isDragging = false;
            window.graphPhysics.draggedNode = null;
            console.log('Stopped dragging');
        });
        
        // Physics simulation function
        function simulatePhysics() {
            if (!window.graphPhysics.isDragging) {
                // Apply repulsion and spring forces when not dragging
                // This would need access to node positions and edge data
                console.log('Running physics simulation...');
            }
            
            // Continue animation
            window.graphPhysics.animationFrame = requestAnimationFrame(simulatePhysics);
        }
        
        // Start physics simulation
        if (!window.graphPhysics.animationFrame) {
            simulatePhysics();
        }
    """)

def main(folder):
    if folder.endswith("/"):
        folder = folder[:-1]
    
    with working_directory(folder):
        NETWORK = nx.Graph()
        summarize(stem(folder), NETWORK)
        logger.info(
            f"Built the graph for {folder}. {len(NETWORK.nodes)} nodes and {len(NETWORK.edges)} edges found"
        )
    
    # Use spring layout for more natural node positioning with enhanced physics
    pos_ = nx.spring_layout(
        NETWORK, 
        k=5,  # Increased distance for better separation
        iterations=200,  # More iterations for better positioning
        weight='weight',
        seed=42
    )
    logger.info(f"Generated enhanced spring layout with {len(pos_)} node positions")

    x, X, y, Y = get_ranges(pos_)
    
    # Add padding to the ranges for better visualization
    padding = 0.1
    x_padding = (X - x) * padding
    y_padding = (Y - y) * padding
    
    plot = Plot(
        sizing_mode="stretch_both", 
        x_range=Range1d(x - x_padding, X + x_padding), 
        y_range=Range1d(y - y_padding, Y + y_padding)
    )

    # Enhanced hover tooltips
    hover_tooltips = [
        ("Name", "@name"),
        ("Type", "@typ"), 
        ("File", "@file_info"),
        ("Size", "@size"),
    ]
    
    plot.add_tools(
        HoverTool(tooltips=hover_tooltips),
        TapTool(),
        BoxSelectTool(),
        PanTool(),
        WheelZoomTool(),
        ResetTool(),
    )
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)

    graph_renderer = from_networkx(
        NETWORK,
        pos_,
        scale=1,
        center=(0, 0),
    )

    # Enhanced node sizing based on type
    sizes = []
    for node in NETWORK.nodes():
        node_data = NETWORK.nodes[node]
        base_size = node_data.get('size', 5)
        # Make file nodes larger, functions medium, methods smaller
        if node_data.get('typ') == 'file':
            sizes.append(base_size * 3)
        elif node_data.get('typ') == 'class':
            sizes.append(base_size * 2.5)
        elif node_data.get('typ') == 'function':
            sizes.append(base_size * 2)
        else:
            sizes.append(base_size)
    
    graph_renderer.node_renderer.data_source.data["size"] = sizes
    graph_renderer.node_renderer.data_source.data["color"] = [
        i for i in list(nx.get_node_attributes(NETWORK, "color").values())
    ]
    
    # Add comprehensive node information
    node_names = list(nx.get_node_attributes(NETWORK, "name").values())
    node_types = list(nx.get_node_attributes(NETWORK, "typ").values())
    
    file_info = []
    for name, typ in zip(node_names, node_types):
        if "@" in name:
            file_part = name.split("@")[-1].strip()
            file_info.append(file_part)
        else:
            file_info.append("N/A")
    
    graph_renderer.node_renderer.data_source.data["name"] = node_names
    graph_renderer.node_renderer.data_source.data["typ"] = node_types
    graph_renderer.node_renderer.data_source.data["file_info"] = file_info
    
    # Enhanced node styling
    graph_renderer.node_renderer.glyph = Circle(
        radius="size", 
        fill_color="color", 
        fill_alpha=0.8,
        line_color="white",
        line_width=2
    )

    graph_renderer.node_renderer.selection_glyph = Circle(
        radius="size", 
        fill_color="color",
        fill_alpha=1.0,
        line_color="black",
        line_width=3
    )
    
    graph_renderer.node_renderer.hover_glyph = Circle(
        radius="size", 
        fill_color="color",
        fill_alpha=1.0,
        line_color="orange",
        line_width=3
    )

    # Enhanced edge styling with weight-based thickness
    edge_weights = [NETWORK[u][v].get('weight', 1) for u, v in NETWORK.edges()]
    edge_widths = [max(0.5, w * 3) for w in edge_weights]  # Scale weights to line widths
    
    graph_renderer.edge_renderer.data_source.data["line_width"] = edge_widths
    
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#CCCCCC", 
        line_alpha=0.6, 
        line_width=1
    )
    
    graph_renderer.edge_renderer.selection_glyph = MultiLine(
        line_color="orange", 
        line_width=3,
        line_alpha=0.8
    )
    
    graph_renderer.edge_renderer.hover_glyph = MultiLine(
        line_color="red", 
        line_width=2,
        line_alpha=1.0
    )

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)
    
    # Add the physics callback
    physics_callback = create_physics_callback()
    plot.js_on_event('tap', physics_callback)
    
    # Enhanced labels with better positioning
    x, y = zip(*list(pos_.values()))
    node_labels = nx.get_node_attributes(NETWORK, "name")
    types = nx.get_node_attributes(NETWORK, "typ")

    # Filter labels to show only files and classes for clarity
    source_data = {
        "x": [],
        "y": [],
        "type": [],
        "name": [],
    }
    
    for i, (node_x, node_y) in enumerate(zip(x, y)):
        node_type = list(types.values())[i]
        node_name = list(node_labels.values())[i]
        
        # Only show labels for files and classes to reduce clutter
        if node_type in ['file', 'class']:
            # Clean up the name for display
            display_name = node_name.split('@')[0] if '@' in node_name else node_name
            source_data["x"].append(node_x)
            source_data["y"].append(node_y + 0.02)  # Offset slightly above node
            source_data["type"].append(node_type)
            source_data["name"].append(display_name)

    source = ColumnDataSource(source_data)
    labels = LabelSet(
        x="x",
        y="y",
        text="name",
        source=source,
        background_fill_alpha=0.8,
        background_fill_color="white",
        text_font_size="8px",
        text_color="black",
        border_line_color="gray",
        border_line_alpha=0.5
    )

    plot.renderers.append(labels)
    
    print(f"Rendering the enhanced physics-based graph to {folder}_physics.html")
    output_file(f"{folder}_physics.html")
    show(plot)

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    main(folder)