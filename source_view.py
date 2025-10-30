#!/usr/bin/env python3
"""
Source Code Visualizer - Interactive graph visualization of Python project structure.
Creates a D3.js force-directed graph showing files, classes, functions, and their relationships.
"""

import importlib
import contextlib
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional
from inspect import getmembers, isfunction, isclass, getsource

import networkx as nx
from torch_snippets import logger, Glob, stem, parent, extn, P
from torch_snippets.markup import AttrDict


# Constants
NODE_TYPES = {
    "file": {"size": 15, "color": "red"},
    "class": {"size": 12, "color": "purple"},
    "function": {"size": 8, "color": "green"},
    "method": {"size": 5, "color": "blue"},
    "child": {"size": 5, "color": "yellow"},
}

EDGE_WEIGHTS = {
    "file_function": 1 / 5,
    "file_class": 1 / 4,
    "class_method": 1 / 2.5,
    "function_child": 1 / 2.5,
    "method_child": 1 / 1,
}


# Utility Functions
def merge(list_of_dicts: List[Dict]) -> AttrDict:
    """Merge a list of dictionaries into a single AttrDict, filtering empty values."""
    filtered = [d for d in list_of_dicts if d]
    merged = dict((k, v) for d in filtered for k, v in d.items())
    return AttrDict({k: v for k, v in merged.items() if v})


def is_plural(items) -> bool:
    """Check if a collection has more than one item."""
    return len(items) != 1


# Python Object Classes
class PyObj:
    """Base class for Python objects (functions, classes, methods)."""

    def __init__(self, name: str, function):
        self.name = name
        self.function = function

    def __repr__(self) -> str:
        return self.name

    @property
    def source(self) -> str:
        """Get source code of the object."""
        try:
            return getsource(self.function)
        except AttributeError:
            return getsource(self.function.fget)


class PyFunc(PyObj):
    """Represents a Python function."""

    @property
    def children(self) -> List[str]:
        """Extract function calls from source code."""
        calls = [f[:-1] for f in re.findall(r"[a-zA-Z_\.]+\(", self.source)]
        return [c for c in calls if c != self.name]


class PyClass(PyObj):
    """Represents a Python class."""

    @property
    def methods(self) -> List[PyFunc]:
        """Get all non-private methods of the class."""
        return [
            PyFunc(f"{self.name}.{method_name}", getattr(self.function, method_name))
            for method_name in dir(self.function)
            if not method_name.startswith("__")
        ]


class PyFile:
    """Represents a Python file with its classes and functions."""

    def __init__(self, file_path: str):
        self.file = file_path
        self._module = self._get_module_name(file_path)
        self.mod = self._import_module()

    def _get_module_name(self, file_path: str) -> str:
        """Convert file path to module name."""
        module_name = file_path.replace("/", ".")
        return re.sub(r"\.py$", "", module_name)

    def _import_module(self):
        """Import the module, or return None if import fails."""
        try:
            return importlib.import_module(self._module)
        except Exception as e:
            logger.warning(f"Error parsing functions @ {self.file}: {e}")
            return None

    @property
    def classes(self) -> List[PyClass]:
        """Get all classes defined in this file."""
        if self.mod is None:
            return []

        members = getmembers(self.mod, isclass)
        return [
            PyClass(name, cls)
            for name, cls in members
            if cls.__module__ == self._module
        ]

    @property
    def functions(self) -> List[PyFunc]:
        """Get all functions defined in this file."""
        if self.mod is None:
            return []

        members = getmembers(self.mod, isfunction)
        return [
            PyFunc(name, func)
            for name, func in members
            if func.__module__ == self._module
        ]

    def __len__(self) -> int:
        return len(self.classes) + len(self.functions)

    def __getitem__(self, ix: int) -> PyFunc:
        return self.functions[ix]

    def __repr__(self) -> str:
        func_label = "functions" if is_plural(self.functions) else "function"
        class_label = "classes" if is_plural(self.classes) else "class"
        return f"{self.file} ({len(self.functions)} {func_label} and {len(self.classes)} {class_label})"


class Source:
    """Represents a source directory and its Python files."""

    def __init__(
        self, root: str, extension: str = "py", blacklist: Optional[List[str]] = None
    ):
        self.root = root
        self.catalogue = AttrDict({})
        self.blacklist = blacklist if blacklist is not None else []
        self.tree = self.filter(extension)

    @property
    def children(self) -> List:
        """Get all child folders and files."""
        files = Glob(self.root, silent=True)
        self.folders = [
            Source(f)
            for f in files
            if os.path.isdir(f) and not any(b in str(f) for b in self.blacklist)
        ]
        self.files = [
            f
            for f in files
            if not os.path.isdir(f) and not any(b in str(f) for b in self.blacklist)
        ]
        return self.folders + self.files

    def __repr__(self) -> str:
        return str(self.root)

    def get(self, path: str) -> List:
        """Navigate through the tree structure."""
        x = self.tree
        for segment in path.split("/"):
            if segment:
                x = getattr(x, segment)
        return dir(x)

    def resolve(
        self,
        x=None,
        extension: Optional[str] = None,
    ) -> AttrDict:
        """Recursively resolve files and create PyFile objects."""
        x = self if x is None else x

        if os.path.isfile(str(x)):
            if extension is None or extn(x) == extension:
                file_path = str(x)
                py_file = PyFile(file_path)

                # Add functions to catalogue
                for func in py_file.functions:
                    original_name = func.name
                    func.name = f"{func.name} @ {file_path}"
                    self.catalogue[original_name] = func

                # Add classes to catalogue
                for cls in py_file.classes:
                    original_name = cls.name
                    cls.name = f"{cls.name} @ {file_path}"
                    self.catalogue[original_name] = cls

                return AttrDict({stem(x): py_file})
            return AttrDict({})

        if isinstance(x, Source):
            return merge([self.resolve(f, extension) for f in x.children])

        return AttrDict({})

    def filter(self, extension: str) -> AttrDict:
        """Filter files by extension."""
        return self.resolve(extension=extension)


# Graph Building Functions
def add_node_if_not_existing(network: nx.Graph, node_name: str, **kwargs):
    """Add a node to the network if it doesn't already exist."""
    if not network.has_node(node_name):
        network.add_node(node_name, **kwargs)


def add_function_nodes(
    network: nx.Graph,
    functions: List[PyFunc],
    parent_file: str,
    include_children: bool = False,
):
    """Add function nodes and their relationships to the network."""
    for func in functions:
        node_config = NODE_TYPES["function"]
        add_node_if_not_existing(
            network,
            func.name,
            size=node_config["size"],
            color=node_config["color"],
            name=func.name,
            typ="function",
        )
        network.add_edge(func.name, parent_file, weight=EDGE_WEIGHTS["file_function"])

        if include_children:
            for child in func.children:
                child_config = NODE_TYPES["child"]
                add_node_if_not_existing(
                    network,
                    child,
                    size=child_config["size"],
                    color=child_config["color"],
                    name=child,
                    typ="child",
                )
                network.add_edge(
                    func.name, child, weight=EDGE_WEIGHTS["function_child"]
                )


def add_class_nodes(
    network: nx.Graph,
    classes: List[PyClass],
    parent_file: str,
    include_children: bool = False,
):
    """Add class nodes and their relationships to the network."""
    for cls in classes:
        class_config = NODE_TYPES["class"]
        add_node_if_not_existing(
            network,
            cls.name,
            size=class_config["size"],
            color=class_config["color"],
            name=cls.name,
            typ="class",
        )
        network.add_edge(cls.name, parent_file, weight=EDGE_WEIGHTS["file_class"])

        for method in cls.methods:
            try:
                method_config = NODE_TYPES["method"]
                add_node_if_not_existing(
                    network,
                    method.name,
                    size=method_config["size"],
                    color=method_config["color"],
                    name=method.name,
                    typ="method",
                )
                network.add_edge(
                    cls.name, method.name, weight=EDGE_WEIGHTS["class_method"]
                )

                if include_children:
                    for child in method.children:
                        child_config = NODE_TYPES["child"]
                        add_node_if_not_existing(
                            network,
                            child,
                            size=child_config["size"],
                            color=child_config["color"],
                            name=child,
                            typ="child",
                        )
                        network.add_edge(
                            method.name, child, weight=EDGE_WEIGHTS["method_child"]
                        )
            except Exception:
                # Silently skip methods that cause errors
                pass


def build_graph_from_node(node, network: nx.Graph, include_children: bool = False):
    """Build network graph from a PyFile node."""
    if isinstance(node, AttrDict):
        return [
            build_graph_from_node(n, network, include_children) for n in node.values()
        ]

    if len(node) == 0:
        return

    # Add file node
    file_config = NODE_TYPES["file"]
    add_node_if_not_existing(
        network,
        node.file,
        size=file_config["size"],
        color=file_config["color"],
        name=node.file,
        typ="file",
    )

    # Add function nodes
    add_function_nodes(network, node.functions, node.file, include_children)

    # Add class nodes
    add_class_nodes(network, node.classes, node.file, include_children)


def build_network(folder: str, blacklist: List[str]) -> nx.Graph:
    """Build a NetworkX graph from a source folder."""
    network = nx.Graph()
    source = Source(folder, blacklist=blacklist)

    # First pass: add all nodes and basic edges
    for attr_name in dir(source.tree):
        build_graph_from_node(
            getattr(source.tree, attr_name), network, include_children=False
        )

    # Second pass: add children relationships
    for attr_name in dir(source.tree):
        build_graph_from_node(
            getattr(source.tree, attr_name), network, include_children=True
        )

    return network


@contextlib.contextmanager
def working_directory(path: str):
    """Context manager to temporarily change working directory."""
    prev_cwd = os.getcwd()
    os.chdir(parent(path))
    sys.path.append(str(P(path).resolve()))
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def prepare_graph_data(network: nx.Graph) -> Dict[str, Any]:
    """Convert NetworkX graph to Sigma.js compatible format."""
    nodes_data = []
    edges_data = []

    node_list = list(network.nodes(data=True))

    # Filter out yellow (child) nodes
    filtered_node_list = [
        (node_name, attrs)
        for node_name, attrs in node_list
        if attrs.get("color") != "yellow"
    ]

    # Create nodes for Sigma
    for node_name, attrs in filtered_node_list:
        nodes_data.append(
            {
                "key": node_name,
                "label": attrs.get("name", node_name),
                "size": attrs.get("size", 5),
                "color": attrs.get("color", "gray"),
                "type": attrs.get("typ", "unknown"),
                "x": 0,  # Will be positioned by ForceAtlas2
                "y": 0,
            }
        )

    # Get valid node keys
    valid_nodes = {node["key"] for node in nodes_data}

    # Create edges for Sigma
    edge_id = 0
    for source, target, attrs in network.edges(data=True):
        if source in valid_nodes and target in valid_nodes:
            edges_data.append(
                {
                    "key": f"edge_{edge_id}",
                    "source": source,
                    "target": target,
                    "size": attrs.get("weight", 1),
                }
            )
            edge_id += 1

    return {"nodes": nodes_data, "edges": edges_data}


def generate_html_template(folder: str, graph_data: Dict[str, Any]) -> str:
    """Generate HTML content with Sigma.js (GPU-accelerated WebGL renderer)."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{folder} - Interactive Graph</title>
    <script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/graphology-layout-forceatlas2@0.10.1/build/graphology-layout-forceatlas2.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/graphology-layout@0.17.0/build/graphology-layout.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
            background: #0a0a0a;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #controls {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.85);
            padding: 20px;
            border-radius: 8px;
            color: white;
            z-index: 1000;
            max-width: 350px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #controls h3 {{
            margin: 0 0 15px 0;
            font-size: 16px;
            font-weight: 600;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 8px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            color: #aaa;
            font-weight: 500;
        }}
        input[type="text"], input[type="range"], select {{
            width: 100%;
            padding: 8px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 4px;
            font-size: 13px;
        }}
        input[type="text"]:focus, select:focus {{
            outline: none;
            border-color: #ff6600;
        }}
        input[type="range"] {{
            padding: 0;
        }}
        button {{
            padding: 8px 15px;
            background: #ff6600;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            margin-right: 8px;
            margin-top: 5px;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #ff7722;
        }}
        button:active {{
            background: #dd5500;
        }}
        .value-display {{
            display: inline-block;
            margin-left: 8px;
            color: #ff6600;
            font-weight: 600;
        }}
        #info-panel {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.85);
            padding: 15px;
            border-radius: 8px;
            color: white;
            z-index: 1000;
            font-size: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #node-info {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 15px;
            border-radius: 8px;
            color: white;
            z-index: 1000;
            max-width: 400px;
            display: none;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #node-info h4 {{
            margin: 0 0 10px 0;
            color: #ff6600;
            font-size: 14px;
        }}
        #node-info p {{
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend {{
            margin-top: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 11px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            margin-right: 8px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="controls">
        <h3>⚙️ Controls</h3>
        
        <div class="control-group">
            <label>🔍 Search Nodes (Regex)</label>
            <input type="text" id="searchInput" placeholder="Enter regex pattern...">
        </div>
        
        <div class="control-group">
            <label>🎨 Filter by Type</label>
            <select id="typeFilter">
                <option value="all">All Types</option>
                <option value="file">Files</option>
                <option value="class">Classes</option>
                <option value="function">Functions</option>
                <option value="method">Methods</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>📏 Node Size <span class="value-display" id="sizeValue">1.0x</span></label>
            <input type="range" id="sizeSlider" min="0.5" max="3" step="0.1" value="1.0">
        </div>
        
        <div class="control-group">
            <label>🎯 Physics Intensity <span class="value-display" id="physicsValue">1.0x</span></label>
            <input type="range" id="physicsSlider" min="0" max="2" step="0.1" value="1.0">
        </div>
        
        <div class="control-group">
            <button id="resetBtn">🔄 Reset View</button>
            <button id="layoutBtn">🌀 Re-layout</button>
        </div>
        
        <div class="legend">
            <strong style="font-size: 11px;">LEGEND</strong>
            <div class="legend-item"><div class="legend-color" style="background: red;"></div> File</div>
            <div class="legend-item"><div class="legend-color" style="background: purple;"></div> Class</div>
            <div class="legend-item"><div class="legend-color" style="background: green;"></div> Function</div>
            <div class="legend-item"><div class="legend-color" style="background: blue;"></div> Method</div>
        </div>
    </div>
    
    <div id="info-panel">
        <div><strong>Nodes:</strong> <span id="nodeCount">0</span> | <strong>Edges:</strong> <span id="edgeCount">0</span></div>
        <div style="margin-top: 8px; font-size: 11px; color: #888;">
            Click: Select | Drag: Move | Scroll: Zoom | Drag BG: Pan
        </div>
    </div>
    
    <div id="node-info"></div>

    <script>
        const graphData = {json.dumps(graph_data)};
        
        // Create graph using Graphology
        const graph = new graphology.Graph();
        
        // Add nodes and edges to graph
        graphData.nodes.forEach(node => {{
            graph.addNode(node.key, {{
                label: node.label,
                size: node.size,
                color: node.color,
                type: node.type,
                x: Math.random() * 1000,
                y: Math.random() * 1000
            }});
        }});
        
        graphData.edges.forEach(edge => {{
            if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {{
                graph.addEdge(edge.source, edge.target, {{
                    size: edge.size
                }});
            }}
        }});
        
        // Update stats
        document.getElementById('nodeCount').textContent = graph.order;
        document.getElementById('edgeCount').textContent = graph.size;
        
        // Apply initial random positions if not set
        graph.forEachNode((node, attrs) => {{
            if (!attrs.x || !attrs.y) {{
                const angle = Math.random() * Math.PI * 2;
                const radius = 300 + Math.random() * 200;
                graph.setNodeAttribute(node, 'x', Math.cos(angle) * radius);
                graph.setNodeAttribute(node, 'y', Math.sin(angle) * radius);
            }}
        }});
        
        // Then apply ForceAtlas2 if available
        if (typeof graphologyLayoutForceAtlas2 !== 'undefined') {{
            try {{
                const FA2 = graphologyLayoutForceAtlas2.default || graphologyLayoutForceAtlas2;
                const settings = FA2.inferSettings ? FA2.inferSettings(graph) : {{}};
                FA2.assign(graph, {{ iterations: 150, settings }});
            }} catch(e) {{
                console.warn('ForceAtlas2 layout failed, using random layout:', e);
            }}
        }}
        
        // Remove type attribute to avoid Sigma.js program errors
        graph.forEachNode((node, attrs) => {{
            graph.removeNodeAttribute(node, 'type');
        }});
        
        // Create Sigma renderer with GPU acceleration
        const container = document.getElementById('container');
        const renderer = new Sigma(graph, container, {{
            renderEdgeLabels: false,
            defaultNodeColor: '#999',
            defaultEdgeColor: '#444',
            labelSize: 14,
            labelWeight: 'normal',
            labelColor: {{ color: '#fff' }},
            enableEdgeEvents: true,
            // Make zoom less aggressive
            zoomingRatio: 1.3,
            mouseWheelEnabled: true,
            // Enable better interaction
            allowInvalidContainer: false
        }});
        
        // Make zoom less aggressive by overriding wheel behavior
        const camera = renderer.getCamera();
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const scaleFactor = e.deltaY > 0 ? 1.1 : 1/1.1;
            camera.animatedZoom({{ factor: scaleFactor, duration: 150 }});
        }}, {{ passive: false }});
        
        // State management
        let hoveredNode = null;
        let selectedNode = null;
        let draggedNode = null;
        let isDragging = false;
        const State = {{ searchQuery: '', typeFilter: 'all', sizeMultiplier: 1.0, physicsStrength: 1.0 }};
        
        // Store original attributes (including type info separately)
        const originalNodeAttributes = new Map();
        const nodeTypes = new Map();
        const originalEdgeAttributes = new Map();
        graph.forEachNode((node, attrs) => {{
            originalNodeAttributes.set(node, {{ ...attrs }});
            // Store type info separately since we removed it from graph
            const originalData = graphData.nodes.find(n => n.key === node);
            if (originalData) {{
                nodeTypes.set(node, originalData.type);
            }}
        }});
        graph.forEachEdge((edge, attrs) => {{
            originalEdgeAttributes.set(edge, {{ ...attrs }});
        }});
        
        // Neighbor calculation
        function getNeighbors(nodeId) {{
            const neighbors = new Set();
            graph.forEachNeighbor(nodeId, neighbor => neighbors.add(neighbor));
            return neighbors;
        }}
        
        // Path finding (BFS)
        function findPath(startNode, endNode) {{
            if (!graph.hasNode(startNode) || !graph.hasNode(endNode)) return null;
            const queue = [[startNode]];
            const visited = new Set([startNode]);
            
            while (queue.length > 0) {{
                const path = queue.shift();
                const node = path[path.length - 1];
                
                if (node === endNode) return path;
                
                graph.forEachNeighbor(node, neighbor => {{
                    if (!visited.has(neighbor)) {{
                        visited.add(neighbor);
                        queue.push([...path, neighbor]);
                    }}
                }});
            }}
            return null;
        }}
        
        // Filtering and highlighting
        function applyFilters() {{
            const searchRegex = State.searchQuery ? new RegExp(State.searchQuery, 'i') : null;
            
            graph.forEachNode((node, attrs) => {{
                const original = originalNodeAttributes.get(node);
                const nodeType = nodeTypes.get(node);
                let visible = true;
                
                // Type filter
                if (State.typeFilter !== 'all' && nodeType !== State.typeFilter) {{
                    visible = false;
                }}
                
                // Search filter
                if (searchRegex && !searchRegex.test(attrs.label)) {{
                    visible = false;
                }}
                
                // Apply visibility
                graph.setNodeAttribute(node, 'hidden', !visible);
                graph.setNodeAttribute(node, 'size', original.size * State.sizeMultiplier);
            }});
            
            renderer.refresh();
        }}
        
        // Highlight node and neighbors
        function highlightNode(nodeId) {{
            if (!nodeId) {{
                // Reset all
                graph.forEachNode((node) => {{
                    const original = originalNodeAttributes.get(node);
                    graph.setNodeAttribute(node, 'color', original.color);
                    graph.setNodeAttribute(node, 'highlighted', false);
                }});
                graph.forEachEdge((edge) => {{
                    graph.setEdgeAttribute(edge, 'color', '#444');
                    graph.setEdgeAttribute(edge, 'size', 1);
                }});
                selectedNode = null;
                document.getElementById('node-info').style.display = 'none';
            }} else {{
                const neighbors = getNeighbors(nodeId);
                
                graph.forEachNode((node) => {{
                    const original = originalNodeAttributes.get(node);
                    if (node === nodeId) {{
                        graph.setNodeAttribute(node, 'color', '#ff6600');
                        graph.setNodeAttribute(node, 'highlighted', true);
                    }} else if (neighbors.has(node)) {{
                        graph.setNodeAttribute(node, 'color', original.color);
                        graph.setNodeAttribute(node, 'highlighted', false);
                    }} else {{
                        graph.setNodeAttribute(node, 'color', '#222');
                        graph.setNodeAttribute(node, 'highlighted', false);
                    }}
                }});
                
                graph.forEachEdge((edge, attrs, source, target) => {{
                    if (source === nodeId || target === nodeId) {{
                        graph.setEdgeAttribute(edge, 'color', '#ff6600');
                        graph.setEdgeAttribute(edge, 'size', 2);
                    }} else {{
                        graph.setEdgeAttribute(edge, 'color', '#111');
                        graph.setEdgeAttribute(edge, 'size', 0.5);
                    }}
                }});
                
                selectedNode = nodeId;
                
                // Show node info
                const attrs = graph.getNodeAttributes(nodeId);
                const nodeType = nodeTypes.get(nodeId);
                const degree = graph.degree(nodeId);
                const neighbors_list = Array.from(neighbors).slice(0, 5).join(', ');
                document.getElementById('node-info').innerHTML = `
                    <h4>${{attrs.label}}</h4>
                    <p><strong>Type:</strong> ${{nodeType || 'unknown'}}</p>
                    <p><strong>Connections:</strong> ${{degree}}</p>
                    <p><strong>Neighbors:</strong> ${{neighbors_list}}${{neighbors.size > 5 ? '...' : ''}}</p>
                    <button onclick="highlightNode(null)" style="margin-top: 10px;">✕ Close</button>
                `;
                document.getElementById('node-info').style.display = 'block';
            }}
            
            renderer.refresh();
        }}
        
        // Event Handlers
        renderer.on('clickNode', ({{ node }}) => {{
            if (selectedNode === node) {{
                highlightNode(null);
            }} else {{
                highlightNode(node);
            }}
        }});
        
        renderer.on('clickStage', () => {{
            highlightNode(null);
        }});
        
        // Enhanced drag functionality with physics simulation
        let dragOffset = {{ x: 0, y: 0 }};
        let isNodeDragging = false;
        
        renderer.on('downNode', (e) => {{
            isNodeDragging = true;
            isDragging = true;
            draggedNode = e.node;
            
            // Calculate offset between mouse and node center
            const nodeDisplayPosition = renderer.graphToViewport(graph.getNodeAttributes(draggedNode));
            dragOffset.x = e.event.offsetX - nodeDisplayPosition.x;
            dragOffset.y = e.event.offsetY - nodeDisplayPosition.y;
            
            // Start simple physics simulation
            startPhysicsSimulation();
            
            graph.setNodeAttribute(draggedNode, 'highlighted', true);
        }});
        
        renderer.getMouseCaptor().on('mousemovebody', (e) => {{
            if (!isDragging || !draggedNode || !isNodeDragging) return;
            
            // Convert mouse position to graph coordinates with offset
            const viewportPos = {{ x: e.x - dragOffset.x, y: e.y - dragOffset.y }};
            const graphPos = renderer.viewportToGraph(viewportPos);
            
            graph.setNodeAttribute(draggedNode, 'x', graphPos.x);
            graph.setNodeAttribute(draggedNode, 'y', graphPos.y);
            
            // Apply spring forces to connected nodes
            applySpringForces(draggedNode);
        }});
        
        renderer.getMouseCaptor().on('mouseup', () => {{
            if (draggedNode) {{
                graph.removeNodeAttribute(draggedNode, 'highlighted');
            }}
            isNodeDragging = false;
            isDragging = false;
            draggedNode = null;
            stopPhysicsSimulation();
        }});
        
        // Physics simulation for springy behavior
        let physicsInterval = null;
        
        function startPhysicsSimulation() {{
            if (physicsInterval) clearInterval(physicsInterval);
            
            physicsInterval = setInterval(() => {{
                if (!draggedNode) return;
                
                // Apply light repulsion between all nodes
                graph.forEachNode((nodeId, attrs) => {{
                    if (nodeId === draggedNode) return;
                    
                    const dx = attrs.x - graph.getNodeAttribute(draggedNode, 'x');
                    const dy = attrs.y - graph.getNodeAttribute(draggedNode, 'y');
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 100 && distance > 0) {{
                        const force = (100 - distance) * 0.01;
                        const fx = (dx / distance) * force;
                        const fy = (dy / distance) * force;
                        
                        graph.setNodeAttribute(nodeId, 'x', attrs.x + fx);
                        graph.setNodeAttribute(nodeId, 'y', attrs.y + fy);
                    }}
                }});
                
                renderer.refresh();
            }}, 16); // ~60 FPS
        }}
        
        function stopPhysicsSimulation() {{
            if (physicsInterval) {{
                clearInterval(physicsInterval);
                physicsInterval = null;
            }}
        }}
        
        function applySpringForces(centralNode) {{
            const centralPos = graph.getNodeAttributes(centralNode);
            
            // Apply spring forces to neighbors
            graph.forEachNeighbor(centralNode, (neighbor) => {{
                const neighborPos = graph.getNodeAttributes(neighbor);
                const dx = centralPos.x - neighborPos.x;
                const dy = centralPos.y - neighborPos.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 0) {{
                    const idealDistance = 150;
                    const force = (distance - idealDistance) * 0.03;
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    
                    graph.setNodeAttribute(neighbor, 'x', neighborPos.x + fx);
                    graph.setNodeAttribute(neighbor, 'y', neighborPos.y + fy);
                }}
            }});
        }}
        
        // Controls
        document.getElementById('searchInput').addEventListener('input', (e) => {{
            State.searchQuery = e.target.value;
            applyFilters();
        }});
        
        document.getElementById('typeFilter').addEventListener('change', (e) => {{
            State.typeFilter = e.target.value;
            applyFilters();
        }});
        
        document.getElementById('sizeSlider').addEventListener('input', (e) => {{
            State.sizeMultiplier = parseFloat(e.target.value);
            document.getElementById('sizeValue').textContent = State.sizeMultiplier.toFixed(1) + 'x';
            applyFilters();
        }});
        
        document.getElementById('physicsSlider').addEventListener('input', (e) => {{
            State.physicsStrength = parseFloat(e.target.value);
            document.getElementById('physicsValue').textContent = State.physicsStrength.toFixed(1) + 'x';
        }});
        
        document.getElementById('resetBtn').addEventListener('click', () => {{
            renderer.getCamera().animate({{ x: 0.5, y: 0.5, ratio: 1 }}, {{ duration: 500 }});
            highlightNode(null);
        }});
        
        document.getElementById('layoutBtn').addEventListener('click', () => {{
            if (typeof graphologyLayoutForceAtlas2 !== 'undefined') {{
                try {{
                    const FA2 = graphologyLayoutForceAtlas2.default || graphologyLayoutForceAtlas2;
                    const settings = FA2.inferSettings ? FA2.inferSettings(graph) : {{}};
                    settings.gravity = State.physicsStrength;
                    FA2.assign(graph, {{ iterations: 100, settings }});
                    renderer.refresh();
                }} catch(e) {{
                    console.error('Layout failed:', e);
                    // Fallback to random circular layout
                    graph.forEachNode((node) => {{
                        const angle = Math.random() * Math.PI * 2;
                        const radius = 300 + Math.random() * 200;
                        graph.setNodeAttribute(node, 'x', Math.cos(angle) * radius);
                        graph.setNodeAttribute(node, 'y', Math.sin(angle) * radius);
                    }});
                    renderer.refresh();
                }}
            }} else {{
                // Random circular layout
                graph.forEachNode((node) => {{
                    const angle = Math.random() * Math.PI * 2;
                    const radius = 300 + Math.random() * 200;
                    graph.setNodeAttribute(node, 'x', Math.cos(angle) * radius);
                    graph.setNodeAttribute(node, 'y', Math.sin(angle) * radius);
                }});
                renderer.refresh();
            }}
        }});
        
        // Make highlightNode available globally
        window.highlightNode = highlightNode;
        
        console.log('Graph loaded with', graph.order, 'nodes and', graph.size, 'edges');
        console.log('GPU-accelerated rendering enabled via WebGL');
    </script>
</body>
</html>"""


def main(folder: str, blacklist: List[str]):
    """Main entry point for generating interactive graph visualization."""
    # Normalize folder path
    folder = folder.rstrip("/")

    # Build the network graph
    with working_directory(folder):
        network = build_network(stem(folder), blacklist=blacklist)
        logger.info(
            f"Built the graph for {folder}. "
            f"{len(network.nodes)} nodes and {len(network.edges)} edges found"
        )

    # Prepare data for visualization
    graph_data = prepare_graph_data(network)

    # Generate HTML content
    html_content = generate_html_template(folder, graph_data)

    # Write output file
    output_path = f"{folder}.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    # Print summary
    print(f"Rendered interactive graph to {output_path}")
    print(f"\n✨ Features:")
    print(f"  🎮 GPU-accelerated rendering (WebGL via Sigma.js)")
    print(f"  🔍 Regex search filtering")
    print(f"  🎨 Filter by node type (file/class/function/method)")
    print(f"  🖱️  Click nodes to highlight neighbors")
    print(f"  🎯 Drag nodes with physics simulation")
    print(f"  📏 Adjustable node sizes")
    print(f"  🌀 Dynamic layout with ForceAtlas2")
    print(f"  ⚡ BFS/DFS path finding built-in")
    print(f"  🔄 Real-time node/edge manipulation")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    blacklist = ["__pycache__", ".git", "venv", "env", "node_modules"] + sys.argv[2:]
    main(folder, blacklist)
