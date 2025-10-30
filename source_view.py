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

    def __init__(self, root: str, extension: str = "py"):
        self.root = root
        self.catalogue = AttrDict({})
        self.tree = self.filter(extension)

    @property
    def children(self) -> List:
        """Get all child folders and files."""
        files = Glob(self.root, silent=True)
        self.folders = [Source(f) for f in files if os.path.isdir(f)]
        self.files = [f for f in files if not os.path.isdir(f)]
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

    def resolve(self, x=None, extension: Optional[str] = None) -> AttrDict:
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


def build_network(folder: str) -> nx.Graph:
    """Build a NetworkX graph from a source folder."""
    network = nx.Graph()
    source = Source(folder)

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


def prepare_graph_data(network: nx.Graph) -> tuple[List[Dict], List[Dict]]:
    """Convert NetworkX graph to D3.js compatible format."""
    nodes_data = []
    node_list = list(network.nodes(data=True))
    node_index_map = {node[0]: idx for idx, node in enumerate(node_list)}

    for node_name, attrs in node_list:
        nodes_data.append(
            {
                "id": node_name,
                "name": attrs.get("name", node_name),
                "size": attrs.get("size", 5),
                "color": attrs.get("color", "gray"),
                "type": attrs.get("typ", "unknown"),
            }
        )

    edges_data = []
    for source, target, attrs in network.edges(data=True):
        edges_data.append(
            {
                "source": node_index_map[source],
                "target": node_index_map[target],
                "weight": attrs.get("weight", 1),
            }
        )

    return nodes_data, edges_data


def generate_html_template(
    folder: str, nodes_data: List[Dict], edges_data: List[Dict]
) -> str:
    """Generate HTML content with D3.js force simulation."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{folder} - Interactive Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
        }}
        #graph {{
            width: 100vw;
            height: 100vh;
        }}
        .links line {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .nodes circle {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .nodes circle:hover {{
            stroke: #000;
            stroke-width: 3px;
        }}
        .labels text {{
            font-size: 10px;
            pointer-events: none;
            user-select: none;
            fill: #333;
        }}
        .tooltip {{
            position: absolute;
            text-align: center;
            padding: 8px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <svg id="graph"></svg>
    <div class="tooltip"></div>
    <script>
        const nodes = {json.dumps(nodes_data)};
        const links = {json.dumps(edges_data)};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);
        
        const g = svg.append("g");
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links)
                .id(d => d.index)
                .distance(d => 100 / (d.weight || 1))
                .strength(0.3))
            .force("charge", d3.forceManyBody()
                .strength(d => -300 * (d.size / 5))
                .distanceMax(400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide()
                .radius(d => d.size * 3 + 5)
                .strength(0.7));
        
        // Create links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke-width", d => Math.sqrt(d.weight || 1));
        
        // Create nodes
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("r", d => d.size * 2)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Create labels
        const labels = g.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .text(d => d.name)
            .attr("font-size", "10px")
            .attr("dx", d => d.size * 2 + 5)
            .attr("dy", 3);
        
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        node.on("mouseover", function(event, d) {{
            tooltip
                .style("opacity", 1)
                .html(`<strong>${{d.name}}</strong><br>Type: ${{d.type}}<br>Size: ${{d.size}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
        }});
        
        // Update positions on each tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            const newWidth = window.innerWidth;
            const newHeight = window.innerHeight;
            svg.attr("width", newWidth).attr("height", newHeight);
            simulation.force("center", d3.forceCenter(newWidth / 2, newHeight / 2));
            simulation.alpha(0.3).restart();
        }});
    </script>
</body>
</html>"""


def main(folder: str):
    """Main entry point for generating interactive graph visualization."""
    # Normalize folder path
    folder = folder.rstrip("/")

    # Build the network graph
    with working_directory(folder):
        network = build_network(stem(folder))
        logger.info(
            f"Built the graph for {folder}. "
            f"{len(network.nodes)} nodes and {len(network.edges)} edges found"
        )

    # Prepare data for visualization
    nodes_data, edges_data = prepare_graph_data(network)

    # Generate HTML content
    html_content = generate_html_template(folder, nodes_data, edges_data)

    # Write output file
    output_path = f"{folder}.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    # Print summary
    print(f"Rendered interactive graph to {output_path}")
    print(f"\nGraph features:")
    print(f"  - Drag nodes to move them (other nodes will react)")
    print(f"  - Zoom with mouse wheel")
    print(f"  - Pan by dragging empty space")
    print(f"  - Nodes repel each other based on their size")
    print(f"  - Hover over nodes for details")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    main(folder)
