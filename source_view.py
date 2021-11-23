#/bin/python
import importlib
import contextlib
import os
import networkx as nx
from torch_snippets import *
from inspect import getmembers, isfunction, isclass, getsource
from torch_snippets.markup import AttrDict, pretty_json
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle
from bokeh.models import HoverTool, TapTool, BoxSelectTool, PanTool, WheelZoomTool, ResetTool
from bokeh.models.graphs import from_networkx
from bokeh.models.graphs import NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource, LabelSet

def merge(l):
    d = AttrDict(dict((k,v) for d in l for k,v in d.items()))
    return AttrDict({k:v for k,v in d.items() if v != {}})

def is_plural(item):
    return len(item) != 1

class PyObj:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def __repr__(self): return self.name

    @property
    def source(self):
        return getsource(self.function)

class PyFunc(PyObj):
    @property
    def children(self):
        calls = [f[:-1] for f in re.findall(r'[a-zA-Z_\.]+\(', self.source)]
        return [c for c in calls if c!=self.name]

class PyClass(PyObj):
    @property
    def methods(self):
        return [PyFunc(f'{self.name}.{f}', getattr(self.function, f)) for f in dir(self.function) if not f.startswith('__')]

class PyFile:
    def __init__(self, file):
        self.file = file
        self._module = file.replace('/', '.').replace('.py','')
        try:
            self.mod = importlib.import_module(self.file.replace('/','.').replace('.py',''))
        except Exception as e:
            logger.warning(f'Error parsing functions @ {self.file}: {e}')
            self.mod = None

    @property
    def classes(self):
        if self.mod is None:
            return []
        else:
            members = getmembers(self.mod, isclass)
            return [PyClass(name, cls)
                for (name, cls) in members
                if cls.__module__ == self._module]

    @property
    def functions(self):
        if self.mod is not None:
            members = getmembers(self.mod, isfunction)
            return [PyFunc(name, function)
                for (name, function) in members
                if function.__module__ == self._module]
        else:
            return []
        
    def __len__(self):
        return len(self.classes) + len(self.functions)
    
    def __getitem__(self, ix):
        return self.functions[ix]

    def  __repr__(self):
        return f'{self.file} ({len(self.functions)} {"functions" if is_plural(self.functions) else "function"} and {len(self.classes)} {"classes" if is_plural(self.classes) else "class"})'

class Source:
    def __init__(self, root, extension='py'):
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
        for i in path.split('/'):
            if i!='':
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
                    function.name = f'{function.name} @ {_file}'
                    self.catalogue[_name] = function
                for cls in file.classes:
                    _name = cls.name
                    cls.name = f'{cls.name} @ {_file}'
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
        return [pretty(_node) for k,_node in node.items()]
    if len(node) == 0: return
    add_node_if_not_existing(NETWORK, node.file, size=15, color='red', name=node.file, typ='file')
    for f in node.functions:
        add_node_if_not_existing(NETWORK, f.name, size=8, color='green', name=f.name, typ='function')
        NETWORK.add_edge(f.name, node.file, weight=1/5)
        if children:
            [(add_node_if_not_existing(NETWORK, child, size=5, color='yellow', name=child, typ='child'), NETWORK.add_edge(f.name, child, weight=1/2.5)) 
            for child in f.children]
    for cls in node.classes:
        add_node_if_not_existing(NETWORK, cls.name, size=12, color='purple', name=cls.name, typ='class')
        NETWORK.add_edge(cls.name, node.file, weight=1/4)
        for method in cls.methods:
            try:
                add_node_if_not_existing(NETWORK, method.name, size=5, color='blue', name=method.name, typ='method')
                NETWORK.add_edge(cls.name, method.name, weight=1/2.5)
                if children:
                    [(add_node_if_not_existing(NETWORK, child, size=5, color='yellow', name=child, typ='child'), NETWORK.add_edge(method.name, child, weight=1/1)) 
                    for child in method.children]
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
    print(str(P(path).resolve()))
    sys.path.append(str(P(path).resolve()))
    print(sys.path)
    yield
    os.chdir(prev_cwd)

def main(folder):
    if folder.endswith('/'): folder = folder[:-1]
    with working_directory(folder):
        NETWORK = nx.Graph()
        summarize(stem(folder), NETWORK)

    # find('layout', dir(nx))
    pos_ = nx.kamada_kawai_layout(NETWORK)
    # pos_ = nx.rescale_layout(NETWORK)

    plot = Plot(
        plot_width=1024,
        plot_height=1024,
        x_range=Range1d(-1.1,1.1),
        y_range=Range1d(-1.1,1.1)
    )

    plot.add_tools(
        HoverTool(tooltips=None),
        TapTool(),
        BoxSelectTool(),
        PanTool(), 
        WheelZoomTool(), 
        ResetTool()
    )
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)

    graph_renderer = from_networkx(
        NETWORK,
        pos_,
        scale=1,
        center=(0,0),
    )

    graph_renderer.node_renderer.data_source.data['size'] = [i*2 for i in list(nx.get_node_attributes(NETWORK, 'size').values())]
    graph_renderer.node_renderer.data_source.data['color'] = [i for i in list(nx.get_node_attributes(NETWORK, 'color').values())]
    graph_renderer.node_renderer.glyph = Circle(
        size='size',
        fill_color='color'
    )

    graph_renderer.node_renderer.selection_glyph = Circle(
        size='size',
        fill_color='color'
    )
    graph_renderer.node_renderer.hover_glyph = Circle(
        size='size',
        fill_color='color'
    )

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#CCCCCC",
        line_alpha=0.8,
        line_width=1
    )
    graph_renderer.edge_renderer.selection_glyph = MultiLine(
        line_color=(0,0,0),
        line_width=3
    )
    graph_renderer.edge_renderer.hover_glyph = MultiLine(
        line_color=(0,0,0),
        line_width=3
    )

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)
    x, y = zip(*list(pos_.values()))
    node_labels = nx.get_node_attributes(NETWORK, 'name')
    types = nx.get_node_attributes(NETWORK, 'typ')

    source = pd.DataFrame({'x': x, 'y': y, 'type': [types[i] for i in types.keys()],
                           'name': [node_labels[i] for i in node_labels.keys()]})
    # source = source[source['type'].map(lambda x: x in ['file','class'])].reset_index(drop=True)
    source.to_dict(orient='list')
    source = ColumnDataSource(source)
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_alpha=0, text_font_size='10px')

    plot.renderers.append(labels)
    output_file(f"{folder}.html")
    show(plot)
