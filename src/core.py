"""
Base Node graph primitives and the Pipeline context manager.
"""


class Node:
    """A unit of work in the graph. Connect nodes with `a >> b`."""

    _current_pipeline = None  # set while inside an active `with Pipeline():` block

    def __init__(self):
        self.inputs = []
        self.outputs = []
        if Node._current_pipeline is not None:
            Node._current_pipeline._register(self)

    def __rshift__(self, other):
        """`a >> b` marks `a` as feeding into `b` and returns `b` for chaining."""
        other.inputs.append(self)
        self.outputs.append(other)
        return other

    def update(self, dt):
        """Called once per frame, in dependency order. Override to do work."""


class Pipeline:
    """Collects every `Node` created inside the `with` block and ticks them each frame."""

    def __init__(self):
        self.nodes = []

    def _register(self, node):
        self.nodes.append(node)

    def __enter__(self):
        Node._current_pipeline = self
        return self

    def __exit__(self, exc_type, exc, tb):
        Node._current_pipeline = None
        if exc_type is None:
            self.nodes = _topological_sort(self.nodes)
        return False

    def tick(self, dt):
        """Update every node once, in topological order."""
        for node in self.nodes:
            node.update(dt)


def _topological_sort(nodes):
    """Kahn's algorithm: order nodes so each node's inputs update before it does."""
    in_degree = {node: len(node.inputs) for node in nodes}
    ready = [node for node in nodes if in_degree[node] == 0]
    ordered = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for downstream in node.outputs:
            if downstream not in in_degree:
                continue
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                ready.append(downstream)
    if len(ordered) != len(nodes):
        raise RuntimeError("Pipeline graph has a cycle")
    return ordered
