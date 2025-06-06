import torch
import asyncio
import logging
import numpy as np
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TokenByteTrie:
    """A trie data structure for efficient token-to-byte mapping."""

    def __init__(
        self, decode, device=None, atomic_tokens=None, eot_token=None, max_batch_size=64
    ):
        """Initialize a `TokenByteTrie`.

        Args:
            decode (list[bytes]): List representing the token vocabulary.
            device (str, optional): Device to use for weight sum and max computations ('cpu' or 'cuda').
            atomic_tokens (list[bytes], optional): List of tokens that should be treated as atomic units rather than being split into bytes.
            eot_token (bytes|None, optional): End-of-token token. Default is None, which represents EOT as None.
            max_batch_size (int, optional): Maximum batch size for weight sum sparse matrix multiplication.
        """
        self.decode = decode
        self.max_batch_size = max_batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda' or None")

        self.eot_token = eot_token
        self._build_trie(atomic_tokens or [])
        self._renumber()
        self._build_node2prefix()
        self._build_reachability_matrix()
        self.token_ids = torch.tensor(
            self.token_id_to_leaf[:, 0], dtype=torch.long, device=self.device
        )

    def _build_trie(self, atomic_tokens):
        """Builds a trie data structure from the vocabulary.

        Returns:
            (dict): A dictionary where keys are token IDs and values are lists of characters.
        """
        for token in atomic_tokens:
            if token not in self.decode:
                raise ValueError(f"Atomic token {token} not in vocabulary")

        self.word2leaf = {}
        self.children = [{}]  # First node is root
        self.root = 0
        self.token_id_to_leaf = []
        self.lookup = {}

        for token_id, word in enumerate(self.decode):
            if word in self.lookup:
                raise ValueError(f"Duplicate word in vocabulary: {word}")
            self.lookup[word] = token_id

            curr = self.root
            letters = [word] if word in atomic_tokens else word
            for letter in letters:
                if letter not in self.children[curr]:
                    self.children[curr][letter] = len(self.children)
                    self.children.append({})
                curr = self.children[curr][letter]

            self.children[curr][self.eot_token] = last = len(self.children)
            self.children.append({})
            assert word not in self.word2leaf
            self.word2leaf[word] = last
            self.token_id_to_leaf.append((token_id, last))

        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))
        self.jump = [
            np.array(sorted(x.values()), dtype=np.int32) for x in self.children
        ]

    def _renumber(self):
        """Renumber the states of the trie so that they are named by a contiguous
        range of integers and those integers respect the topological ordering
        of the trie. This improves the efficiency of the updating the trie as
        it improves memory locality.
        """
        self.ordering = np.array(list(self._order(self.root)), np.int32)
        ordering = {}
        for i, x in enumerate(self._order_full(self.root)):
            ordering[x] = i
        self._rename(f=lambda x: ordering[x])

    def _order(self, node):
        """Generate a topological ordering of nodes beneath the given node.

        Args:
            node (int): Starting node index

        Yields:
            int: Node indices in topological order
        """
        for a in self.children[node]:
            if a is not None:
                yield from self._order(self.children[node][a])
        yield node

    def _order_full(self, node):
        """Generate a complete topological ordering including all child nodes.

        Args:
            node (int): Starting node index

        Yields:
            (int): Node indices in complete topological order
        """
        for a in self.children[node]:
            yield from self._order_full(self.children[node][a])
        yield node

    def _rename(self, f):
        """Rename all node indices in the trie using the provided mapping function.

        Args:
            f (callable): Function that maps old node indices to new node indices
        """
        N = len(self.children)

        new_children = [{} for _ in range(N)]
        nodes = range(N)

        for x in nodes:
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        self.root = f(self.root)
        self.children = new_children
        self.word2leaf = {w: f(x) for w, x in self.word2leaf.items()}
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))

        self.token_id_to_leaf = np.array(
            [(i, f(x)) for i, x in self.token_id_to_leaf], dtype=np.int32
        )
        self.leaf2token_id = dict(
            zip(self.token_id_to_leaf[:, 1], self.token_id_to_leaf[:, 0])
        )

        self.ordering = np.array([f(x) for x in self.ordering])
        self.jump = [np.array(sorted(x.values()), dtype=np.int32) for x in new_children]

    def _build_node2prefix(self):
        """Builds a mapping from each node to its prefix.

        Returns:
            (dict): A dictionary where keys are node IDs and values are lists of characters.
        """
        node2prefix = {self.root: []}
        for x in reversed(range(len(self.children))):
            for letter, y in self.children[x].items():
                if letter is None:
                    node2prefix[y] = node2prefix[x]
                elif isinstance(letter, bytes):
                    node2prefix[y] = node2prefix[x] + list(letter)
                else:
                    node2prefix[y] = node2prefix[x] + [letter]

        self.node2prefix = node2prefix

    def _build_parent_map(self):
        """Builds a mapping from each node to its parent node in the trie.

        Returns:
            (dict): A dictionary where keys are child nodes and values are their parent nodes.
        """
        parent = {}
        for node in range(len(self.children)):
            for child in self.jump[node]:
                parent[child] = node
        return parent

    def _build_reachability_matrix(self):
        """Constructs a sparse reachability matrix for efficient weight propagation.

        The matrix M is constructed such that M[i,j] = 1 if node j is either:
        - The leaf node i itself (self-connection)
        - An ancestor of leaf node i in the trie
        """
        leaf_indices = self.token_id_to_leaf[:, 1]
        parent = self._build_parent_map()

        rows, cols = [], []
        for i, node in enumerate(leaf_indices):
            # self connections
            rows.append(i)
            cols.append(node)

            current = node
            while current in parent:  # Walk up to root
                ancestor = parent[current]
                rows.append(i)
                cols.append(ancestor)
                current = ancestor

        self.src_indices = torch.tensor(rows, dtype=torch.long, device=self.device)
        self.dst_indices = torch.tensor(cols, dtype=torch.long, device=self.device)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)

        self.M = torch.sparse_coo_tensor(
            indices, values, (len(leaf_indices), len(self.children))
        ).to_sparse_csr()

    def _preprocess_ws(self, batch_ws):
        """Preprocess weight sums for batch processing.

        Args:
            batch_ws (list|np.ndarray|torch.Tensor): List of weight sum tensors or lists of weight sums.

        Returns:
            (torch.Tensor): Stacked weight sum tensor.
        """
        processed_batch_ws = []
        for ws in batch_ws:
            if not isinstance(ws, torch.Tensor):
                ws = torch.tensor(ws, device=self.device, dtype=torch.float32)
            elif ws.device != self.device or ws.dtype != torch.float32:
                ws = ws.to(device=self.device, dtype=torch.float32)
            assert ws.shape[0] == len(self.decode), [ws.shape[0], len(self.decode)]
            processed_batch_ws.append(ws)
        return torch.stack(processed_batch_ws)

    def weight_sum(self, ws):
        """Computes the sum of weights of all leaf nodes (tokens) that are descendants of each node in the trie.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.decode)`,).

        Returns:
            (numpy.ndarray): Summed weights for each node in the trie, shape (num_nodes,).
        """
        return self.batch_weight_sum(self._preprocess_ws([ws]))[0]

    def batch_weight_sum(self, ws):
        """Batch version of `weight_sum`.

        Args:
            ws (torch.Tensor): Batch of token weights, shape (batch_size × `len(self.decode)`).

        Returns:
            (numpy.ndarray): Summed weights for each node in the trie, shape (batch_size × num_nodes).
        """
        ws = self._preprocess_ws(ws)
        batch_size = ws.shape[0]
        all_masses = []
        # If you are getting illegal memory access errors here,
        # try reducing the max_batch_size.
        for i in range(0, batch_size, self.max_batch_size):
            batch_ws = ws[i : i + self.max_batch_size]
            masses = torch.sparse.mm(batch_ws[:, self.token_ids], self.M)
            all_masses.append(masses)
        return torch.cat(all_masses, dim=0)

    def weight_max(self, ws):
        """Computes the maximum weight of all descendant leaf nodes (tokens) for each node in the trie.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.decode)`,).

        Returns:
            (numpy.ndarray): Maximum weights for each node in the trie, shape (num_nodes,).
        """
        return self.batch_weight_max(self._preprocess_ws([ws]))[0]

    def batch_weight_max(self, ws):
        """Batch version of `weight_max`.

        Args:
            ws (torch.Tensor): Batch of token weights, shape (batch_size × `len(self.decode)`).

        Returns:
            (numpy.ndarray): Maximum weights for each node in the trie, shape (batch_size × num_nodes).
        """
        ws = self._preprocess_ws(ws)

        # Get leaf weights
        leaf_weights = ws[:, self.token_ids]  # shape: (batch_size × num_leafs)
        batch_size = leaf_weights.shape[0]

        # Use scatter_reduce to propagate maximum values in parallel
        result = torch.zeros((batch_size, len(self.children)), device=self.device)
        result.scatter_reduce_(
            dim=1,
            index=self.dst_indices.expand(batch_size, -1),
            src=leaf_weights[:, self.src_indices],
            reduce="amax",
            include_self=False,
        )

        return result

    def visualize(self, ws=None):
        """Visualize the trie structure using Graphviz.

        Args:
            ws (np.ndarray|None): Optional weight vector to display at each node. Should be of length `len(self.children)`.

        Returns:
            (graphviz.Digraph): The generated graph object
        """
        try:
            import graphviz
        except ImportError:  # pragma: no cover
            raise ImportError(
                "Please install graphviz: pip install graphviz"
            )  # pragma: no cover

        if ws is not None and len(ws) != len(self.children):
            raise ValueError(
                f"Weight vector length ({len(ws)}) must match number of nodes ({len(self.children)})"
            )

        dot = graphviz.Digraph(comment="Token Character Trie")
        dot.attr(rankdir="LR")

        # Create a subgraph for the legend
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", fontsize="10")
            legend.attr("node", fontsize="7", width="0.1", height="0.1")

            # Example internal node
            legend.node(
                "legend_internal",
                "Internal Node ID\n'Prefix'\nWeight (if provided)",
                shape="circle",
            )

            # Example leaf node
            legend.node("legend_leaf", "Complete Token", shape="doublecircle")

            legend.edge(
                "legend_internal",
                "legend_leaf",
                label="Token item",
                fontsize="10",
            )

            # Align legend horizontally
            legend.attr(rankdir="TB")
            legend.attr(rank="same")

        # Add the main trie nodes and edges
        for node_id in range(len(self.children)):
            prefix = self.node2prefix[node_id]

            if ws is not None:
                label = f"{node_id}\n'{prefix}'\n{ws[node_id]:.4f}"
            else:
                label = f"{node_id}\n'{prefix}'"

            # Color nodes based on mass if provided
            if ws is not None:
                max_ws = ws.max()
                if max_ws > 0:
                    intensity = int(255 * (1 - ws[node_id] / max_ws))
                    color = f"#{intensity:02x}{255:02x}{intensity:02x}"
                else:
                    color = "#ffffff"  # white for zero mass
            else:
                color = "#ffffff"  # default white

            if node_id in self.leaf2word:
                dot.node(
                    str(node_id),
                    label,
                    shape="doublecircle",
                    style="filled",
                    fillcolor=color,
                )
            else:
                dot.node(
                    str(node_id), label, shape="circle", style="filled", fillcolor=color
                )

        for node_id, children in enumerate(self.children):
            for char, child_id in children.items():
                if char is not None:
                    edge_label = str(char)
                else:
                    edge_label = "End-of-Token"

                dot.edge(str(node_id), str(child_id), label=edge_label)

        return dot


class TrieOp(Enum):
    """Enumeration of supported trie operations."""

    SUM = "sum"
    MAX = "max"


class AsyncTokenByteTrie:
    """An asynchronous wrapper for TokenByteTrie implementations that provides automatic request batching."""

    def __init__(self, trie):
        """Initialize an `AsyncTokenByteTrie`.

        Args:
            trie (TokenByteTrie): The underlying `TokenByteTrie` instance
        """
        self.trie = trie
        self._queue = None
        self._task = None

    @classmethod
    def from_vocab(cls, vocab, **kwargs):
        """Creates an `AsyncTokenByteTrie` from a vocabulary.

        Args:
            vocab (list): The vocabulary over which the trie will be defined.
            **kwargs (dict): Additional arguments passed to the trie constructor

        Returns:
            (AsyncTokenByteTrie): The initialized asynchronous trie instance.
        """
        trie = TokenByteTrie(decode=vocab, **kwargs)
        return cls(trie)

    def _queue_request(self, request, op):
        if not self._task or self._task.done():
            self.start()

        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((request, future, op))
        return future

    async def weight_sum(self, ws):
        """Queue a `weight_sum` request. Multiple concurrent calls will be automatically batched
        together.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.trie.decode)`,).

        Returns:
            (np.ndarray): The calculated mass sums for the given distribution.
        """
        return await self._queue_request(ws, TrieOp.SUM)

    async def weight_max(self, ws):
        """Queue a `weight_max` request. Multiple concurrent calls will be automatically batched
        together.

        Args:
            ws (torch.Tensor): Token weights, shape (`len(self.trie.decode)`,).

        Returns:
            (np.ndarray): The calculated max weights for the given distribution.
        """
        return await self._queue_request(ws, TrieOp.MAX)

    def start(self):
        """Start the background processing task if not already running."""
        if not self._task or self._task.done():
            logger.debug("starting background loop")
            # Create a new queue so that it is bound to the current event loop
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._background_loop())

    async def _background_loop(self):
        """Background task that processes queued weight sum and max requests.

        Continuously monitors the queue for new requests and processes them in batches
        using the underlying trie implementation.

        Raises:
            (Exception): If any error occurs during processing, it is propagated to all
                         pending futures in the current batch.
        """
        while True:
            try:
                op_groups = defaultdict(list)

                request, future, op = await self._queue.get()
                op_groups[op].append((request, future))

                try:
                    while True:
                        request, future, op = self._queue.get_nowait()
                        op_groups[op].append((request, future))
                except asyncio.QueueEmpty:
                    pass

                for op, group in op_groups.items():
                    requests, futures = zip(*group)

                    if op == TrieOp.SUM:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"processing {len(requests)} sum requests")
                        results = self.trie.batch_weight_sum(requests)
                    elif op == TrieOp.MAX:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"processing {len(requests)} max requests")
                        results = self.trie.batch_weight_max(requests)
                    else:
                        raise ValueError(f"Unknown trie operation: {op}")

                    for future, result in zip(futures, results):
                        future.set_result(result)

            except Exception as e:
                for group in op_groups.values():
                    for _, future in group:
                        if not future.done():
                            future.set_exception(e)
                raise

    async def cleanup(self):
        """Async cleanup - preferred method"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def shutdown(self):
        """Stop the background processing task and cleanup resources."""
        if self._task is not None:
            try:
                self._task.cancel()
            except RuntimeError:
                # Ignore runtime errors that might occur if event loop is closed
                pass
            self._task = None

    def __del__(self):
        self.shutdown()
