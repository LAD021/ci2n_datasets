import time
from dataclasses import dataclass
from functools import lru_cache as cache

import networkx as nx


def my_graph_edit_distance(G1, G2, node_match, edge_match, timeout):

    for _, _, cost in my_optimize_edit_paths(
            G1,
            G2,
            node_match,
            edge_match,
            timeout=timeout,
    ):
        yield cost


def my_optimize_edit_paths(
        G1,
        G2,
        node_match=None,
        edge_match=None,
        node_subst_cost=None,
        node_del_cost=None,
        node_ins_cost=None,
        edge_subst_cost=None,
        edge_del_cost=None,
        edge_ins_cost=None,
        upper_bound=None,
        strictly_decreasing=True,
        roots=None,
        timeout=None,
):
    """GED (graph edit distance) calculation: advanced interface.

    Graph edit path is a sequence of node and edge edit operations
    transforming graph G1 to graph isomorphic to G2.  Edit operations
    include substitutions, deletions, and insertions.

    Graph edit distance is defined as minimum cost of edit path.

    Parameters
    ----------
    G1, G2: graphs
        The two graphs G1 and G2 must be of the same type.

    node_match : callable
        A function that returns True if node n1 in G1 and n2 in G2
        should be considered equal during matching.

        The function will be called like

           node_match(G1.nodes[n1], G2.nodes[n2]).

        That is, the function will receive the node attribute
        dictionaries for n1 and n2 as inputs.

        Ignored if node_subst_cost is specified.  If neither
        node_match nor node_subst_cost are specified then node
        attributes are not considered.

    edge_match : callable
        A function that returns True if the edge attribute dictionaries
        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should
        be considered equal during matching.

        The function will be called like

           edge_match(G1[u1][v1], G2[u2][v2]).

        That is, the function will receive the edge attribute
        dictionaries of the edges under consideration.

        Ignored if edge_subst_cost is specified.  If neither
        edge_match nor edge_subst_cost are specified then edge
        attributes are not considered.

    node_subst_cost, node_del_cost, node_ins_cost : callable
        Functions that return the costs of node substitution, node
        deletion, and node insertion, respectively.

        The functions will be called like

           node_subst_cost(G1.nodes[n1], G2.nodes[n2]),
           node_del_cost(G1.nodes[n1]),
           node_ins_cost(G2.nodes[n2]).

        That is, the functions will receive the node attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function node_subst_cost overrides node_match if specified.
        If neither node_match nor node_subst_cost are specified then
        default node substitution cost of 0 is used (node attributes
        are not considered during matching).

        If node_del_cost is not specified then default node deletion
        cost of 1 is used.  If node_ins_cost is not specified then
        default node insertion cost of 1 is used.

    edge_subst_cost, edge_del_cost, edge_ins_cost : callable
        Functions that return the costs of edge substitution, edge
        deletion, and edge insertion, respectively.

        The functions will be called like

           edge_subst_cost(G1[u1][v1], G2[u2][v2]),
           edge_del_cost(G1[u1][v1]),
           edge_ins_cost(G2[u2][v2]).

        That is, the functions will receive the edge attribute
        dictionaries as inputs.  The functions are expected to return
        positive numeric values.

        Function edge_subst_cost overrides edge_match if specified.
        If neither edge_match nor edge_subst_cost are specified then
        default edge substitution cost of 0 is used (edge attributes
        are not considered during matching).

        If edge_del_cost is not specified then default edge deletion
        cost of 1 is used.  If edge_ins_cost is not specified then
        default edge insertion cost of 1 is used.

    upper_bound : numeric
        Maximum edit distance to consider.

    strictly_decreasing : bool
        If True, return consecutive approximations of strictly
        decreasing cost.  Otherwise, return all edit paths of cost
        less than or equal to the previous minimum cost.

    roots : 2-tuple
        Tuple where first element is a node in G1 and the second
        is a node in G2.
        These nodes are forced to be matched in the comparison to
        allow comparison between rooted graphs.

    timeout : numeric
        Maximum number of seconds to execute.
        After timeout is met, the current best GED is returned.

    Returns
    -------
    Generator of tuples (node_edit_path, edge_edit_path, cost)
        node_edit_path : list of tuples (u, v)
        edge_edit_path : list of tuples ((u1, v1), (u2, v2))
        cost : numeric

    See Also
    --------
    graph_edit_distance, optimize_graph_edit_distance, optimal_edit_paths

    References
    ----------
    .. [1] Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick
       Martineau. An Exact Graph Edit Distance Algorithm for Solving
       Pattern Recognition Problems. 4th International Conference on
       Pattern Recognition Applications and Methods 2015, Jan 2015,
       Lisbon, Portugal. 2015,
       <10.5220/0005209202710278>. <hal-01168816>
       https://hal.archives-ouvertes.fr/hal-01168816

    """
    # TODO: support DiGraph

    import numpy as np
    import scipy as sp

    @dataclass
    class CostMatrix:
        C: ...
        lsa_row_ind: ...
        lsa_col_ind: ...
        ls: ...


    def make_CostMatrix(C, m, n):
        # assert(C.shape == (m + n, m + n))
        lsa_row_ind, lsa_col_ind = sp.optimize.linear_sum_assignment(C)

        # Fixup dummy assignments:
        # each substitution i<->j should have dummy assignment m+j<->n+i
        # NOTE: fast reduce of Cv relies on it
        # Create masks for substitution and dummy indices
        is_subst = (lsa_row_ind < m) & (lsa_col_ind < n)
        is_dummy = (lsa_row_ind >= m) & (lsa_col_ind >= n)

        # Map dummy assignments to the correct indices
        lsa_row_ind[is_dummy] = lsa_col_ind[is_subst] + m
        lsa_col_ind[is_dummy] = lsa_row_ind[is_subst] + n

        return CostMatrix(
            C, lsa_row_ind, lsa_col_ind, C[lsa_row_ind, lsa_col_ind].sum()
        )


    def extract_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))
        row_ind = [k in i or k - m in j for k in range(m + n)]
        col_ind = [k in j or k - n in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]


    def reduce_C(C, i, j, m, n):
        # assert(C.shape == (m + n, m + n))

        col_ind, row_ind = cached_ind(i, j, m, n)
        return C[row_ind, :][:, col_ind]

    @cache
    def cached_ind(i, j, m, n):
        row_ind = [k not in i and k - m not in j for k in range(m + n)]
        col_ind = [k not in j and k - n not in i for k in range(m + n)]
        return col_ind, row_ind

    def reduce_ind(ind, i):
        # assert set(ind) == set(range(len(ind)))
        rind = ind[[k not in i for k in ind]]
        for k in set(i):
            rind[rind >= k] -= 1
        return rind

    def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=None):
        """
        Parameters:
            u, v: matched vertices, u=None or v=None for
               deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_uv: partial vertex edit path
                list of tuples (u, v) of previously matched vertex
                    mappings u<->v, u=None or v=None for
                    deletion/insertion

        Returns:
            list of (i, j): indices of edge mappings g<->h
            localCe: local CostMatrix of edge mappings
                (basically submatrix of Ce at cross of rows i, cols j)
        """
        M = len(pending_g)
        N = len(pending_h)

        matched_set_p_p = {(p, p) for p, q in matched_uv}
        matched_set_q_q = {(q, q) for p, q in matched_uv}
        matched_set_p_u = {(p, u) for p, q in matched_uv}
        matched_set_u_p = {(u, p) for p, q in matched_uv}
        matched_set_q_v = {(q, v) for p, q in matched_uv}
        matched_set_v_q = {(v, q) for p, q in matched_uv}
        matched_set_u_u = {(u, u)}
        matched_set_v_v = {(v, v)}
        # matched_set_p_q = {(p, q) for p, q in matched_uv}
        # matched_set_q_p = {(q, p) for p, q in matched_uv}
        matched_set_p_u_q_v = {(p, u, q, v) for p, q in matched_uv}

        # assert Ce.C.shape == (M + N, M + N)

        # only attempt to match edges after one node match has been made
        # this will stop self-edges on the first node being automatically deleted
        # even when a substitution is the better option
        if matched_uv is None or len(matched_uv) == 0:
            g_ind = []
            h_ind = []
        else:

            matched_set_g = matched_set_p_u | matched_set_u_p | matched_set_p_p | matched_set_u_u
            matched_set_h = matched_set_q_v | matched_set_v_q | matched_set_q_q | matched_set_v_v

            g_ind = [
                i
                for i in range(M)
                if pending_g[i][:2] in matched_set_g
            ]
            h_ind = [
                j
                for j in range(N)
                if pending_h[j][:2] in matched_set_h
            ]
        m = len(g_ind)
        n = len(h_ind)

        if m or n:
            C = extract_C(Ce.C, g_ind, h_ind, M, N)
            # assert C.shape == (m + n, m + n)

            # Forbid structurally invalid matches
            # NOTE: inf remembered from Ce construction
            for k, i in enumerate(g_ind):
                g = pending_g[i][:2]
                for l, j in enumerate(h_ind):
                    h = pending_h[j][:2]
                    if nx.is_directed(G1) or nx.is_directed(G2):
                        if any(
                                g == (p, u) and h == (q, v) or g == (u, p) and h == (v, q)
                                for p, q in matched_uv
                        ):
                            continue
                    else:
                        g1, g2 = g
                        h1, h2 = h
                        any11 = (g1, g2, h1, h2) in matched_set_p_u_q_v
                        any12 = (g1, g2, h2, h1) in matched_set_p_u_q_v
                        any13 = (g2, g1, h1, h2) in matched_set_p_u_q_v
                        any14 = (g2, g1, h2, h1) in matched_set_p_u_q_v

                        any1 = any11 or any12 or any13 or any14
                        # any2 = any(
                        #         g in ((p, u), (u, p)) and h in ((q, v), (v, q))
                        #         for p, q in matched_uv
                        # )
                        # assert any1 == any2

                        if any1:
                            continue

                    any1 = (g in matched_set_u_u | matched_set_p_p)
                    # any2 = (g == (u, u) or any(g == (p, p) for p, q in matched_uv))
                    # assert any1 == any2
                    if any1:
                        continue

                    any1 = (h in matched_set_v_v | matched_set_q_q)
                    # any2 = (h == (v, v) or any(h == (q, q) for p, q in matched_uv))
                    # assert any1 == any2
                    if any1:
                        continue
                    C[k, l] = inf

            localCe = make_CostMatrix(C, m, n)
            ij = [
                (
                    g_ind[k] if k < m else M + h_ind[l],
                    h_ind[l] if l < n else N + g_ind[k],
                )
                for k, l in zip(localCe.lsa_row_ind, localCe.lsa_col_ind)
                if k < m or l < n
            ]

        else:
            ij = []
            localCe = CostMatrix(np.empty((0, 0)), [], [], 0)

        return ij, localCe

    def reduce_Ce(Ce, ij, m, n):
        if len(ij):
            i, j = zip(*ij)
            m_i = m - sum(1 for t in i if t < m)
            n_j = n - sum(1 for t in j if t < n)
            return make_CostMatrix(reduce_C(Ce.C, i, j, m, n), m_i, n_j)
        return Ce

    def get_edit_ops(
            matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of
                (i, j): indices of vertex mapping u<->v
                Cv_ij: reduced CostMatrix of pending vertex mappings
                    (basically Cv with row i, col j removed)
                list of (x, y): indices of edge mappings g<->h
                Ce_xy: reduced CostMatrix of pending edge mappings
                    (basically Ce with rows x, cols y removed)
                cost: total cost of edit operation
            NOTE: most promising ops first
        """
        m = len(pending_u)
        n = len(pending_v)
        # assert Cv.C.shape == (m + n, m + n)

        # 1) a vertex mapping from optimal linear sum assignment
        i, j = min(
            (k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind) if k < m or l < n
        )
        xy, localCe = match_edges(
            pending_u[i] if i < m else None,
            pending_v[j] if j < n else None,
            pending_g,
            pending_h,
            Ce,
            matched_uv,
        )
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
        # assert Ce.ls <= localCe.ls + Ce_xy.ls
        if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
            pass
        else:
            # get reduced Cv efficiently
            Cv_ij = CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                reduce_ind(Cv.lsa_row_ind, (i, m + j)),
                reduce_ind(Cv.lsa_col_ind, (j, n + i)),
                Cv.ls - Cv.C[i, j],
            )
            yield (i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls

        # 2) other candidates, sorted by lower-bound cost estimate
        other = []
        fixed_i, fixed_j = i, j
        if m <= n:
            candidates = (
                (t, fixed_j)
                for t in range(m + n)
                if t != fixed_i and (t < m or t == m + fixed_j)
            )
        else:
            candidates = (
                (fixed_i, t)
                for t in range(m + n)
                if t != fixed_j and (t < n or t == n + fixed_i)
            )
        for i, j in candidates:
            if prune(matched_cost + Cv.C[i, j] + Ce.ls):
                continue
            Cv_ij = make_CostMatrix(
                reduce_C(Cv.C, (i,), (j,), m, n),
                m - 1 if i < m else m,
                n - 1 if j < n else n,
            )
            # assert Cv.ls <= Cv.C[i, j] + Cv_ij.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + Ce.ls):
                continue
            xy, localCe = match_edges(
                pending_u[i] if i < m else None,
                pending_v[j] if j < n else None,
                pending_g,
                pending_h,
                Ce,
                matched_uv,
            )
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls):
                continue
            Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
            # assert Ce.ls <= localCe.ls + Ce_xy.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls + Ce_xy.ls):
                continue
            other.append(((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls))

        yield from sorted(other, key=lambda t: t[4] + t[1].ls + t[3].ls)

    def get_edit_paths(
            matched_uv,
            pending_u,
            pending_v,
            Cv,
            matched_gh,
            pending_g,
            pending_h,
            Ce,
            matched_cost,
    ):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            matched_gh: partial edge edit path
                list of tuples (g, h) of edge mappings g<->h,
                g=None or h=None for deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path

        Returns:
            sequence of (vertex_path, edge_path, cost)
                vertex_path: complete vertex edit path
                    list of tuples (u, v) of vertex mappings u<->v,
                    u=None or v=None for deletion/insertion
                edge_path: complete edge edit path
                    list of tuples (g, h) of edge mappings g<->h,
                    g=None or h=None for deletion/insertion
                cost: total cost of edit path
            NOTE: path costs are non-increasing
        """
        # debug_print('matched-uv:', matched_uv)
        # debug_print('matched-gh:', matched_gh)
        # debug_print('matched-cost:', matched_cost)
        # debug_print('pending-u:', pending_u)
        # debug_print('pending-v:', pending_v)
        # debug_print(Cv.C)
        # assert list(sorted(G1.nodes)) == list(sorted(list(u for u, v in matched_uv if u is not None) + pending_u))
        # assert list(sorted(G2.nodes)) == list(sorted(list(v for u, v in matched_uv if v is not None) + pending_v))
        # debug_print('pending-g:', pending_g)
        # debug_print('pending-h:', pending_h)
        # debug_print(Ce.C)
        # assert list(sorted(G1.edges)) == list(sorted(list(g for g, h in matched_gh if g is not None) + pending_g))
        # assert list(sorted(G2.edges)) == list(sorted(list(h for g, h in matched_gh if h is not None) + pending_h))
        # debug_print()

        if prune(matched_cost + Cv.ls + Ce.ls):
            return

        if not max(len(pending_u), len(pending_v)):
            # assert not len(pending_g)
            # assert not len(pending_h)
            # path completed!
            # assert matched_cost <= maxcost_value
            nonlocal maxcost_value
            maxcost_value = min(maxcost_value, matched_cost)
            yield matched_uv, matched_gh, matched_cost

        else:
            edit_ops = get_edit_ops(
                matched_uv,
                pending_u,
                pending_v,
                Cv,
                pending_g,
                pending_h,
                Ce,
                matched_cost,
            )
            for ij, Cv_ij, xy, Ce_xy, edit_cost in edit_ops:
                i, j = ij
                # assert Cv.C[i, j] + sum(Ce.C[t] for t in xy) == edit_cost
                if prune(matched_cost + edit_cost + Cv_ij.ls + Ce_xy.ls):
                    continue

                # dive deeper
                u = pending_u.pop(i) if i < len(pending_u) else None
                v = pending_v.pop(j) if j < len(pending_v) else None
                matched_uv.append((u, v))
                for x, y in xy:
                    len_g = len(pending_g)
                    len_h = len(pending_h)
                    matched_gh.append(
                        (
                            pending_g[x] if x < len_g else None,
                            pending_h[y] if y < len_h else None,
                        )
                    )
                sortedx = sorted(x for x, y in xy)
                sortedy = sorted(y for x, y in xy)
                G = [
                    (pending_g.pop(x) if x < len(pending_g) else None)
                    for x in reversed(sortedx)
                ]
                H = [
                    (pending_h.pop(y) if y < len(pending_h) else None)
                    for y in reversed(sortedy)
                ]

                yield from get_edit_paths(
                    matched_uv,
                    pending_u,
                    pending_v,
                    Cv_ij,
                    matched_gh,
                    pending_g,
                    pending_h,
                    Ce_xy,
                    matched_cost + edit_cost,
                )

                # backtrack
                if u is not None:
                    pending_u.insert(i, u)
                if v is not None:
                    pending_v.insert(j, v)
                matched_uv.pop()
                for x, g in zip(sortedx, reversed(G)):
                    if g is not None:
                        pending_g.insert(x, g)
                for y, h in zip(sortedy, reversed(H)):
                    if h is not None:
                        pending_h.insert(y, h)
                for _ in xy:
                    matched_gh.pop()

    # Initialization

    pending_u = list(G1.nodes)
    pending_v = list(G2.nodes)

    initial_cost = 0
    if roots:
        root_u, root_v = roots
        if root_u not in pending_u or root_v not in pending_v:
            raise nx.NodeNotFound("Root node not in graph.")

        # remove roots from pending
        pending_u.remove(root_u)
        pending_v.remove(root_v)

    # cost matrix of vertex mappings
    m = len(pending_u)
    n = len(pending_v)
    C = np.zeros((m + n, m + n))
    if node_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                node_subst_cost(G1.nodes[u], G2.nodes[v])
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = node_subst_cost(G1.nodes[root_u], G2.nodes[root_v])
    elif node_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(node_match(G1.nodes[u], G2.nodes[v]))
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)
        if roots:
            initial_cost = 1 - node_match(G1.nodes[root_u], G2.nodes[root_v])
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if node_del_cost:
        del_costs = [node_del_cost(G1.nodes[u]) for u in pending_u]
    else:
        del_costs = [1] * len(pending_u)
    # assert not m or min(del_costs) >= 0
    if node_ins_cost:
        ins_costs = [node_ins_cost(G2.nodes[v]) for v in pending_v]
    else:
        ins_costs = [1] * len(pending_v)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n: n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m: m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Cv = make_CostMatrix(C, m, n)
    # debug_print(f"Cv: {m} x {n}")
    # debug_print(Cv.C)

    pending_g = list(G1.edges)
    pending_h = list(G2.edges)

    # cost matrix of edge mappings
    m = len(pending_g)
    n = len(pending_h)
    C = np.zeros((m + n, m + n))
    if edge_subst_cost:
        C[0:m, 0:n] = np.array(
            [
                edge_subst_cost(G1.edges[g], G2.edges[h])
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    elif edge_match:
        C[0:m, 0:n] = np.array(
            [
                1 - int(edge_match(G1.edges[g], G2.edges[h]))
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)
    else:
        # all zeroes
        pass
    # assert not min(m, n) or C[0:m, 0:n].min() >= 0
    if edge_del_cost:
        del_costs = [edge_del_cost(G1.edges[g]) for g in pending_g]
    else:
        del_costs = [1] * len(pending_g)
    # assert not m or min(del_costs) >= 0
    if edge_ins_cost:
        ins_costs = [edge_ins_cost(G2.edges[h]) for h in pending_h]
    else:
        ins_costs = [1] * len(pending_h)
    # assert not n or min(ins_costs) >= 0
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n: n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m: m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Ce = make_CostMatrix(C, m, n)
    # debug_print(f'Ce: {m} x {n}')
    # debug_print(Ce.C)
    # debug_print()

    maxcost_value = Cv.C.sum() + Ce.C.sum() + 1

    if timeout is not None:
        if timeout <= 0:
            raise nx.NetworkXError("Timeout value must be greater than 0")
        start = time.perf_counter()

    def prune(cost):
        if timeout is not None:
            if time.perf_counter() - start > timeout:
                return True
        if upper_bound is not None:
            if cost > upper_bound:
                return True
        if cost > maxcost_value:
            return True
        if strictly_decreasing and cost >= maxcost_value:
            return True
        return False

    # Now go!

    done_uv = [] if roots is None else [roots]

    for vertex_path, edge_path, cost in get_edit_paths(
            done_uv, pending_u, pending_v, Cv, [], pending_g, pending_h, Ce, initial_cost
    ):
        # assert sorted(G1.nodes) == sorted(u for u, v in vertex_path if u is not None)
        # assert sorted(G2.nodes) == sorted(v for u, v in vertex_path if v is not None)
        # assert sorted(G1.edges) == sorted(g for g, h in edge_path if g is not None)
        # assert sorted(G2.edges) == sorted(h for g, h in edge_path if h is not None)
        # print(vertex_path, edge_path, cost, file = sys.stderr)
        # assert cost == maxcost_value
        yield list(vertex_path), list(edge_path), float(cost)