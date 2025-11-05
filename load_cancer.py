import numpy as np
import sys
from itertools import product
from causal_graphs.graph_real_world import load_graph_file

g = load_graph_file("causal_graphs/real_data/small_graphs/cancer.bif")
print(g)

# Monte Carlo estimates via ancestral sampling
N = 500_000
s = g.sample(batch_size=N, as_array=False)

p_c = (s['Cancer'] == 0).mean()
p_xpos = (s['Xray'] == 0).mean()
p_c_given_xpos = ((s['Cancer'] == 0) & (s['Xray'] == 0)).sum() / (s['Xray'] == 0).sum()
p_d_true = (s['Dyspnoea'] == 0).mean()
p_d_true_given_c_false = ((s['Dyspnoea'] == 0) & (s['Cancer'] == 1)).sum() / (s['Cancer'] == 1).sum()

print("\nMonte Carlo estimates:")
print("P(Cancer=True) ~", round(p_c, 6))
print("P(Xray=positive) ~", round(p_xpos, 6))
print("P(Cancer=True | Xray=positive) ~", round(p_c_given_xpos, 6))
print("P(Dyspnoea=True) ~", round(p_d_true, 6))
print("P(Dyspnoea=True | Cancer=False) ~", round(p_d_true_given_c_false, 6))


# Exact enumeration (no external dependencies)
def exact_marginals(graph):
    vars_list = graph.variables
    names = [v.name for v in vars_list]
    sizes = [v.prob_dist.num_categs for v in vars_list]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Precompute CPT accessors
    cpts = {}
    parents = {}
    for v in vars_list:
        pf = v.prob_dist.prob_func
        if hasattr(pf, "input_names"):
            pars = list(pf.input_names)
        else:
            pars = []
        parents[v.name] = pars
        cpts[v.name] = pf.val_grid

    def joint_prob(assign):
        p = 1.0
        for vi, v in enumerate(vars_list):
            vname = v.name
            pars = parents[vname]
            grid = cpts[vname]
            if len(pars) == 0:
                p *= float(grid[assign[vi]])
            else:
                idx = tuple(assign[name_to_idx[pn]] for pn in pars)
                p *= float(grid[idx][assign[vi]])
        return p

    # Accumulate marginals we care about
    p_c_true = 0.0
    p_x_pos = 0.0
    p_c_true_and_x_pos = 0.0
    p_d_true = 0.0
    p_d_true_and_c_false = 0.0
    p_c_false = 0.0

    for assign in product(*[range(s) for s in sizes]):
        jp = joint_prob(assign)
        c = assign[name_to_idx['Cancer']]
        x = assign[name_to_idx['Xray']]
        d = assign[name_to_idx['Dyspnoea']]
        if c == 0:
            p_c_true += jp
        else:
            p_c_false += jp
        if x == 0:
            p_x_pos += jp
            if c == 0:
                p_c_true_and_x_pos += jp
        if d == 0:
            p_d_true += jp
            if c == 1:
                p_d_true_and_c_false += jp

    return {
        "P(C=True)": p_c_true,
        "P(X=positive)": p_x_pos,
        "P(C=True|X=positive)": p_c_true_and_x_pos / p_x_pos,
        "P(D=True)": p_d_true,
        "P(D=True|C=False)": p_d_true_and_c_false / p_c_false,
    }


exact = exact_marginals(g)
print("\nExact (enumeration):")
for k, v in exact.items():
    print(k, "=", round(v, 6))


# Optional: exact inference via pgmpy if environment supports it
try:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "pgmpy >=0.1.x uses 'int|float' type hints requiring Python 3.10+. "
            "Upgrade Python or install an older pgmpy (e.g., 0.1.23)."
        )
    from pgmpy.readwrite import BIFReader
    from pgmpy.inference import VariableElimination

    reader = BIFReader("causal_graphs/real_data/small_graphs/cancer.bif")
    model = reader.get_model()
    infer = VariableElimination(model)

    print("\npgmpy (exact):")
    print("P(Cancer)", infer.query(variables=["Cancer"]).values)
    print(
        "P(Cancer|Xray=positive)",
        infer.query(variables=["Cancer"], evidence={"Xray": "positive"}).values,
    )
    print("P(Dyspnoea)", infer.query(variables=["Dyspnoea"]).values)
except Exception as e:
    print("\npgmpy exact inference skipped:", e)
