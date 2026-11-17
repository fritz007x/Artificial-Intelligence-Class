"""Builds notebooks/02_simulation.ipynb (code + markdown) so the notebook stays in sync
with the source. Execute it with:
    python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/02_simulation.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# 02 · Core simulation mapping: OrchidShield triage agent

This notebook covers Phase 1, Step 2. It builds a **synthetic** SOC environment, stabilises the agent's outputs
with fixed seeds, and benchmarks the algorithm families the agent uses. Each section states a **hypothesis**,
measures it, and notes the **failure cases** found. All numbers can be reproduced: run top to bottom and the
same seed gives the same results, which are written to `data/results/simulation_metrics.json`.
""")
code("""
import sys, json, time, random
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "configs").exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from core.config import load_settings
S = load_settings(); SEED = S.app.seed
random.seed(SEED); np.random.seed(SEED)
RESULTS = {"seed": SEED, **S.run_metadata()}
print(S.run_metadata())
""")
md("""
## 1. Environment and PEAS recap
Assets, users and indicators are generated deterministically. The environment is **partially observable**
(the true compromise state is hidden and some hosts have no EDR), **stochastic**, **sequential**, **dynamic**
(the attacker moves), **discrete** at the decision level, and **multi-agent** (attacker vs. defender).
""")
code("""
from sim.generator import make_environment
env = make_environment(SEED)
A = env["assets"]
pd.DataFrame(A.values())[["asset_id","asset_type","criticality","zone","edr_installed","in_maintenance"]]
""")
md("""
## 2. Problem formulation and uninformed vs. informed search
**State:** the host the attacker controls. **Initial state:** a compromised workstation. **Goal:** any crown jewel
(criticality ≥ 4). **Actions:** traverse a network edge. **Cost:** exploitation difficulty (crossing zones costs more).

*Hypothesis H1:* A* with the hop heuristic finds the same optimal cost as UCS but expands fewer nodes. BFS and DFS
are not cost-optimal.
""")
code("""
from reasoning import search
G = search.build_graph(A)
crown = {a for a,v in A.items() if v["criticality"] >= 4}
rows = []
for start in ["ws-001","ws-003","ws-006","ws-008","ws-011","web-01","file-01"]:
    goal = lambda n, s=start: n in crown and n != s
    h = search.hop_heuristic(G, crown - {start})
    for res in [search.bfs(G,start,goal), search.dfs(G,start,goal), search.iddfs(G,start,goal),
                search.ucs(G,start,goal), search.greedy(G,start,goal,h), search.astar(G,start,goal,h)]:
        rows.append({"start": start, "algo": res.algorithm, "cost": res.cost, "expanded": res.expanded,
                     "path": " → ".join(res.path)})
df = pd.DataFrame(rows)
summary = df.groupby("algo")[["cost","expanded"]].mean().round(2)
opt = df[df.algo=="UCS"].set_index("start").cost
summary["optimal_rate"] = df.groupby("algo").apply(lambda g: float((g.set_index("start").cost <= opt + 1e-9).mean()))
RESULTS["search"] = summary.reset_index().to_dict(orient="records")
summary
""")
code("""
# heuristic admissibility / consistency check over every node
h = search.hop_heuristic(G, crown)
adm = all(h(n) <= min((search.dijkstra_all(G,n).get(g, 1e9) for g in crown), default=1e9) + 1e-9 for n in G)
con = all(h(n) <= c + h(m) + 1e-9 for n in G for m, c in G[n])
RESULTS["heuristic"] = {"admissible": adm, "consistent": con}
print("admissible:", adm, " consistent:", con)
""")
md("""
## 3. Optimisation: which hosts to isolate (local search)
Choose at most *k* hosts to isolate so that the criticality reachable from a compromised host is minimised, plus a
disruption penalty. Domain controllers and the SIEM are forbidden.

*Hypothesis H2:* hill climbing with restarts and simulated annealing reach the exhaustive optimum on this graph
with far fewer evaluations.
""")
code("""
rows = []
for comp in ["ws-004","ws-008","web-01"]:
    p = search.SegmentationProblem(A, {comp}, k=2, forbidden={"dc-01","siem-01",comp})
    t=time.perf_counter(); ex=search.exhaustive(p); te=time.perf_counter()-t
    hc=search.hill_climb(p, random.Random(SEED)); sa=search.simulated_annealing(p, random.Random(SEED))
    rows.append({"compromised": comp, "optimum": ex[1], "opt_set": sorted(ex[0]), "exhaustive_evals": ex[2],
                 "hill_climb": hc[1], "hc_evals": hc[2], "annealing": sa[1], "sa_evals": sa[2]})
opt_df = pd.DataFrame(rows); RESULTS["local_search"] = rows; opt_df
""")
md("""
**Failure case found:** the landscape is full of plateaus, because isolating a single host rarely reduces reach until
the cut is complete. With 5 restarts, hill climbing scored 21.5 against an optimum of 3.5, and simulated annealing
with 400 steps did the same. Raising the defaults to 20 restarts and 1,500 steps fixed both (see `reasoning/search.py`).

## 4. Adversarial search: defender vs. attacker
MAX (defender) isolates one host per turn, and MIN (attacker) spreads to an adjacent host. The utility is the
negative squared criticality of the hosts lost, minus the cost of isolation.

*Hypothesis H3:* alpha-beta returns the same value as minimax with fewer nodes. Expectimax (with a random attacker)
is never worse for the defender than minimax.
""")
code("""
from reasoning.adversarial import ContainmentGame
rows=[]
for comp in [{"ws-004"},{"dc-01"},{"srv-app-03"}]:
    for depth in (2,3,4):
        r={"compromised":",".join(comp),"depth":depth}
        for m in ("minimax","alphabeta","expectimax"):
            g=ContainmentGame(A,{"dc-01","siem-01"}); t=time.perf_counter()
            mv,val,nodes=g.best_isolation(comp,depth,m)
            r[f"{m}_move"]=mv; r[f"{m}_value"]=round(val,2); r[f"{m}_nodes"]=nodes
        rows.append(r)
adv=pd.DataFrame(rows); RESULTS["adversarial"]=rows
adv[["compromised","depth","minimax_value","alphabeta_value","minimax_nodes","alphabeta_nodes","alphabeta_move","expectimax_value"]]
""")
md("""
## 5. CSP: scheduling response tasks
Variables are response tasks. Domains are (analyst, slot) pairs. The constraints are skills, seniority for
approval-gated work, one task per analyst per slot, precedence, and separation of duties.

*Hypothesis H4:* MRV+LCV with forward checking or MAC needs far fewer assignments than plain backtracking, and
min-conflicts solves it quickly.
""")
code("""
from reasoning import csp
analysts=[a.model_dump() for a in S.csp.analysts]
def problem(n_inc, n_slots):
    T=[]
    for i in range(n_inc):
        T += [csp.Task(f"triage{i}","triage"), csp.Task(f"enrich{i}","endpoint"),
              csp.Task(f"isolate{i}","endpoint",True,after=[f"enrich{i}"],separate_from=[f"triage{i}"]),
              csp.Task(f"disable{i}","identity",True,after=[f"triage{i}"])]
    # dependants listed first = a deliberately poor static variable order for plain backtracking
    return csp.build_schedule_csp(T[::-1], analysts, list(range(n_slots)))
rows=[]
for n, slots in [(2,3),(3,5),(4,6),(3,4)]:      # last instance is over-constrained (unsatisfiable)
    for name,kw in [("backtracking",{}),("MRV+LCV",{"mrv":True,"lcv":True}),("MRV+FC",{"mrv":True,"forward_check":True}),("MAC(AC-3)",{"mrv":True,"lcv":True,"mac":True})]:
        c=problem(n,slots); t=time.perf_counter(); sol=csp.backtrack(c,**kw)
        rows.append({"incidents":n,"slots":slots,"solver":name,"solved":bool(sol and csp.verify(c,sol)),"assignments":c.stats["assignments"],"backtracks":c.stats["backtracks"],"ms":round(1000*(time.perf_counter()-t),1)})
    c=problem(n,slots); t=time.perf_counter(); sol=csp.min_conflicts(c,random.Random(SEED))
    rows.append({"incidents":n,"slots":slots,"solver":"min-conflicts","solved":bool(sol and csp.verify(c,sol)),"assignments":c.stats["assignments"],"backtracks":0,"ms":round(1000*(time.perf_counter()-t),1)})
csp_df=pd.DataFrame(rows); RESULTS["csp"]=rows; csp_df
""")
md("""
## 6. Logic: propositional entailment (resolution vs. model checking)
*Hypothesis H5:* resolution refutation and truth-table model checking agree on every case in
`03_logic/entailment_tests.yaml`.
""")
code("""
import yaml
from reasoning.logic import parse_clause, pl_resolution, tt_entails, satisfiable
cases=yaml.safe_load((ROOT/"03_logic/entailment_tests.yaml").read_text())
cnf=[parse_clause(l) for l in (ROOT/"03_logic/cnf_rules.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
rows=[]
for c in cases["propositional"]:
    kb=cnf+[frozenset([f]) for f in c["facts"]]
    if c["query"] is None:
        rows.append({"id":c["id"],"expect":c["expect"],"resolution":None,"model_check":None,"satisfiable":satisfiable(kb)}); continue
    r,nr=pl_resolution(kb,c["query"]); t,nt=tt_entails(kb,c["query"])
    rows.append({"id":c["id"],"expect":c["expect"],"resolution":r,"resolvents":nr,"model_check":t,"models":nt})
logic_df=pd.DataFrame(rows); RESULTS["propositional"]=rows; logic_df
""")
md("""
## 7. First-order logic: the agent's knowledge base and four-way entailment
""")
code("""
from agent.knowledge import KnowledgeBase
from agent.entailment import EntailmentChecker
rows=[]
for c in cases["fol"]:
    kb=KnowledgeBase(S,{"assets":{},"users":{},"indicators":[]})
    for f in c["facts"]: kb.tell(f)
    r=EntailmentChecker(S,kb).check(c["query"])
    rows.append({"id":c["id"],"query":c["query"],"expected":c["expect"],"got":r.status.value,"proof_steps":len(r.proof)})
fol_df=pd.DataFrame(rows); RESULTS["fol"]=rows; fol_df
""")
md("""
## 8. Bayesian network: posterior under evidence, explaining away, missing data
""")
code("""
from reasoning.bayes import BayesNet
bn=BayesNet.from_json(ROOT/"04_uncertainty/bayes_net.json")
ev_sets={"no evidence":{}, "IOC":{"IOCMatch":True}, "IOC+EDR":{"IOCMatch":True,"EDRDetection":True},
 "AuthAnomaly":{"AuthAnomaly":True,"Maintenance":False}, "AuthAnomaly in maintenance":{"AuthAnomaly":True,"Maintenance":True},
 "IOC + analyst FP":{"IOCMatch":True,"AnalystFP":True}, "Auth anomaly, EDR missing":{"AuthAnomaly":True,"EDRDetection":None,"LateralSignal":False}}
bn_df=pd.DataFrame([{"evidence":k,"P(Compromise)":round(bn.posterior("Compromise",v),3)} for k,v in ev_sets.items()])
RESULTS["bayes"]=bn_df.to_dict(orient="records"); bn_df
""")
code("""
# calibration of the BN against the generative model: sample worlds from the BN itself, compare posterior to frequency
rng=np.random.default_rng(SEED); samples=[]
for _ in range(20000):
    a={}
    for n in bn.nodes: a[n.name]=rng.random()<n.p_true(a)
    samples.append(a)
ev={"IOCMatch":True,"AuthAnomaly":True}
match=[s for s in samples if all(s[k]==v for k,v in ev.items())]
emp=np.mean([s["Compromise"] for s in match]); RESULTS["bayes_check"]={"exact":bn.posterior("Compromise",ev),"sampled":float(emp),"n":len(match)}
RESULTS["bayes_check"]
""")
md("""
## 9. Temporal reasoning: HMM filtering, Viterbi decoding, and Kalman burst detection
*Hypothesis H6:* Viterbi recovers the hidden kill-chain stage far better than a "most common observation" baseline,
and the Kalman innovation flags bursts with few false alarms.
""")
code("""
from reasoning.temporal import HMM, kalman_rate
hmm=HMM.from_json(ROOT/"04_uncertainty/hmm.json"); rng=np.random.default_rng(SEED)
obs_to_state={"quiet":"benign","scan":"recon","auth_anomaly":"initial_access","malware":"initial_access","lateral":"lateral_movement","exfil":"exfiltration"}
acc_v,acc_b,acc_f=[],[],[]
for _ in range(300):
    xs,os_=hmm.sample(30,rng)
    acc_v.append(np.mean([a==b for a,b in zip(xs,hmm.viterbi(os_))]))
    acc_b.append(np.mean([a==obs_to_state[o] for a,o in zip(xs,os_)]))
    acc_f.append(np.mean([xs[t]==hmm.states[int(hmm.filter(os_[:t+1]).argmax())] for t in range(len(os_))]))
RESULTS["hmm"]={"viterbi_acc":float(np.mean(acc_v)),"filter_acc":float(np.mean(acc_f)),"naive_baseline_acc":float(np.mean(acc_b))}
RESULTS["hmm"]
""")
code("""
fa=det=0; trials=500
for _ in range(trials):
    base=rng.poisson(0.4,20).astype(float)
    if kalman_rate(list(base)).max_z>S.thresholds.kalman_anomaly_sigma: fa+=1
    b=base.copy(); b[15]+=6
    if kalman_rate(list(b)).max_z>S.thresholds.kalman_anomaly_sigma: det+=1
RESULTS["kalman"]={"false_alarm_rate":fa/trials,"detection_rate":det/trials}
RESULTS["kalman"]
""")
md("""
## 10. Machine learning, neural network and RL (from `05_ml/train.py`)
The training pipeline is a script so that CI can reproduce it. Here we load its outputs.
""")
code("""
M=json.loads((ROOT/"05_ml/artifacts/metrics.json").read_text())
ml_df=pd.DataFrame(M["test"]).T[["roc_auc","f1","precision","recall","fpr","ece"]].round(3)
RESULTS["ml_selected"]=M["selected_model"]; RESULTS["learning_case"]=M["learning_case"]
print("selected:",M["selected_model"]); ml_df
""")
code("""
pd.DataFrame({"q_learning":M["rl"]["q_learning"],"rule_policy":M["rl"]["rule_policy"]}).round(3)
""")
md("""
## 11. End-to-end: the six demonstration scenarios
""")
code("""
from agent.orchestrator import Agent
from sim.scenarios import load_scenario
import tempfile
s2=load_settings(overrides={"paths.runtime_dir":tempfile.mkdtemp()})
agent=Agent(s2); rows=[]
for name in ["normal","uncertain","contradiction","constraint","adversarial"]:
    sc=load_scenario(name); r=agent.run_case(sc["observations"],scenario=name,action_requests=sc.get("action_requests"))
    top=r.inference.risks[0] if r.inference.risks else None
    rows.append({"scenario":name,"iterations":r.iterations,"top_asset":top.asset_id if top else None,
                 "P(compromise)":round(top.p_compromise_bayes,2) if top else None,
                 "entailed":sum(e.status.value=="entailed" for e in r.entailments),
                 "contradicted":sum(e.status.value=="contradicted" for e in r.entailments),
                 "pending_approval":[f"{p.action.value}({p.target})" for p in r.proposals if p.status.value=="pending_approval"],
                 "blocked":[f"{p.tool}({p.target})" for p in r.proposals if p.status.value=="blocked"]})
e2e=pd.DataFrame(rows); RESULTS["scenarios"]=rows; e2e
""")
md("""
## 12. Hypotheses, results and failure cases
| # | Hypothesis | Result |
|---|---|---|
| H1 | A* is cost-optimal and expands fewer nodes than UCS | See §2 table (optimal_rate = 1.0 for UCS/A*) |
| H2 | Local search reaches the exhaustive optimum | Yes after raising restarts (failure case documented in §3) |
| H3 | Alpha-beta = minimax value with fewer nodes | See §4 node counts |
| H4 | MRV/LCV/FC/MAC beat plain backtracking | See §5. MAC needs the fewest assignments and proves the over-constrained instance unsatisfiable fastest. MRV+LCV without propagation can be *worse* (4 incidents), which is a useful counter-example |
| H5 | Resolution ≡ model checking | See §6 |
| H6 | Viterbi ≫ naive stage guess; Kalman low false-alarm | See §9 |

**Failure cases recorded**
1. Hill-climbing plateau (§3), fixed with more restarts.
2. `ipaddress.is_private` treats the RFC 5737 documentation ranges as private, so `BlockWarranted` was never derived.
   Fixed by an explicit list of internal networks (`agent/knowledge.py`).
3. Rank-based ROC-AUC without tie handling scored a constant predictor 1.0. Fixed by averaging ranks.
4. Alpha-beta move ordering broke ties between equally critical hosts by set iteration order, which changes with
   `PYTHONHASHSEED`. The segmentation point flipped between `jump-01` and `srv-app-01` from run to run. Fixed by
   breaking ties on the host name.
5. The HMM labels a single auth anomaly as "benign" stage. This is expected with a strong benign prior, so stage
   evidence is only one input to fusion and never a trigger.
""")
code("""
out=ROOT/"data/results/simulation_metrics.json"; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(RESULTS,indent=1,default=str)); print("wrote",out.relative_to(ROOT))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                "language": "python"}})
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "02_simulation.ipynb")
print("notebook written")
