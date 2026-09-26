#!/usr/bin/env python3
"""
Intervalos de confianca a 95% para todos os resultados multi-corrida da tese,
e tabelas do WESAD com PR-AUC.

    python3 confidence_intervals.py            # escreve results/ci_tables.tex

IC da media sobre seeds ou folds com a distribuicao t:
    media +/- t(0.975, n-1) * SD / sqrt(n)
Em GroupKFold e LOSO os folds sao tratados como amostras independentes, o que e
a aproximacao habitual (os conjuntos de treino sobrepoem-se).

Fontes:
  WESAD (corrida de 2026-09-26, com PR-AUC): results_wesad_pr_*.csv
  Combinado: results_pooled_baselines.csv, results_v5.csv, results_seq_*.csv,
             results_group_ABCD.csv, results_group_{summary,seq,both}.csv
  SWELL por sujeito: results_swell_group_*.csv + baselines calculadas aqui
             (mesma receita do wesad_baselines.py) -> results_swell_group_baselines.csv
  Transferencia: results_transfer.csv
  Ablacao F1/F2: results_wesad_F{1,2}_D.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

NAMES = {"Majority": "Majority classifier", "Dataset-only": "Dataset-only rule",
         "LogReg": "Logistic regression", "RandomForest": "Random forest",
         "A": "Model A (LSTM)", "B": "Model B (Deep LSTM)", "C": "Model C (CNN-LSTM)",
         "D": "Model D (CNN-Transformer)", "summary": "Model E (Summary MLP)",
         "seq": "Model F (Sequence CNN)", "both": "Model G (Dual-Input CNN)"}
ORDER = ["Majority", "Dataset-only", "LogReg", "RandomForest", "summary",
         "A", "B", "C", "D", "seq", "both"]
TRAINED = ["LogReg", "RandomForest", "summary", "A", "B", "C", "D", "seq", "both"]


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    if n < 2 or x.std(ddof=1) == 0:
        return m, m, m, n
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h, n


# ---------------------------------------------------------------- carregar
def wesad(cv):
    b = pd.read_csv("results_wesad_pr_baselines.csv")
    b = b[b.cv == cv]
    out = {m: b[b.model == m].sort_values("fold").reset_index(drop=True)
           for m in ["Majority", "LogReg", "RandomForest"]}
    ab = pd.read_csv(f"results_wesad_pr_{cv}_ABCD.csv")
    for m in "ABCD":
        out[m] = ab[ab.model == m].sort_values("seed").reset_index(drop=True)
    for a in ["summary", "seq", "both"]:
        out[a] = pd.read_csv(f"results_wesad_pr_{cv}_{a}.csv").sort_values("seed").reset_index(drop=True)
    return out


def pooled(cv):
    b = pd.read_csv("results_pooled_baselines.csv")
    b = b[b.cv == cv]
    out = {m: b[b.model == m].sort_values("fold").reset_index(drop=True)
           for m in ["Majority", "Dataset-only", "LogReg", "RandomForest"]}
    ab = pd.read_csv("results_v5.csv" if cv == "random" else "results_group_ABCD.csv")
    for m in "ABCD":
        out[m] = ab[ab.model == m].sort_values("seed").reset_index(drop=True)
    pre = "results_seq_" if cv == "random" else "results_group_"
    for a in ["summary", "seq", "both"]:
        out[a] = pd.read_csv(f"{pre}{a}.csv").sort_values("seed").reset_index(drop=True)
    # No combinado a classe maioritaria e stress: o majority tem especificidade 0.
    out["Majority"]["spec"] = out["Majority"]["spec"].fillna(0.0)
    return out


def swell_group():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    F = ["hr_mean", "nni_mean", "nni_min", "nni_max", "sdnn", "rmssd", "nn50", "pnn50"]
    d = pd.read_csv("/home/eduarda/Desktop/ProcessedECGData_v5.csv").dropna(subset=F + ["label"])
    d = d[~d.file.str.contains("ECG_values")].reset_index(drop=True)
    X, y = d[F].to_numpy(float), d.label.to_numpy().astype(int)
    g = d.file.str.extract(r"^(S\d+)")[0].to_numpy()
    mk = {"LogReg": lambda: LogisticRegression(max_iter=2000),
          "RandomForest": lambda: RandomForestClassifier(
              n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)}
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        sc = StandardScaler().fit(X[tr])
        rows.append(dict(model="Majority", fold=k,
                         acc=accuracy_score(y[te], np.full(len(te), int(y[tr].mean() > 0.5))) * 100,
                         spec=0.0, auc=0.5))
        for nm, f in mk.items():
            m = f().fit(sc.transform(X[tr]), y[tr])
            p = m.predict_proba(sc.transform(X[te]))[:, 1]
            tn, fp, fn, tp = confusion_matrix(y[te], p > 0.5, labels=[0, 1]).ravel()
            rows.append(dict(model=nm, fold=k, acc=accuracy_score(y[te], p > 0.5) * 100,
                             spec=tn / (tn + fp) * 100, auc=roc_auc_score(y[te], p)))
    r = pd.DataFrame(rows)
    r.to_csv("results_swell_group_baselines.csv", index=False)
    out = {m: r[r.model == m].reset_index(drop=True) for m in ["Majority", "LogReg", "RandomForest"]}
    for a in ["summary", "seq", "both"]:
        out[a] = pd.read_csv(f"results_swell_group_{a}.csv")
    return out


# ---------------------------------------------------------------- formatar
def fmt(x, pct):
    return f"{x:.2f}" if pct else f"{x:.3f}"


def cell_ci(d, k, pct):
    if k not in d or d[k].isna().all():
        return "---"
    m, lo, hi, n = ci(d[k])
    if n < 2 or hi == lo:
        return fmt(m, pct)
    top = 100.0 if pct else 1.0          # o IC da t pode sair dos limites da metrica
    lo, hi = max(lo, 0.0), min(hi, top)
    return f"{fmt(m, pct)} [{fmt(lo, pct)}, {fmt(hi, pct)}]"


def cell_sd(d, k, pct, bold=False):
    if k not in d or d[k].isna().all():
        return "---"
    x = d[k].dropna()
    s = fmt(x.mean(), pct)
    if len(x) > 1 and x.std(ddof=1) > 0:
        s += f" $\\pm$ {fmt(x.std(ddof=1), pct)}"
    return f"\\textbf{{{s}}}" if bold else s


def ci_table(R, metrics, caption, label):
    head = " & ".join(f"\\textbf{{{h}}}" for _, h, _ in metrics)
    L = ["\\begin{table}[h!]", "\\centering", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{|l|" + "c|" * len(metrics) + "}", "\\hline",
         f"\\textbf{{Model}} & {head} \\\\ \\hline"]
    for m in ORDER:
        if m in R:
            L.append(NAMES[m] + " & " + " & ".join(cell_ci(R[m], k, p) for k, _, p in metrics)
                     + " \\\\ \\hline")
    L += ["\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(L)


def best(R, k):
    """Todos os modelos treinados empatados no melhor valor (a 2 casas / 3 casas)."""
    v = {m: R[m][k].mean() for m in TRAINED if m in R and k in R[m]}
    r = 2 if k in ("acc", "spec") else 3
    top = round(max(v.values()), r)
    return {m for m, x in v.items() if round(x, r) == top}


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    out = []
    W = {cv: wesad(cv) for cv in ("random", "group", "loso")}

    # --- tabela principal: WESAD split aleatorio, com PR-AUC
    R = W["random"]
    bb = {k: best(R, k) for k in ("acc", "spec", "auc", "prauc")}
    L = ["% ===== Tabela tab:wesad_random (substitui a atual) =====",
         "\\begin{table}[h!]", "\\centering", "\\small",
         "\\begin{tabular}{|l|c|c|c|c|c|}", "\\hline",
         "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Specificity (\\%)} & \\textbf{AUC} & \\textbf{PR-AUC} & \\textbf{Parameters} \\\\ \\hline"]
    for m in ["Majority", "LogReg", "RandomForest", "summary", "A", "B", "C", "D", "seq", "both"]:
        par = f"{int(R[m]['params'].iloc[0]):,}" if "params" in R[m] else "---"
        L.append(NAMES[m] + " & " + " & ".join(
            cell_sd(R[m], k, p, bold=(m in bb[k])) for k, p in
            (("acc", 1), ("spec", 1), ("auc", 0), ("prauc", 0))) + f" & {par} \\\\ \\hline")
    L += ["\\end{tabular}",
          "\\caption{Performance on WESAD under a random 80/20 split. Mean $\\pm$ standard "
          "deviation over five runs (seeds 42 to 46) on identical splits. The majority classifier "
          "never predicts stress, hence its specificity of 100\\%; its PR-AUC equals the proportion "
          "of stress windows. Bold marks the best trained model in each column. 95\\% confidence "
          "intervals are given in Table \\ref{tab:ci_wesad_random}.}",
          "\\label{tab:wesad_random}", "\\end{table}", ""]
    out.append("\n".join(L))

    # --- tabela principal: WESAD por sujeito, com PR-AUC
    G, S = W["group"], W["loso"]
    bg = {k: best(G, k) for k in ("acc", "auc", "prauc")}
    bs = {k: best(S, k) for k in ("acc", "auc", "prauc")}
    L = ["% ===== Tabela tab:wesad_subject (substitui a atual) =====",
         "\\begin{table}[h!]", "\\centering", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{|l|c|c|c|c|c|c|}", "\\hline",
         "\\textbf{Model} & \\multicolumn{3}{c|}{\\textbf{GroupKFold}} & \\multicolumn{3}{c|}{\\textbf{LOSO}} \\\\ \\cline{2-7}",
         " & \\textbf{Accuracy (\\%)} & \\textbf{AUC} & \\textbf{PR-AUC} & \\textbf{Accuracy (\\%)} & \\textbf{AUC} & \\textbf{PR-AUC} \\\\ \\hline"]
    for m in ["Majority", "LogReg", "RandomForest", "summary", "A", "B", "C", "D", "seq", "both"]:
        L.append(NAMES[m] + " & "
                 + " & ".join(cell_sd(G[m], k, p, bold=(m in bg[k])) for k, p in (("acc", 1), ("auc", 0), ("prauc", 0)))
                 + " & "
                 + " & ".join(cell_sd(S[m], k, p, bold=(m in bs[k])) for k, p in (("acc", 1), ("auc", 0), ("prauc", 0)))
                 + " \\\\ \\hline")
    L += ["\\end{tabular}}",
          "\\caption{Performance on WESAD under subject-wise partitioning: GroupKFold with five "
          "folds and leave-one-subject-out over the 15 subjects. Mean $\\pm$ standard deviation "
          "across folds. 95\\% confidence intervals are given in Tables \\ref{tab:ci_wesad_group} "
          "and \\ref{tab:ci_wesad_loso}.}",
          "\\label{tab:wesad_subject}", "\\end{table}", ""]
    out.append("\n".join(L))

    # --- apendice
    out.append("% ===== Apendice: acrescentar a appendices/SupportWork.tex =====\n"
               "\\subsection{Confidence Intervals}\n\\label{ap:confidence_intervals}\n\n"
               "The tables below give the 95\\% confidence interval of the mean of each metric, "
               "computed with the $t$ distribution over the runs or folds of each protocol, "
               "$\\bar{x} \\pm t_{0.975,\\,n-1}\\, s / \\sqrt{n}$. Under GroupKFold and "
               "leave-one-subject-out the folds are treated as independent samples, which is "
               "the usual approximation; their training sets overlap. Where the interval extends beyond the range of the metric it is truncated at 0 or at its maximum. Values reported from a "
               "single evaluation, such as the confusion matrix and the explainability analysis "
               "of Model G, have no interval.\n")
    M4 = [("acc", "Accuracy (\\%)", 1), ("spec", "Specificity (\\%)", 1), ("auc", "AUC", 0), ("prauc", "PR-AUC", 0)]
    M3 = [("acc", "Accuracy (\\%)", 1), ("spec", "Specificity (\\%)", 1), ("auc", "AUC", 0)]
    out.append(ci_table(W["random"], M4, "95\\% confidence intervals on WESAD under a random 80/20 split, five runs.", "tab:ci_wesad_random"))
    out.append(ci_table(W["group"], M4, "95\\% confidence intervals on WESAD under GroupKFold, five folds.", "tab:ci_wesad_group"))
    out.append(ci_table(W["loso"], M4, "95\\% confidence intervals on WESAD under leave-one-subject-out, 15 folds.", "tab:ci_wesad_loso"))
    P = {cv: pooled(cv) for cv in ("random", "group")}
    out.append(ci_table(P["random"], M3, "95\\% confidence intervals on the pooled dataset under a random 80/20 split, five runs.", "tab:ci_pooled_random"))
    out.append(ci_table(P["group"], M3, "95\\% confidence intervals on the pooled dataset under GroupKFold, five folds over 37 subjects.", "tab:ci_pooled_group"))
    SW = swell_group()
    out.append(ci_table(SW, M3, "95\\% confidence intervals on SWELL-KW alone under GroupKFold, five folds over 22 subjects.", "tab:ci_swell_group"))

    # transferencia
    t = pd.read_csv("results_transfer.csv")
    tmap = {"LogReg": "LogReg", "RandomForest": "RandomForest", "MLP resumo": "summary",
            "Conv1D so serie": "seq", "Conv1D serie+resumo": "both"}
    L = ["\\begin{table}[h!]", "\\centering", "\\footnotesize", "\\begin{tabular}{|l|c|c|}", "\\hline",
         "\\textbf{Model} & \\textbf{Trained on WESAD, tested on SWELL-KW} & \\textbf{Trained on SWELL-KW, tested on WESAD} \\\\ \\hline"]
    for src, key in tmap.items():
        cells = []
        for tr in ("WESAD", "SWELL"):
            x = t[(t.model == src) & (t.train == tr)]
            cells.append(cell_ci(x.reset_index(drop=True), "auc", 0))
        L.append(NAMES[key] + " & " + " & ".join(cells) + " \\\\ \\hline")
    L += ["\\end{tabular}",
          "\\caption{95\\% confidence intervals of the cross-dataset \\gls{AUC}. Logistic regression "
          "and the random forest are deterministic and were evaluated once.}",
          "\\label{tab:ci_transfer}", "\\end{table}", ""]
    out.append("\n".join(L))

    # ablacao F1/F2
    ab = {"F2": pd.read_csv("results_wesad_F2_D.csv"), "F1": pd.read_csv("results_wesad_F1_D.csv")}
    L = ["\\begin{table}[h!]", "\\centering", "\\footnotesize", "\\begin{tabular}{|l|c|c|c|c|}", "\\hline",
         "\\textbf{Subset} & \\textbf{Accuracy (\\%)} & \\textbf{Sensitivity (\\%)} & \\textbf{Specificity (\\%)} & \\textbf{AUC} \\\\ \\hline"]
    for k, d in ab.items():
        L.append(k + " & " + " & ".join(cell_ci(d, c, p) for c, p in (("acc", 1), ("rec", 1), ("spec", 1), ("auc", 0))) + " \\\\ \\hline")
    L += ["\\end{tabular}",
          "\\caption{95\\% confidence intervals of the feature subset ablation for Model D on WESAD, five runs per subset.}",
          "\\label{tab:ci_ablation}", "\\end{table}", ""]
    out.append("\n".join(L))

    os.makedirs("results", exist_ok=True)
    open("results/ci_tables.tex", "w").write("\n".join(out))
    print("escrito results/ci_tables.tex")
