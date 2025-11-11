import numpy as np
import pandas as pd

#  Combine SSE files
def combine_sse_files(input_file1, input_file2, out_file):
    
    files = [input_file1, input_file2]

    with open(out_file, "w") as fout:
        for fname in files:
            with open(fname, "r") as fin:
                for line in fin:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        fout.write(",".join(parts[:4]) + "\n")
    return out_file
    

# Spin classification 
def classify_spin(data, out_file):
    f = open(out_file, "w")
    f.write("#name,actual_spin,actual_spin_mult,actual 1st_excited-GS:energygap,"
            "diff_act_pred_IS-LS,diff_act_pred_HS-IS,predicted_spin,predicted_spin_mult\n")

    df = pd.DataFrame(data, columns=["name", "state", "actual", "predicted"])
    df["state"] = df["state"].astype(int)
    df["actual"] = df["actual"].astype(float)
    df["predicted"] = df["predicted"].astype(float)

    # Pre-group by molecule name
    groups = df.groupby("name")

    for mol, g in groups:
        # ensure sorted by state number (3→5 or 4→6)
        g = g.sort_values("state")
        for i in range(len(g) - 1):
            s1, s2 = g.iloc[i], g.iloc[i + 1]
            if s2["state"] != s1["state"] + 2:
                continue

            # Extract values 
            actual1, actual2 = s1["actual"], s2["actual"]
            pred1, pred2 = s1["predicted"], s2["predicted"]
            sstate = s1["state"]

            # ======== Actual spin classification ========
            def spin_label(a1, a2, sstate):
                if a1 > 0 and a2 > 0:
                    return ("singlet", 1, f"3-1:{a1}") if sstate == 3 else ("doublet", 2, f"4-2:{a1}")
                elif a1 < 0 and a2 < 0:
                    return ("quintet", 5, f"3-5:{-a2}") if sstate == 3 else ("sextet", 6, f"4-6:{-a2}")
                elif a1 < 0 and a2 > 0:
                    if abs(a1) < abs(a2):
                        return ("triplet", 3, f"1-3:{-a1}") if sstate == 3 else ("quartet", 4, f"2-4:{-a1}")
                    else:
                        return ("triplet", 3, f"5-3:{a2}") if sstate == 3 else ("quartet", 4, f"6-4:{a2}")
                elif a1 > 0 and a2 < 0:
                    if abs(a2) < abs(a1):
                        return ("singlet", 1, f"5-1:{a1+a2}") if sstate == 3 else ("doublet", 2, f"6-2:{a1+a2}")
                    else:
                        return ("quintet", 5, f"1-5:{-a2 - a1}") if sstate == 3 else ("sextet", 6, f"2-6:{-a2 - a1}")
                return ("unknown", 0, "none")

            act_label, act_mult, gap = spin_label(actual1, actual2, sstate)
            diff1 = actual1 - pred1
            diff2 = actual2 - pred2

            # ======== Predicted spin classification ========
            def pred_spin_label(p1, p2, sstate):
                if p1 > 0 and p2 > 0:
                    return ("singlet", 1) if sstate == 3 else ("doublet", 2)
                elif p1 < 0 and p2 < 0:
                    return ("quintet", 5) if sstate == 3 else ("sextet", 6)
                elif p1 < 0 and p2 > 0:
                    return ("triplet", 3) if sstate == 3 else ("quartet", 4)
                elif p1 > 0 and p2 < 0:
                    if abs(p2) < abs(p1):
                        return ("singlet", 1) if sstate == 3 else ("doublet", 2)
                    else:
                        return ("quintet", 5) if sstate == 3 else ("sextet", 6)
                return ("unknown", 0)

            pred_label, pred_mult = pred_spin_label(pred1, pred2, sstate)

            # ======== Write results ========
            f.write(f"{mol},{act_label},{act_mult},{gap},{diff1},{diff2},{pred_label},{pred_mult}\n")

    f.close()

    return out_file


#  Evaluate Accuracy
def evaluate_accuracy(inp_file):
    data = np.genfromtxt(inp_file, dtype=str, delimiter=",", skip_header=1)
    total = len(data)
    mismatch = np.sum(data[:, 2] != data[:, 7])
    print(f"Total: {total}")
    print(f"Accurate predictions: {total - mismatch}")
    print(f"Inaccurate predictions: {mismatch}")


if __name__ == "__main__":
    input_file1="monometal_adiabatic_Des-d_IS-LS_test_actVSpred.dat"
    input_file2="monometal_adiabatic_Des-d_HS-IS_test_actVSpred.dat"
    out_file="monometal_adiabatic_Des-d_total_SSE_test_actVSpred.dat"

    concated_SSE = combine_sse_files(input_file1=input_file1, input_file2=input_file2, out_file=out_file)

    data = np.genfromtxt(concated_SSE, dtype=str, delimiter=",")
    spin_file = "monometal_adiabatic_Des-d_spin_test_actVSpred.dat"

    spin_out_file = classify_spin(data, out_file=spin_file)
    evaluate_accuracy(inp_file=spin_out_file)

