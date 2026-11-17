import pandas as pd, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "labels"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
tr = pd.read_csv(f"{D}/train.csv")
print("train rows", len(tr), "studies", tr.StudyInstanceUID.nunique())
g = tr[tr[LAB].notna().any(axis=1)]
print("gold rows", len(g)); print(g[LAB].describe().T[["count", "mean", "min", "max"]])
st = pd.read_csv(f"{D}/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
pk = pd.read_csv(f"{D}/report_labels_v2.csv").set_index("StudyInstanceUID")[LAB]
print("steven", st.shape, "pilk", pk.shape, "common", len(st.index.intersection(pk.index)))
print("steven uniq sample", np.unique(st.values.round(3))[:30])
print("pilk uniq sample", np.unique(pk.values.round(3))[:30])
ser = pd.read_csv(f"{D}/train_series.csv")
print("train_series studies", ser.StudyInstanceUID.nunique(), "planes", ser.Anatomical_Plane.value_counts().to_dict())
