# ============================================================
# NanoShield AI - STABLE WORKING VERSION
# ML + Hybrid + Aggregation + R² + Graphs + CSV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------
# 1. MATERIAL DATABASE
# ------------------------------------------------
materials_db = {
    "silver": 1.3,
    "tio2": 0.8,
    "zno": 1.1,
    "gold": 0.6
}

# ------------------------------------------------
# 2. CREATE TRAINING DATA
# ------------------------------------------------
X = []
y = []

for coeff in materials_db.values():
    for size in range(10, 101, 10):
        for conc in range(10, 101, 10):
            toxicity = coeff * (100/size) * conc
            X.append([coeff, size, conc])
            y.append(toxicity)

X = np.array(X)
y = np.array(y)

# ------------------------------------------------
# 3. TRAIN ML MODEL
# ------------------------------------------------
model = LinearRegression()
model.fit(X, y)

r2 = r2_score(y, model.predict(X))

print("===== NanoShield AI =====")
print("Model R² Score:", round(r2, 4))
print()

# ------------------------------------------------
# 4. SAFE PREDICTION FUNCTION
# ------------------------------------------------
def predict_toxicity(material, size, conc):
    coeff = materials_db.get(material.lower(), 1.0)
    features = np.array([[coeff, size, conc]])
    return model.predict(features)[0]

# ------------------------------------------------
# 5. HYBRID MODEL
# ------------------------------------------------
def hybrid_toxicity(matA, sizeA, matB, sizeB, total_conc, ratioA):
    ratioB = 1 - ratioA
    
    concA = total_conc * ratioA
    concB = total_conc * ratioB
    
    T1 = predict_toxicity(matA, sizeA, concA)
    T2 = predict_toxicity(matB, sizeB, concB)
    
    return (ratioA * T1) + (ratioB * T2)

# ------------------------------------------------
# 6. AGGREGATION EFFECT
# ------------------------------------------------
def aggregation_adjustment(toxicity, concentration):
    if concentration > 60:
        return toxicity * 0.85
    return toxicity

# ------------------------------------------------
# 7. BIO DECISION SYSTEM
# ------------------------------------------------
def cell_aggregation(score):
    if score < 30:
        return "Low"
    elif score < 70:
        return "Moderate"
    else:
        return "High"

def application_path(score):
    if score < 30:
        return "Biomedical / Cosmetic"
    elif score < 70:
        return "Antimicrobial / Therapeutic"
    else:
        return "Industrial / Restricted"

# ------------------------------------------------
# 8. EXAMPLE HYBRID CASE
# ------------------------------------------------
matA = "silver"
sizeA = 20

matB = "tio2"
sizeB = 80

ratioA = 0.6
total_conc = 50

tox = hybrid_toxicity(matA, sizeA, matB, sizeB, total_conc, ratioA)
tox = aggregation_adjustment(tox, total_conc)

print("Hybrid: 60% Silver (20nm) + 40% TiO2 (80nm)")
print("Predicted Toxicity:", round(tox, 2))
print("Cell Aggregation Risk:", cell_aggregation(tox))
print("Recommended Application:", application_path(tox))
print()

# ------------------------------------------------
# 9. GRAPH 1 - Toxicity vs Concentration
# ------------------------------------------------
concentrations = np.linspace(10, 100, 50)
tox_values = []

for c in concentrations:
    t = hybrid_toxicity(matA, sizeA, matB, sizeB, c, ratioA)
    t = aggregation_adjustment(t, c)
    tox_values.append(t)

plt.figure()
plt.plot(concentrations, tox_values)
plt.xlabel("Concentration (µg/mL)")
plt.ylabel("Toxicity Score")
plt.title("Hybrid Toxicity vs Concentration")
plt.show()

# ------------------------------------------------
# 10. GRAPH 2 - 3D SURFACE (Single Material)
# ------------------------------------------------
sizes = np.linspace(10, 100, 30)
concs = np.linspace(10, 100, 30)

S, C = np.meshgrid(sizes, concs)
T = np.zeros_like(S)

for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        T[i, j] = predict_toxicity("silver", S[i, j], C[i, j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, C, T)

ax.set_xlabel("Particle Size (nm)")
ax.set_ylabel("Concentration (µg/mL)")
ax.set_zlabel("Toxicity Score")
ax.set_title("3D Toxicity Surface (Silver)")

plt.show()

# ------------------------------------------------
# 11. EXPORT CSV
# ------------------------------------------------
df = pd.DataFrame({
    "Concentration": concentrations,
    "Predicted_Toxicity": tox_values
})

df.to_csv("NanoShield_AI_Output.csv", index=False)

print("CSV file saved as NanoShield_AI_Output.csv")





