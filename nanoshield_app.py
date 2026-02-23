# ============================================================
# NanoShield AI - FINAL COMPLETE VERSION
# ML + Hybrid + Aggregation + R² + 3D + CSV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------
# 1. MATERIAL DATABASE (Intrinsic Toxicity Factor)
# ------------------------------------------------
materials_db = {
    "silver": 1.3,
    "tio2": 0.8,
    "zno": 1.1,
    "gold": 0.6
}

# ------------------------------------------------
# 2. GENERATE TRAINING DATA
# ------------------------------------------------
def generate_training_data():
    X = []
    y = []
    for coeff in materials_db.values():
        for size in range(10, 101, 10):
            for conc in range(10, 101, 10):
                toxicity = coeff * (100/size) * conc
                X.append([coeff, size, conc])
                y.append(toxicity)
    return np.array(X), np.array(y)

X_train, y_train = generate_training_data()

# ------------------------------------------------
# 3. TRAIN ML MODEL
# ------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

r2 = r2_score(y_train, model.predict(X_train))

print("===== NanoShield AI - FINAL SYSTEM =====")
print("Model R² Score:", round(r2, 4))
print()

# ------------------------------------------------
# 4. SAFE ML PREDICTION FUNCTION
# ------------------------------------------------
def predict_toxicity(material, size_nm, concentration):
    coeff = materials_db.get(material.lower(), 1.0)
    
    size_arr = np.array(size_nm, ndmin=1)
    conc_arr = np.array(concentration, ndmin=1)

    if size_arr.size == 1 and conc_arr.size > 1:
        size_arr = np.full(conc_arr.shape, size_arr[0])
    if conc_arr.size == 1 and size_arr.size > 1:
        conc_arr = np.full(size_arr.shape, conc_arr[0])
        
    coeff_arr = np.full(size_arr.shape, coeff)
    features = np.column_stack((coeff_arr, size_arr, conc_arr))

    return model.predict(features)

# ------------------------------------------------
# 5. HYBRID TOXICITY MODEL
# ------------------------------------------------
def hybrid_toxicity(matA, sizeA, matB, sizeB, total_conc, ratio_A):
    ratio_B = 1 - ratio_A
    
    conc_A = np.array(total_conc) * ratio_A
    conc_B = np.array(total_conc) * ratio_B
    
    T1 = predict_toxicity(matA, sizeA, conc_A)
    T2 = predict_toxicity(matB, sizeB, conc_B)
    
    return (ratio_A * T1) + (ratio_B * T2)

# ------------------------------------------------
# 6. PARTICLE AGGREGATION EFFECT
# ------------------------------------------------
def particle_aggregation_adjustment(toxicity, concentration):
    toxicity = np.array(toxicity)
    concentration = np.array(concentration, ndmin=1)

    if concentration.size == 1:
        if concentration[0] > 60:
            toxicity *= 0.85
    else:
        toxicity[concentration > 60] *= 0.85
        
    return toxicity

# ------------------------------------------------
# 7. BIOLOGICAL DECISION SYSTEM
# ------------------------------------------------
def cell_aggregation_risk(score):
    if score < 30:
        return "Low"
    elif score < 70:
        return "Moderate"
    else:
        return "High"

def application_pathway(score):
    if score < 30:
        return "Biomedical / Cosmetic Use"
    elif score < 70:
        return "Antimicrobial / Therapeutic Use"
    else:
        return "Industrial / Restricted Use"

# ------------------------------------------------
# 8. DEMO HYBRID SYSTEM
# ------------------------------------------------
matA = "silver"
sizeA = 20
matB = "tio2"
sizeB = 80
ratio_A = 0.6
total_conc = 50

tox_value = hybrid_toxicity(matA, sizeA, matB, sizeB, total_conc, ratio_A)
tox_value = particle_aggregation_adjustment(tox_value, total_conc)[0]

print("Hybrid System: 60% Silver (20nm) + 40% TiO2 (80nm)")
print("Predicted Toxicity:", round(tox_value, 2))
print("Cell Aggregation Risk:", cell_aggregation_risk(tox_value))
print("Recommended Application:", application_pathway(tox_value))
print()

# ------------------------------------------------
# 9. GRAPH 1 - Toxicity vs Concentration
# ------------------------------------------------
conc_range = np.linspace(10, 100, 50)

tox_curve = hybrid_toxicity(matA, sizeA, matB, sizeB, conc_range, ratio_A)
tox_curve = particle_aggregation_adjustment(tox_curve, conc_range)

plt.figure()
plt.plot(conc_range, tox_curve)
plt.xlabel("Concentration (µg/mL)")
plt.ylabel("Predicted Toxicity Score")
plt.title("Hybrid Toxicity vs Concentration")
plt.show()

# ------------------------------------------------
# 10. GRAPH 2 - 3D SURFACE (Size vs Conc vs Toxicity)
# ------------------------------------------------
size_range = np.linspace(10, 100, 30)
conc_range_3d = np.linspace(10, 100, 30)

size_mesh, conc_mesh = np.meshgrid(size_range, conc_range_3d)

tox_mesh = predict_toxicity("silver",
                            size_mesh.flatten(),
                            conc_mesh.flatten())

tox_mesh = tox_mesh.reshape(size_mesh.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(size_mesh, conc_mesh, tox_mesh)

ax.set_xlabel("Particle Size (nm)")
ax.set_ylabel("Concentration (µg/mL)")
ax.set_zlabel("Toxicity Score")
ax.set_title("3D Toxicity Surface (Silver Nanoparticles)")

plt.show()

# ------------------------------------------------
# 11. EXPORT CSV REPORT
# ------------------------------------------------
output_df = pd.DataFrame({
    "Concentration": conc_range,
    "Predicted_Toxicity": tox_curve
})

output_df.to_csv("NanoShield_AI_Output.csv", index=False)

print("CSV File Generated: NanoShield_AI_Output.csv")


