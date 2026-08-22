#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### de donde sale el outlier
##### emplear smote y sacar las tres variables #####


# In[1]:


###### MODELOS PREDICTIVOS 2023

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import unicodedata
from docx import Document
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import seaborn as sns
from statsmodels.stats.multitest import multipletests


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

from xgboost import XGBClassifier
from pygam import LogisticGAM, s

from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


# In[ ]:


############################# AÑO 2023 ####################


# In[2]:


def leer_excel_con_pandas(ruta_archivo):
    # Leer archivo Excel, hoja por defecto (la primera)
    pre_23 = pd.read_excel(ruta_archivo,sheet_name='BASE')
    return pre_23
# ESTADO DE SITUACION FINANCIERA 2023
if __name__ == "__main__":
    ruta_excel = r"C:\Users\acerquera\OneDrive - Media Investment Optimization S.A\Escritorio\estadistica\udea\TESIS-QUIEBRA\2df_final_completo_23.xlsx"  # Cambia si está en otra ruta
    pre_23 = leer_excel_con_pandas(ruta_excel)
    print(pre_23.keys())  # Mostrar nombres de las hojas


# In[3]:


vars_modelo = [
    'raz', 'teso', 'rota', 'margenb', 'margen',
    'ractiv', 'rpatri', 'cicop', 'endeu', 'niven', 'apalc', 'apallar',
    'apalfin', 'apaltot', 'cobint', 'activos_pasivos',
    'pasivo_corto_pasivo_total', 'margen_operacional',
    'ctno_ventas_preciso'
]

nulos_vars = (
    pre_23[vars_modelo]
    .isna()
    .sum()
    .reset_index()
)

nulos_vars.columns = ['Variable', 'Cantidad_nulos']

nulos_vars['Porcentaje_nulos'] = (
    nulos_vars['Cantidad_nulos'] / len(pre_23) * 100
).round(2)

nulos_vars = nulos_vars.sort_values(
    by='Cantidad_nulos',
    ascending=False
)

nulos_vars


# In[6]:


vars_modelo_con_apallar = [
    'raz', 'teso', 'rota', 'margenb', 'margen',
    'ractiv', 'rpatri', 'cicop', 'endeu', 'niven', 'apalc', 'apallar',
    'apalfin', 'apaltot', 'cobint', 'activos_pasivos',
    'pasivo_corto_pasivo_total', 'margen_operacional',
    'ctno_ventas_preciso'
]

# Máscara de empresas completas en todas las variables del modelo
mask_completos = pre_23[vars_modelo_con_apallar].notna().all(axis=1)

# Base que quedaría si dejamos apallar y quitamos filas con nulos
pre_23_con_apallar_completo = pre_23.loc[mask_completos].copy()

# Resumen general
resumen_riesgo = pd.DataFrame({
    'Horizonte': ['riesgo_24', 'riesgo_2425'],
    'Casos_riesgo_base_original': [
        pre_23['riesgo_24'].sum(),
        pre_23['riesgo_2425'].sum()
    ],
    'Casos_riesgo_base_completa': [
        pre_23_con_apallar_completo['riesgo_24'].sum(),
        pre_23_con_apallar_completo['riesgo_2425'].sum()
    ],
    'Casos_riesgo_perdidos': [
        pre_23['riesgo_24'].sum() - pre_23_con_apallar_completo['riesgo_24'].sum(),
        pre_23['riesgo_2425'].sum() - pre_23_con_apallar_completo['riesgo_2425'].sum()
    ],
    'Total_empresas_base_original': [
        len(pre_23),
        len(pre_23)
    ],
    'Total_empresas_base_completa': [
        len(pre_23_con_apallar_completo),
        len(pre_23_con_apallar_completo)
    ]
})

resumen_riesgo['Porcentaje_riesgo_retenido'] = (
    resumen_riesgo['Casos_riesgo_base_completa'] /
    resumen_riesgo['Casos_riesgo_base_original'] * 100
).round(2)

resumen_riesgo['Porcentaje_total_retenido'] = (
    resumen_riesgo['Total_empresas_base_completa'] /
    resumen_riesgo['Total_empresas_base_original'] * 100
).round(2)

resumen_riesgo


# In[ ]:


####################3 volver a cargar el excel con nuevas variabele #############


# In[ ]:


##### aca se ponen las trece mas las dos dicotomicas si sin importsntes o no


# In[7]:


vars_keep = [
    "raz","teso","rota",
    "margenb","margen","margen_operacional",
    "ractiv","rpatri",
    "niven",
    "apalc","apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso","activos_pasivos"
]

# Deja solo las que existan en pre_23- 13 variables 
vars_keep = [v for v in vars_keep if v in pre_23.columns]

print("✅ Variables a escanear:", len(vars_keep))
print(vars_keep)


# In[8]:


indicadores = [
    "raz", "teso", "rota",
    "margenb", "margen", "margen_operacional",
    "ractiv","rpatri","activos_pasivos",
    "niven", 
    "apalc", "apaltot",
     "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]

# Dejar solo las que existan en pre_23
indicadores = [v for v in indicadores if v in pre_23.columns]
print("Indicadores usados:", indicadores)


# In[9]:


# Mostrar columnas numeradas
for i, col in enumerate(pre_23.columns, start=1):
    print(i, col)


# In[ ]:


##### OVERVIEW one-year metric table #####


# In[10]:


from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np

resultados = []

for var in indicadores:
    x = pd.to_numeric(pre_23.loc[pre_23["riesgo_24"] == 0, var], errors="coerce").dropna()
    y = pd.to_numeric(pre_23.loc[pre_23["riesgo_24"] == 1, var], errors="coerce").dropna()

    n0, n1 = len(x), len(y)

    if (n0 >= 10) and (n1 >= 10):
        u_res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        u_stat, p_value = float(u_res.statistic), float(u_res.pvalue)
    else:
        u_stat, p_value = np.nan, np.nan

    # --- NUEVO: Calculamos las medianas y la diferencia ANTES del append ---
    med_no_riesgo = float(x.median()) if n0 > 0 else np.nan
    med_riesgo = float(y.median()) if n1 > 0 else np.nan
    diferencia = med_riesgo - med_no_riesgo if (n0 > 0 and n1 > 0) else np.nan
    # -----------------------------------------------------------------------

    resultados.append({
        "Variable": var,
        "Mediana_NoRiesgo": med_no_riesgo,
        "Mediana_Riesgo": med_riesgo,
        "Diferencia": diferencia,        # --- NUEVO: Agregamos la columna aquí ---
        "U_stat": u_stat,
        "p_value": p_value
    })

# Convertimos a DataFrame
tabla_final = pd.DataFrame(resultados)

# Redondeo para presentación
for c in ["Mediana_NoRiesgo", "Mediana_Riesgo", "Diferencia", "U_stat", "p_value"]:
    tabla_final[c] = pd.to_numeric(tabla_final[c], errors="coerce").round(6)

# --- NUEVO: Ordenamos la tabla por la Magnitud de la Diferencia (de mayor a menor) ---
tabla_final['Magnitud_Diferencia'] = tabla_final['Diferencia'].abs()
tabla_final = tabla_final.sort_values(by="Magnitud_Diferencia", ascending=False).reset_index(drop=True)

# Borramos la columna temporal porque ya hizo su trabajo de ordenar
tabla_final = tabla_final.drop(columns=['Magnitud_Diferencia'])
# -------------------------------------------------------------------------------------

# Mostrar la tabla
tabla_final


# In[11]:


import matplotlib.pyplot as plt
import pandas as pd

top_vars = (
    tabla_final
    .dropna(subset=["p_value"])
    .sort_values("p_value")
    .head(6)["Variable"]
    .tolist()
)

medianas = []

for var in top_vars:
    med_no = pre_23.loc[pre_23["riesgo_24"] == 0, var].median()
    med_si = pre_23.loc[pre_23["riesgo_24"] == 1, var].median()
    
    medianas.append({
        "Variable": var,
        "No riesgo": med_no,
        "Riesgo": med_si
    })

df_medianas = pd.DataFrame(medianas)

ax = df_medianas.set_index("Variable")[["No riesgo", "Riesgo"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Comparación de medianas: Riesgo vs No riesgo")
plt.ylabel("Mediana")
plt.xlabel("Variable")
plt.xticks(rotation=45, ha="right")

# Agregar valores numéricos encima de las barras
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

plt.tight_layout()
plt.show()


# In[ ]:


#### overview  TWO-year metrics table 2023 ####


# In[13]:


from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np

def tabla_medianas_wmw_formato_foto(df, indicadores, target="riesgo_2425", min_n=10):
    indicadores = [v for v in indicadores if v in df.columns]
    filas = []

    for var in indicadores:
        x = pd.to_numeric(df.loc[df[target] == 0, var], errors="coerce").dropna()
        y = pd.to_numeric(df.loc[df[target] == 1, var], errors="coerce").dropna()

        n0, n1 = len(x), len(y)

        if (n0 >= min_n) and (n1 >= min_n):
            res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
            U = float(res.statistic)
            p = float(res.pvalue)
        else:
            U, p = np.nan, np.nan

        # --- NUEVO: Calculamos las medianas y la diferencia ---
        med_no_riesgo = float(x.median()) if n0 > 0 else np.nan
        med_riesgo = float(y.median()) if n1 > 0 else np.nan
        diferencia = med_riesgo - med_no_riesgo if (n0 > 0 and n1 > 0) else np.nan

        filas.append({
            "Variable": var,
            "Mediana_NoRiesgo": med_no_riesgo,
            "Mediana_Riesgo": med_riesgo,
            "Diferencia": diferencia,         # <--- Agregamos la columna Diferencia
            "U_stat": U,
            "p_value": p
        })

    # Convertimos a DataFrame
    tabla = pd.DataFrame(filas)

    # ✅ FIXED FORMATTING
    tabla["Mediana_NoRiesgo"] = tabla["Mediana_NoRiesgo"].astype(float).round(6)
    tabla["Mediana_Riesgo"]   = tabla["Mediana_Riesgo"].astype(float).round(6)
    tabla["Diferencia"]       = tabla["Diferencia"].astype(float).round(6) # Formato para la diferencia
    tabla["U_stat"] = tabla["U_stat"].astype(float).round(0)
    
    tabla["p_value"] = tabla["p_value"].astype(float)
    tabla["p_value"] = tabla["p_value"].apply(
        lambda val: 0.0 if pd.notna(val) and val < 1e-6 else round(val, 6)
    )

    # --- NUEVO: Ordenamos por la magnitud de la diferencia (valor absoluto) ---
    tabla['Magnitud_Diferencia'] = tabla['Diferencia'].abs()
    tabla = tabla.sort_values(by="Magnitud_Diferencia", ascending=False).reset_index(drop=True)
    tabla = tabla.drop(columns=['Magnitud_Diferencia'])

    return tabla

# Ejecución de la función
tabla_2425_foto = tabla_medianas_wmw_formato_foto(
    pre_23, indicadores, target="riesgo_2425", min_n=10
)

tabla_2425_foto


# In[14]:


import matplotlib.pyplot as plt
import pandas as pd

# Seleccionar variables más representativas para riesgo_2425
top_vars_2425 = (
    tabla_2425_foto
    .dropna(subset=["p_value"])
    .sort_values("p_value")
    .head(6)["Variable"]
    .tolist()
)

# Calcular medianas por grupo
medianas_2425 = []

for var in top_vars_2425:
    med_no = pre_23.loc[pre_23["riesgo_2425"] == 0, var].median()
    med_si = pre_23.loc[pre_23["riesgo_2425"] == 1, var].median()
    
    medianas_2425.append({
        "Variable": var,
        "No riesgo": med_no,
        "Riesgo": med_si
    })

df_medianas_2425 = pd.DataFrame(medianas_2425)

# Gráfico de barras
ax = df_medianas_2425.set_index("Variable")[["No riesgo", "Riesgo"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Comparación de medianas: Riesgo 2024-2025 vs No riesgo")
plt.ylabel("Mediana")
plt.xlabel("Variable")
plt.xticks(rotation=45, ha="right")

# Agregar valores numéricos
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

plt.tight_layout()
plt.show()


# In[15]:


import pandas as pd
import numpy as np

# 1. Filtrar solo las variables financieras del modelo
vars_modelo = [v for v in indicadores if v in pre_23.columns]

# 2. Resumen descriptivo con percentiles
resumen_desc = pre_23[vars_modelo].describe(
    percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
).T

# 3. Calcular % de nulos
resumen_desc["%_nulos"] = pre_23[vars_modelo].isna().mean() * 100

# 4. Renombrar 50% a Mediana
resumen_desc = resumen_desc.rename(columns={"50%": "mediana"})

# 5. Seleccionar columnas útiles
resumen_desc = resumen_desc[
    ["count", "std", "%_nulos", "min", "1%", "5%", "mediana", "95%", "99%", "max"]
]

# 6. Formato de salida
pd.options.display.float_format = '{:,.4f}'.format

print("--- RESUMEN DESCRIPTIVO Y AUDITORÍA DE OUTLIERS ---")
print("Se usa mediana en lugar de media por robustez frente a outliers")
print("-" * 90)

resumen_desc


# In[ ]:


##### auditorai de raz y rteso


# In[16]:


# Variables a auditar
vars_auditar = ["raz", "teso"]

# Columnas base asociadas a cada ratio
cols_auditoria = [
    "raz", "teso",
    "CurrentAssets",
    "CurrentLiabilities",
    "CashAndCashEquivalents",
    "riesgo_24",
    "riesgo_2425"
]

# Si tienes identificador de empresa, agrégalo aquí
posibles_id = ["Nit", "NIT", "nit", "empresa", "razon_social", "Nombre", "CompanyName"]
cols_id = [c for c in posibles_id if c in pre_23.columns]

cols_finales = cols_id + [c for c in cols_auditoria if c in pre_23.columns]

# Top 10 valores más altos de raz
top_raz = (
    pre_23[cols_finales]
    .sort_values("raz", ascending=False)
    .head(10)
)

# Top 10 valores más altos de teso
top_teso = (
    pre_23[cols_finales]
    .sort_values("teso", ascending=False)
    .head(10)
)

print("TOP 10 - RAZÓN CORRIENTE MÁS ALTA")
display(top_raz)

print("TOP 10 - TESORERÍA MÁS ALTA")
display(top_teso)


# In[17]:


top_raz["raz_recalculada"] = top_raz["CurrentAssets"] / top_raz["CurrentLiabilities"]
top_teso["teso_recalculada"] = top_teso["CashAndCashEquivalents"] / top_teso["CurrentLiabilities"]

display(top_raz)
display(top_teso)


# In[18]:


# Empresas con pasivos corrientes más bajos pero ratios altos
auditoria_denominador = (
    pre_23[cols_finales]
    .dropna(subset=["CurrentLiabilities"])
    .sort_values("CurrentLiabilities", ascending=True)
    .head(20)
)

display(auditoria_denominador)


# In[19]:


pre_23.loc[
    pre_23["raz"] == pre_23["raz"].max(),
    ["NIT", "raz", "Punto de Entrada_balance"]
]


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

variables_graficar = ["activos_pasivos", "raz"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, var in enumerate(variables_graficar):

    data = pre_23[var].dropna()

    p95 = data.quantile(0.95)
    p99 = data.quantile(0.99)
    max_val = data.max()

    sns.boxplot(
        y=data,
        ax=axes[i],
        showfliers=True
    )

    axes[i].axhline(p95, linestyle="--", label=f"P95 = {p95:,.2f}")
    axes[i].axhline(p99, linestyle=":", label=f"P99 = {p99:,.2f}")
    axes[i].scatter(0, max_val, s=80, marker="x", label=f"Máx = {max_val:,.2f}")

    axes[i].set_title(f"Valores extremos en {var}")
    axes[i].set_ylabel(var)
    axes[i].legend()

plt.tight_layout()
plt.show()


# In[ ]:


#################### WINSORIZACION #########################


# In[21]:


import pandas as pd
import numpy as np

# Función para winsorizar una serie
def winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)

# Variables más explosivas: winsorización 1% - 99%
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

# Variables relativamente más estables: winsorización 0.5% - 99.5%
vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

# Crear columnas winsorizadas con sufijo _win
for col in vars_1_99:
    pre_23[col + "_win"] = winsorize_series(pre_23[col], lower=0.01, upper=0.99)

for col in vars_05_995:
    pre_23[col + "_win"] = winsorize_series(pre_23[col], lower=0.005, upper=0.995)


# In[22]:


cols_check = ["raz", "teso", "margen", "activos_pasivos", "apaltot", "rota", "margen_operacional","apalc","ctno_ventas_preciso"]

for col in cols_check:
    print(f"\n===== {col} ORIGINAL =====")
    print(pre_23[col].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
    
    print(f"\n===== {col}_win WINSORIZADA =====")
    print(pre_23[col + "_win"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))


# In[ ]:


####################### MATRIZ DE CORRELACION Y VIF ####################################


# In[23]:


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# DataFrame base
# =========================
df = pre_23.copy()

# =========================
# Variables winsorizadas ya creadas
# =========================
vars_win = [
    "raz_win",
    "teso_win",
    "rota_win",
    "margenb_win",
    "margen_win",
    "margen_operacional_win",
    "ractiv_win",
    "rpatri_win",
    "activos_pasivos_win",
    "niven_win",
    "apalc_win",
    "apaltot_win",
    "pasivo_corto_pasivo_total_win",
    "ctno_ventas_preciso_win"
]

# =========================
# 1) Matriz de correlación
# =========================
corr_matrix = df[vars_win].corr(method="pearson")
print("===== MATRIZ DE CORRELACIÓN =====")
print(corr_matrix.round(3))

# Pares más correlacionados
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["var_1", "var_2", "corr"]
corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
corr_pairs = corr_pairs.sort_values("abs_corr", ascending=False)

print("\n===== PARES MÁS CORRELACIONADOS =====")
print(corr_pairs[["var_1", "var_2", "corr"]].head(20).round(3))

# =========================
# 2) VIF
# =========================
# VIF requiere casos completos
X_vif = df[vars_win].dropna().copy()

# Asegurar que todo sea numérico
for col in X_vif.columns:
    X_vif[col] = pd.to_numeric(X_vif[col], errors="coerce")

X_vif = X_vif.dropna()

vif_data = pd.DataFrame()
vif_data["variable"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

vif_data = vif_data.sort_values("VIF", ascending=False)

print("\n===== VIF =====")
print(vif_data.round(3))


# In[24]:


import matplotlib.pyplot as plt

# =========================
# Heatmap de correlación
# =========================
plt.figure(figsize=(12, 10))
im = plt.imshow(corr_matrix, interpolation="nearest", aspect="auto")
plt.colorbar(im)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)

plt.title("Matriz de correlación - variables winsorizadas")
plt.tight_layout()
plt.show()


# In[ ]:


############################################### more descriptive analysis ######################3


# In[25]:


import pandas as pd
import numpy as np

# =========================
# Base DataFrame
# =========================
df = pre_23.copy()

# =========================
# 1) Overall distribution of Categoria
# =========================
tabla_categoria = (
    df["Categoria"]
    .value_counts(dropna=False)
    .rename_axis("Categoria")
    .reset_index(name="n_empresas")
)

tabla_categoria["pct_total"] = 100 * tabla_categoria["n_empresas"] / len(df)

print("===== DISTRIBUTION OF Categoria =====")
print(tabla_categoria)

# =========================
# 2) Categoria vs riesgo_24
# =========================
tabla_cat_r24 = (
    df.groupby("Categoria", dropna=False)
      .agg(
          total_empresas=("riesgo_24", "size"),
          riesgo_24_positivos=("riesgo_24", "sum")
      )
      .reset_index()
)

tabla_cat_r24["riesgo_24_rate_pct"] = (
    100 * tabla_cat_r24["riesgo_24_positivos"] / tabla_cat_r24["total_empresas"]
)

print("\n===== Categoria vs riesgo_24 =====")
print(tabla_cat_r24.sort_values("riesgo_24_rate_pct", ascending=False))

# =========================
# 3) Categoria vs riesgo_2425
# =========================
tabla_cat_r2425 = (
    df.groupby("Categoria", dropna=False)
      .agg(
          total_empresas=("riesgo_2425", "size"),
          riesgo_2425_positivos=("riesgo_2425", "sum")
      )
      .reset_index()
)

tabla_cat_r2425["riesgo_2425_rate_pct"] = (
    100 * tabla_cat_r2425["riesgo_2425_positivos"] / tabla_cat_r2425["total_empresas"]
)

print("\n===== Categoria vs riesgo_2425 =====")
print(tabla_cat_r2425.sort_values("riesgo_2425_rate_pct", ascending=False))

# =========================
# 4) PROCESO among riesgo_24 positives only
# =========================
df_r24_pos = df[df["riesgo_24"] == 1].copy()

tabla_proceso_r24 = (
    df_r24_pos["PROCESO"]
    .value_counts(dropna=False)
    .rename_axis("PROCESO")
    .reset_index(name="n")
)

tabla_proceso_r24["pct_dentro_riesgo_24"] = (
    100 * tabla_proceso_r24["n"] / len(df_r24_pos)
)

print("\n===== PROCESO within riesgo_24 = 1 =====")
print(tabla_proceso_r24)

# =========================
# 5) PROCESO among riesgo_2425 positives only
# =========================
df_r2425_pos = df[df["riesgo_2425"] == 1].copy()

tabla_proceso_r2425 = (
    df_r2425_pos["PROCESO"]
    .value_counts(dropna=False)
    .rename_axis("PROCESO")
    .reset_index(name="n")
)

tabla_proceso_r2425["pct_dentro_riesgo_2425"] = (
    100 * tabla_proceso_r2425["n"] / len(df_r2425_pos)
)

print("\n===== PROCESO within riesgo_2425 = 1 =====")
print(tabla_proceso_r2425)

# =========================
# 6) Optional cross-tab Categoria x riesgo_24
# =========================
print("\n===== CROSSTAB Categoria x riesgo_24 =====")
print(pd.crosstab(df["Categoria"], df["riesgo_24"], margins=True))

# =========================
# 7) Optional cross-tab Categoria x riesgo_2425
# =========================
print("\n===== CROSSTAB Categoria x riesgo_2425 =====")
print(pd.crosstab(df["Categoria"], df["riesgo_2425"], margins=True))


# In[ ]:


###### MISSING VALUES EN TABLAS#########


# In[26]:


print("NaN en riesgo_24:", pre_23["riesgo_24"].isna().sum())
print("NaN en riesgo_2425:", pre_23["riesgo_2425"].isna().sum())


# In[20]:


####### ¿cuántas observaciones completas quedan si usas las 14 variables finales?


# In[27]:


vars_win = [
    "raz_win",
    "teso_win",
    "rota_win",
    "margenb_win",
    "margen_win",
    "margen_operacional_win",
    "ractiv_win",
    "rpatri_win",
    "activos_pasivos_win",
    "niven_win",
    "apalc_win",
    "apaltot_win",
    "pasivo_corto_pasivo_total_win",
    "ctno_ventas_preciso_win"
]

print("Observaciones totales:", len(pre_23))
print("Casos completos en predictores:", pre_23[vars_win].dropna().shape[0])
print("Casos perdidos:", len(pre_23) - pre_23[vars_win].dropna().shape[0])
print("Porcentaje retenido:", round(100 * pre_23[vars_win].dropna().shape[0] / len(pre_23), 2), "%")


# In[ ]:


###################### NULLS REMO #########################


# In[28]:


base_logit_24 = pre_23[vars_win + ["riesgo_24"]].dropna()
base_logit_2425 = pre_23[vars_win + ["riesgo_2425"]].dropna()

print("Base completa para riesgo_24:", base_logit_24.shape)
print("Positivos riesgo_24:", base_logit_24["riesgo_24"].sum())
print("Negativos riesgo_24:", (base_logit_24["riesgo_24"] == 0).sum())

print("\nBase completa para riesgo_2425:", base_logit_2425.shape)
print("Positivos riesgo_2425:", base_logit_2425["riesgo_2425"].sum())
print("Negativos riesgo_2425:", (base_logit_2425["riesgo_2425"] == 0).sum())


# In[ ]:


################## MODELO 2024 ############### LOGISTICA v1 conservador 


# In[29]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN DE MÉTRICAS
# =====================================================
def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "error_tipo_I": error_tipo_I,
        "error_tipo_II": error_tipo_II,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


# =====================================================
# 5) FUNCIÓN PRINCIPAL: LOGIT CON PESOS MANUALES
# =====================================================
def correr_logit_manual_weights_2024(
    df,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 30, 40, 50, 60),
    thresholds=(0.35, 0.40, 0.45, 0.50, 0.55),
    error_tipo_II_max=0.45,
    error_tipo_I_max=0.10
):
    print("\n" + "=" * 90)
    print(f"MODELO LOGÍSTICO MANUAL WEIGHTS PARA TARGET: {target}")
    print("=" * 90)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # CV
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    resultados_tuning = []

    print("\n===== TUNING DE PESOS MANUALES Y THRESHOLDS EN CV (TRAIN) =====")

    for k in k_values:
        pipe = Pipeline(steps=[
            ("winsor", PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )),
            ("yeojohnson", PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )),
            ("model", LogisticRegression(
                class_weight={0: 1, 1: k},
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state
            ))
        ])

        y_prob_cv = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        roc_auc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_auc_cv = average_precision_score(y_train, y_prob_cv)

        for t in thresholds:
            met = calcular_metricas(y_train, y_prob_cv, t)
            met["k"] = k
            met["roc_auc_cv"] = roc_auc_cv
            met["pr_auc_cv"] = pr_auc_cv
            resultados_tuning.append(met)

    resultados_df = pd.DataFrame(resultados_tuning)

    print("\n===== RESULTADOS DE TUNING EN CV (TRAIN) =====")
    print(
        resultados_df[
            ["k", "threshold", "precision", "recall", "f1_score", "mcc",
             "error_tipo_I", "error_tipo_II", "roc_auc_cv", "pr_auc_cv",
             "TP", "FN", "FP", "TN"]
        ].round(4)
    )

    # -------------------------------------------------
    # Selección del mejor modelo
    # Regla:
    # 1) Recall mínimo
    # 2) Error Tipo I máximo
    # 3) Entre los viables, mayor MCC
    # -------------------------------------------------
    candidatos = resultados_df[
    (resultados_df["error_tipo_II"] <= error_tipo_II_max) &
    (resultados_df["error_tipo_I"] <= error_tipo_I_max)
    ].copy()


    
    if candidatos.empty:
                print("\n⚠️ No hubo candidatos que cumplieran restricciones.")
                print("Se seleccionará el mejor por MCC global.")
                mejor = resultados_df.sort_values(
                    by=["mcc", "f1_score", "pr_auc_cv"],
                    ascending=False
                ).iloc[0]
    else:
            mejor = candidatos.sort_values(
                by=["mcc", "f1_score", "pr_auc_cv"],
                ascending=False
            ).iloc[0]

    k_final = int(mejor["k"])
    threshold_final = float(mejor["threshold"])

    print("\n===== MEJOR CONFIGURACIÓN SELECCIONADA =====")
    print("k_final:", k_final)
    print("threshold_final:", threshold_final)
    print(mejor[[
        "precision", "recall", "f1_score", "mcc",
        "error_tipo_I", "error_tipo_II",
        "roc_auc_cv", "pr_auc_cv",
        "TP", "FN", "FP", "TN"
    ]].round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    pipe_final = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight={0: 1, 1: k_final},
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    pipe_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación en validation
    # -------------------------------------------------
    y_prob_val = pipe_final.predict_proba(X_val)[:, 1]
    met_val = calcular_metricas(y_val, y_prob_val, threshold_final)

    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("k elegido:", k_final)
    print("Threshold elegido:", threshold_final)
    print("Precision:", round(met_val["precision"], 4))
    print("Recall:", round(met_val["recall"], 4))
    print("F1-score:", round(met_val["f1_score"], 4))
    print("MCC:", round(met_val["mcc"], 4))
    print("Error tipo I:", round(met_val["error_tipo_I"], 4))
    print("Error tipo II:", round(met_val["error_tipo_II"], 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        [[met_val["TP"], met_val["FN"]],
         [met_val["FP"], met_val["TN"]]],
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_final}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Statsmodels explicativo usando el mismo peso final
    # -------------------------------------------------
    X_train_win = pipe_final.named_steps["winsor"].transform(X_train)
    X_train_t = pipe_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = np.where(y_train == 1, k_final, 1.0)

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "pipe_final": pipe_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_tuning_cv": resultados_df,
        "mejor_configuracion_cv": mejor,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "k_final": k_final,
        "threshold_final": threshold_final
    }


# =====================================================
# 6) CORRER SOLO 2024
# =====================================================
res_logit_24_balance_mcc_f1 = correr_logit_manual_weights_2024(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,

    # Pesos menos agresivos que la versión sensible
    k_values=(10, 15, 20, 25, 30, 40),

    # Thresholds más altos para bajar falsos positivos
    thresholds=(0.40, 0.45, 0.50, 0.55, 0.60),

    # Permitimos más Error Tipo II porque aceptamos sacrificar recall
    error_tipo_II_max=0.60,

    # Controlamos falsas alarmas
    error_tipo_I_max=0.10
)


# In[ ]:


######################### logituca 2024 v2 ##############


# In[37]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN DE MÉTRICAS
# =====================================================
def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "error_tipo_I": error_tipo_I,
        "error_tipo_II": error_tipo_II,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


# =====================================================
# 5) FUNCIÓN PRINCIPAL: LOGIT CON PESOS MANUALES
# =====================================================
def correr_logit_manual_weights_2024(
    df,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 25, 30, 35),
    thresholds=(0.40, 0.42,0.45,0.48, 0.50),####### (0.40, 0.45, 0.50) cambiar esto
    error_tipo_II_max=0.50,
    error_tipo_I_max=0.10
):
    print("\n" + "=" * 90)
    print(f"MODELO LOGÍSTICO MANUAL WEIGHTS PARA TARGET: {target}")
    print("=" * 90)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # CV
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    resultados_tuning = []

    print("\n===== TUNING DE PESOS MANUALES Y THRESHOLDS EN CV (TRAIN) =====")

    for k in k_values:
        pipe = Pipeline(steps=[
            ("winsor", PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )),
            ("yeojohnson", PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )),
            ("model", LogisticRegression(
                class_weight={0: 1, 1: k},
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state
            ))
        ])

        y_prob_cv = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        roc_auc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_auc_cv = average_precision_score(y_train, y_prob_cv)

        for t in thresholds:
            met = calcular_metricas(y_train, y_prob_cv, t)
            met["k"] = k
            met["roc_auc_cv"] = roc_auc_cv
            met["pr_auc_cv"] = pr_auc_cv
            resultados_tuning.append(met)

    resultados_df = pd.DataFrame(resultados_tuning)

    print("\n===== RESULTADOS DE TUNING EN CV (TRAIN) =====")
    print(
        resultados_df[
            ["k", "threshold", "precision", "recall", "f1_score", "mcc",
             "error_tipo_I", "error_tipo_II", "roc_auc_cv", "pr_auc_cv",
             "TP", "FN", "FP", "TN"]
        ].round(4)
    )

    # -------------------------------------------------
    # Selección del mejor modelo
    # Regla:
    # 1) Recall mínimo
    # 2) Error Tipo I máximo
    # 3) Entre los viables, mayor MCC
    # -------------------------------------------------
    candidatos = resultados_df[
    (resultados_df["recall"] >= 0.50) &
    (resultados_df["error_tipo_I"] <= 0.10)
    ].copy()

    if candidatos.empty:
        candidatos = resultados_df[
            (resultados_df["recall"] >= 0.45) &
            (resultados_df["error_tipo_I"] <= 0.08)
        ].copy()

    if candidatos.empty:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["error_tipo_II", "mcc", "f1_score", "pr_auc_cv"],
        ascending=[True, False, False, False]
    ).iloc[0]
        

    k_final = int(mejor["k"])
    threshold_final = float(mejor["threshold"])

    print("\n===== MEJOR CONFIGURACIÓN SELECCIONADA =====")
    print("k_final:", k_final)
    print("threshold_final:", threshold_final)
    print(mejor[[
        "precision", "recall", "f1_score", "mcc",
        "error_tipo_I", "error_tipo_II",
        "roc_auc_cv", "pr_auc_cv",
        "TP", "FN", "FP", "TN"
    ]].round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    pipe_final = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight={0: 1, 1: k_final},
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    pipe_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación en validation
    # -------------------------------------------------
    y_prob_val = pipe_final.predict_proba(X_val)[:, 1]
    met_val = calcular_metricas(y_val, y_prob_val, threshold_final)

    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("k elegido:", k_final)
    print("Threshold elegido:", threshold_final)
    print("Precision:", round(met_val["precision"], 4))
    print("Recall:", round(met_val["recall"], 4))
    print("F1-score:", round(met_val["f1_score"], 4))
    print("MCC:", round(met_val["mcc"], 4))
    print("Error tipo I:", round(met_val["error_tipo_I"], 4))
    print("Error tipo II:", round(met_val["error_tipo_II"], 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        [[met_val["TP"], met_val["FN"]],
         [met_val["FP"], met_val["TN"]]],
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_final}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Statsmodels explicativo usando el mismo peso final
    # -------------------------------------------------
    X_train_win = pipe_final.named_steps["winsor"].transform(X_train)
    X_train_t = pipe_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = np.where(y_train == 1, k_final, 1.0)

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "pipe_final": pipe_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_tuning_cv": resultados_df,
        "mejor_configuracion_cv": mejor,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "k_final": k_final,
        "threshold_final": threshold_final
    }


res_logit_24 = correr_logit_manual_weights_2024(
            df=pre_23,
            target="riesgo_24",
            vars_modelo=vars_modelo,
            cuts_winsor=cuts_winsor,
            random_state=42,
            k_values=(20, 25, 30, 35),
            thresholds =  ( 0.40, 0.42,0.45,0.48, 0.50),  #### (0.40, 0.45, 0.50)
            error_tipo_II_max=0.50,
            error_tipo_I_max=0.10
        )


# In[ ]:


################ logistica intermedia v3 ########################


# In[33]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN DE MÉTRICAS
# =====================================================
def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "error_tipo_I": error_tipo_I,
        "error_tipo_II": error_tipo_II,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


# =====================================================
# 5) FUNCIÓN PRINCIPAL: LOGIT CON PESOS MANUALES
# =====================================================
def correr_logit_manual_weights_2024(
    df,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 25, 30),
    thresholds=(0.35, 0.38, 0.40, 0.42, 0.45)
):
    
    print("\n" + "=" * 90)
    print(f"MODELO LOGÍSTICO MANUAL WEIGHTS PARA TARGET: {target}")
    print("=" * 90)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # CV
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    resultados_tuning = []

    print("\n===== TUNING DE PESOS MANUALES Y THRESHOLDS EN CV (TRAIN) =====")

    for k in k_values:
        pipe = Pipeline(steps=[
            ("winsor", PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )),
            ("yeojohnson", PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )),
            ("model", LogisticRegression(
                class_weight={0: 1, 1: k},
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state
            ))
        ])

        y_prob_cv = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        roc_auc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_auc_cv = average_precision_score(y_train, y_prob_cv)

        for t in thresholds:
            met = calcular_metricas(y_train, y_prob_cv, t)
            met["k"] = k
            met["roc_auc_cv"] = roc_auc_cv
            met["pr_auc_cv"] = pr_auc_cv
            resultados_tuning.append(met)

    resultados_df = pd.DataFrame(resultados_tuning)

    print("\n===== RESULTADOS DE TUNING EN CV (TRAIN) =====")
    print(
        resultados_df[
            ["k", "threshold", "precision", "recall", "f1_score", "mcc",
             "error_tipo_I", "error_tipo_II", "roc_auc_cv", "pr_auc_cv",
             "TP", "FN", "FP", "TN"]
        ].round(4)
    )

    # -------------------------------------------------
    # Selección del mejor modelo
    # Regla:
    # 1) Recall mínimo
    # 2) Error Tipo I máximo
    # 3) Entre los viables, mayor MCC
    # -------------------------------------------------
    # -------------------------------------------------
# Selección de logística intermedia ajustada a evento raro
# Regla:
# 1) Mayor MCC
# 2) Mayor F1-score
# 3) Mayor recall
# 4) Menor Error Tipo I
# -------------------------------------------------
    candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "recall", "error_tipo_I"],
        ascending=[False, False, False, True]
    ).iloc[0]


    

    k_final = int(mejor["k"])
    threshold_final = float(mejor["threshold"])

    print("\n===== MEJOR CONFIGURACIÓN SELECCIONADA =====")
    print("k_final:", k_final)
    print("threshold_final:", threshold_final)
    print(mejor[[
        "precision", "recall", "f1_score", "mcc",
        "error_tipo_I", "error_tipo_II",
        "roc_auc_cv", "pr_auc_cv",
        "TP", "FN", "FP", "TN"
    ]].round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    pipe_final = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight={0: 1, 1: k_final},
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    pipe_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación en validation
    # -------------------------------------------------
    y_prob_val = pipe_final.predict_proba(X_val)[:, 1]
    met_val = calcular_metricas(y_val, y_prob_val, threshold_final)

    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("k elegido:", k_final)
    print("Threshold elegido:", threshold_final)
    print("Precision:", round(met_val["precision"], 4))
    print("Recall:", round(met_val["recall"], 4))
    print("F1-score:", round(met_val["f1_score"], 4))
    print("MCC:", round(met_val["mcc"], 4))
    print("Error tipo I:", round(met_val["error_tipo_I"], 4))
    print("Error tipo II:", round(met_val["error_tipo_II"], 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        [[met_val["TP"], met_val["FN"]],
         [met_val["FP"], met_val["TN"]]],
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_final}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Statsmodels explicativo usando el mismo peso final
    # -------------------------------------------------
    X_train_win = pipe_final.named_steps["winsor"].transform(X_train)
    X_train_t = pipe_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = np.where(y_train == 1, k_final, 1.0)

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "pipe_final": pipe_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_tuning_cv": resultados_df,
        "mejor_configuracion_cv": mejor,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "k_final": k_final,
        "threshold_final": threshold_final
    }


res_logit_24 = correr_logit_manual_weights_2024(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 25, 30),
    thresholds=(0.35, 0.38, 0.40, 0.42, 0.45)
)



# In[ ]:


######################################## LOGISTICA 2025 V1 FINAL ##################################


# In[40]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]

# =====================================================
# 3) CORTES DE WINSORIZACIÓN
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN DE MÉTRICAS
# =====================================================
def calcular_metricas(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "error_tipo_I": error_tipo_I,
        "error_tipo_II": error_tipo_II,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


# =====================================================
# 5) FUNCIÓN PRINCIPAL: LOGIT MANUAL WEIGHTS PARA 2425
# =====================================================
def correr_logit_manual_weights_2425(
    df,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 30, 40, 50, 60),
    thresholds=(0.35, 0.40, 0.45, 0.50, 0.55),
    error_tipo_II_max=0.45,
    error_tipo_I_max=0.10
):
    print("\n" + "=" * 90)
    print(f"MODELO LOGÍSTICO MANUAL WEIGHTS PARA TARGET: {target}")
    print("=" * 90)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # CV
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    resultados_tuning = []

    print("\n===== TUNING DE PESOS MANUALES Y THRESHOLDS EN CV (TRAIN) =====")

    for k in k_values:
        pipe = Pipeline(steps=[
            ("winsor", PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )),
            ("yeojohnson", PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )),
            ("model", LogisticRegression(
                class_weight={0: 1, 1: k},
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state
            ))
        ])

        y_prob_cv = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        roc_auc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_auc_cv = average_precision_score(y_train, y_prob_cv)

        for t in thresholds:
            met = calcular_metricas(y_train, y_prob_cv, t)
            met["k"] = k
            met["roc_auc_cv"] = roc_auc_cv
            met["pr_auc_cv"] = pr_auc_cv
            resultados_tuning.append(met)

    resultados_df = pd.DataFrame(resultados_tuning)

    print("\n===== RESULTADOS DE TUNING EN CV (TRAIN) =====")
    print(
        resultados_df[
            ["k", "threshold", "precision", "recall", "f1_score", "mcc",
             "error_tipo_I", "error_tipo_II", "roc_auc_cv", "pr_auc_cv",
             "TP", "FN", "FP", "TN"]
        ].round(4)
    )

    # -------------------------------------------------
    # Selección del mejor modelo
    # -------------------------------------------------
    candidatos = resultados_df[
        (resultados_df["error_tipo_II"] <= error_tipo_II_max) &
        (resultados_df["error_tipo_I"] <= error_tipo_I_max)
    ].copy()

    if candidatos.empty:
        print("\n⚠️ No hubo candidatos que cumplieran restricciones.")
        print("Se seleccionará el mejor por MCC global.")
        mejor = resultados_df.sort_values(
            by=["mcc", "f1_score", "pr_auc_cv"],
            ascending=False
        ).iloc[0]
    else:
        mejor = candidatos.sort_values(
            by=["mcc", "f1_score", "pr_auc_cv"],
            ascending=False
        ).iloc[0]

    k_final = int(mejor["k"])
    threshold_final = float(mejor["threshold"])

    print("\n===== MEJOR CONFIGURACIÓN SELECCIONADA =====")
    print("k_final:", k_final)
    print("threshold_final:", threshold_final)
    print(mejor[[
        "precision", "recall", "f1_score", "mcc",
        "error_tipo_I", "error_tipo_II",
        "roc_auc_cv", "pr_auc_cv",
        "TP", "FN", "FP", "TN"
    ]].round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    pipe_final = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight={0: 1, 1: k_final},
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    pipe_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = pipe_final.predict_proba(X_val)[:, 1]
    met_val = calcular_metricas(y_val, y_prob_val, threshold_final)

    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("k elegido:", k_final)
    print("Threshold elegido:", threshold_final)
    print("Precision:", round(met_val["precision"], 4))
    print("Recall:", round(met_val["recall"], 4))
    print("F1-score:", round(met_val["f1_score"], 4))
    print("MCC:", round(met_val["mcc"], 4))
    print("Error tipo I:", round(met_val["error_tipo_I"], 4))
    print("Error tipo II:", round(met_val["error_tipo_II"], 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        [[met_val["TP"], met_val["FN"]],
         [met_val["FP"], met_val["TN"]]],
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_final}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Statsmodels explicativo usando el mismo peso final
    # -------------------------------------------------
    X_train_win = pipe_final.named_steps["winsor"].transform(X_train)
    X_train_t = pipe_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = np.where(y_train == 1, k_final, 1.0)

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "pipe_final": pipe_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_tuning_cv": resultados_df,
        "mejor_configuracion_cv": mejor,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "k_final": k_final,
        "threshold_final": threshold_final
    }


# =====================================================
# 6) CORRER SOLO 2024-2025
# =====================================================
res_2425 = correr_logit_manual_weights_2425(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    k_values=(20, 30, 40, 50, 60),
    thresholds=(0.35, 0.40, 0.45, 0.50, 0.55),
    error_tipo_II_max=0.45,
    error_tipo_I_max=0.10
)


# In[ ]:


############################################### LOGISTICA DOS 2024-2025


# In[41]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import matthews_corrcoef #####
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS, NO WINSORIZADAS
# =====================================================
# =====================================================
# VARIABLES CRUDAS PARA EL MODELO
# No usar *_win aquí para evitar filtración de información
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]

# =====================================================
# VARIABLES MÁS EXPLOSIVAS: WINSORIZACIÓN 1% - 99%
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

# =====================================================
# VARIABLES MÁS ESTABLES: WINSORIZACIÓN 0.5% - 99.5%
# =====================================================
vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

# =====================================================
# DICCIONARIO DE CORTES PARA EL PIPELINE
# =====================================================
cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 3) DEFINIR PERCENTILES DE WINSORIZACIÓN
# =====================================================
# Por defecto: variables con colas pesadas al 1% y 99%.
# Si alguna variable es más estable, aquí se puede poner 0.5% y 99.5%.


# =====================================================
# 4) FUNCIÓN LOGIT BALANCED SIN FILTRACIÓN
# =====================================================
def correr_logit_balanced_sin_leakage(
    df,
    target,
    vars_modelo,
    cuts_winsor=None,
    threshold_elegido=0.30,
    random_state=42
):
    print("\n" + "="*80)
    print(f"MODELO PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(
            f"Faltan estas columnas en el DataFrame: {faltantes}. "
            "En esta versión debes usar variables crudas, no variables *_win."
        )

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Pipeline completo:
    # 1. Winsorización ajustada solo con train/fold
    # 2. Yeo-Johnson ajustado solo con train/fold
    # 3. Regresión logística balanceada
    # -------------------------------------------------
    pipe_logit_balanced = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight="{0: 1, 1: k}",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = cross_val_predict(
        pipe_logit_balanced,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    print("\n===== LOGIT BALANCED - MÉTRICAS GLOBALES EN CV (TRAIN) =====")
    print("ROC-AUC:", round(roc_auc_score(y_train, y_prob_cv), 4))
    print("PR-AUC:", round(average_precision_score(y_train, y_prob_cv), 4))

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [ 0.50, 0.55, 0.60, 0.65,0.70,0.75]   ############################33 GRILLA THRESHOLDS

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv) ####################################################################

        # Según la convención de los artículos que estás usando:
        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc, #########################################################################################
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== LOGIT BALANCED - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final solo con train
    # -------------------------------------------------
    modelo_final = pipe_logit_balanced
    modelo_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = modelo_final.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val) ################################################################################3

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4)) ######################################################################################
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_elegido}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Modelo explicativo opcional con statsmodels
    # usando SOLO train transformado
    # -------------------------------------------------
    X_train_win = modelo_final.named_steps["winsor"].transform(X_train)
    X_train_t = modelo_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "modelo_final": modelo_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val
    }


# =====================================================
# 5) CORRER PARA riesgo_24
# =====================================================
res_24 = correr_logit_balanced_sin_leakage(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.55,
    random_state=42
)


# =====================================================
# 6) CORRER PARA riesgo_2425
# =====================================================
res_2425 = correr_logit_balanced_sin_leakage(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.55,
    random_state=42
)


# In[ ]:


##### regrresion logistica AMBOS AÑOS  @######################### regularizacion ridge-lasso, 


# In[42]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import matthews_corrcoef #####
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight


# =====================================================
# 1) TRANSFORMADOR DE WINSORIZACIÓN SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS, NO WINSORIZADAS
# =====================================================
# =====================================================
# VARIABLES CRUDAS PARA EL MODELO
# No usar *_win aquí para evitar filtración de información
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]

# =====================================================
# VARIABLES MÁS EXPLOSIVAS: WINSORIZACIÓN 1% - 99%
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

# =====================================================
# VARIABLES MÁS ESTABLES: WINSORIZACIÓN 0.5% - 99.5%
# =====================================================
vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

# =====================================================
# DICCIONARIO DE CORTES PARA EL PIPELINE
# =====================================================
cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 3) DEFINIR PERCENTILES DE WINSORIZACIÓN
# =====================================================
# Por defecto: variables con colas pesadas al 1% y 99%.
# Si alguna variable es más estable, aquí se puede poner 0.5% y 99.5%.

cuts_winsor = {
    # Ejemplo de variables más estables:
    # "raz": (0.005, 0.995),
    # "teso": (0.005, 0.995),

    # El resto queda con default_limits=(0.01, 0.99)
}


# =====================================================
# 4) FUNCIÓN LOGIT BALANCED SIN FILTRACIÓN
# =====================================================
def correr_logit_balanced_sin_leakage(
    df,
    target,
    vars_modelo,
    cuts_winsor=None,
    threshold_elegido=0.30,
    random_state=42
):
    print("\n" + "="*80)
    print(f"MODELO PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Verificar columnas
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if len(faltantes) > 0:
        raise ValueError(
            f"Faltan estas columnas en el DataFrame: {faltantes}. "
            "En esta versión debes usar variables crudas, no variables *_win."
        )

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Pipeline completo:
    # 1. Winsorización ajustada solo con train/fold
    # 2. Yeo-Johnson ajustado solo con train/fold
    # 3. Regresión logística balanceada
    # -------------------------------------------------
    pipe_logit_balanced = Pipeline(steps=[
        ("winsor", PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )),
        ("yeojohnson", PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = cross_val_predict(
        pipe_logit_balanced,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    print("\n===== LOGIT BALANCED - MÉTRICAS GLOBALES EN CV (TRAIN) =====")
    print("ROC-AUC:", round(roc_auc_score(y_train, y_prob_cv), 4))
    print("PR-AUC:", round(average_precision_score(y_train, y_prob_cv), 4))

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35]

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv) ####################################################################

        # Según la convención de los artículos que estás usando:
        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc, #########################################################################################
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== LOGIT BALANCED - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final solo con train
    # -------------------------------------------------
    modelo_final = pipe_logit_balanced
    modelo_final.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = modelo_final.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val) ################################################################################3

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4)) ######################################################################################
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION (threshold={threshold_elegido}) =====")
    print(cm_df)

    # -------------------------------------------------
    # Modelo explicativo opcional con statsmodels
    # usando SOLO train transformado
    # -------------------------------------------------
    X_train_win = modelo_final.named_steps["winsor"].transform(X_train)
    X_train_t = modelo_final.named_steps["yeojohnson"].transform(X_train_win)

    X_train_t_sm = pd.DataFrame(
        X_train_t,
        columns=vars_modelo,
        index=X_train.index
    )

    X_train_t_sm_const = sm.add_constant(X_train_t_sm)

    pesos_train = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    modelo_sm = sm.GLM(
        y_train,
        X_train_t_sm_const,
        family=sm.families.Binomial(),
        freq_weights=pesos_train
    ).fit()

    print("\n===== LOGIT EXPLICATIVO STATSMODELS SOBRE TRAIN =====")
    print(modelo_sm.summary())

    return {
        "modelo_final": modelo_final,
        "modelo_statsmodels": modelo_sm,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val
    }


# =====================================================
# 5) CORRER PARA riesgo_24
# =====================================================
res_24 = correr_logit_balanced_sin_leakage(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.55,
    random_state=42
)


# =====================================================
# 6) CORRER PARA riesgo_2425
# =====================================================
res_2425 = correr_logit_balanced_sin_leakage(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.55,
    random_state=42
)


# In[ ]:


##### MODELO GAM  TODOS LOS AÑOS #######################  V1


# In[43]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

from pygam import LogisticGAM, s

def calcular_pesos_balanceados_estables(y, cap=8):
    """
    Calcula pesos balanceados, pero limita pesos extremos para evitar
    divergencia numérica en el GAM.
    """
    pesos = compute_sample_weight(
        class_weight="balanced",
        y=y
    )

    pesos = np.minimum(pesos, cap)
    pesos = pesos / np.mean(pesos)

    return pesos


def ajustar_gam_estable(
    X,
    y,
    weights,
    n_features,
    n_splines=4,
    lam_grid=(50, 100, 300, 500, 1000), ####### ANTES (100, 500, 1000, 5000, 10000)
    max_iter=5000
):
    """
    Ajusta un LogisticGAM probando varios niveles de regularización.
    Si un lambda falla, prueba uno mayor.
    """

    ultimo_error = None

    for lam_try in lam_grid:
        try:
            terms = crear_terminos_gam(
                n_features=n_features,
                n_splines=n_splines
            )

            gam = LogisticGAM(
                terms,
                lam=lam_try,
                max_iter=max_iter,
                tol=1e-3
            )

            gam.fit(
                X,
                y,
                weights=weights
            )

            print(f"   GAM convergió con lam={lam_try}, n_splines={n_splines}")

            return gam, lam_try

        except Exception as e:
            ultimo_error = e
            print(f"   GAM falló con lam={lam_try}: {type(e).__name__}")

    raise RuntimeError(
        f"El GAM no convergió con ningún lambda. Último error: {ultimo_error}"
    )



# =====================================================
# 1) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN USADOS EN TU TESIS
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN PARA CREAR TÉRMINOS SPLINE DEL GAM
# =====================================================
def crear_terminos_gam(n_features, n_splines=5):
    terms = s(0, n_splines=n_splines)

    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)

    return terms


# =====================================================
# 5) FUNCIÓN PRINCIPAL GAM
# =====================================================
def correr_gam_basico(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    threshold_elegido=0.30,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=0.6,
    max_iter=1000
):
    print("\n" + "="*80)
    print(f"MODELO GAM PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------

    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = np.zeros(len(y_train))

    print("\n===== ENTRENANDO GAM CON VALIDACIÓN CRUZADA =====")

    for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr_fold = X_train.iloc[idx_tr].copy()
        X_te_fold = X_train.iloc[idx_te].copy()
        y_tr_fold = y_train.iloc[idx_tr].copy()
        y_te_fold = y_train.iloc[idx_te].copy()

        # -----------------------------
        # Winsorización solo con fold train
        # -----------------------------
        winsor = PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )

        X_tr_win = winsor.fit_transform(X_tr_fold)
        X_te_win = winsor.transform(X_te_fold)

        # -----------------------------
        # Yeo-Johnson solo con fold train
        # -----------------------------
        pt = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_tr_t = pt.fit_transform(X_tr_win)
        X_te_t = pt.transform(X_te_win)

        # -----------------------------
        # Pesos balanceados solo en fold train
        # -----------------------------
        pesos_fold = calcular_pesos_balanceados_estables(
        y=y_tr_fold,
            cap=8
        )

        gam, lam_usado = ajustar_gam_estable(
        X=X_tr_t,
        y=y_tr_fold.values,
        weights=pesos_fold,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),####### 100, 500, 1000, 5000, 10000)
        max_iter=max_iter
        )

        y_prob_cv[idx_te] = gam.predict_proba(X_te_t)

        print(f"Fold {fold} terminado con lam={lam_usado}")

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [0.50, 0.40, 0.35, 0.30,0.28, 0.25, 0.20]

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv) ################################################################################################

        # Convención usada en tus artículos base:
        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "mcc": mcc,######################################################################################################
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== GAM - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    print("\n===== ENTRENANDO GAM FINAL CON TODO EL TRAIN =====")

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    pt_final = PowerTransformer(
        method="yeo-johnson",
        standardize=True
    )

    X_train_t = pt_final.fit_transform(X_train_win)
    X_val_t = pt_final.transform(X_val_win)
##############
    pesos_train = calcular_pesos_balanceados_estables(
    y=y_train,
    cap=8
    )

    gam_final, lam_final_usado = ajustar_gam_estable(
        X=X_train_t,
        y=y_train.values,
        weights=pesos_train,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),##############100, 500, 1000, 5000, 10000
        max_iter=max_iter
    )



    print(f"GAM final convergió con lam={lam_final_usado}")
    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = gam_final.predict_proba(X_val_t)
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)##############################################################################3

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== GAM - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4)) ########################################################################
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION GAM (threshold={threshold_elegido}) =====")
    print(cm_df)

    return {
        "gam_final": gam_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# =====================================================
# 6) CORRER GAM PARA riesgo_24
# =====================================================
res_gam_24 = correr_gam_basico(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.30,###### we had 0.30
    random_state=42,
    n_splits=5,
    n_splines=4,
    lam=100,
    max_iter=5000
)

# =====================================================
# 7) CORRER GAM PARA riesgo_2425
# =====================================================
res_gam_2425 = correr_gam_basico(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.30,
    random_state=42,
    n_splits=5,
    n_splines=4,
    lam=100,
    max_iter=5000
)


# In[ ]:


##################################### MODELO GAM 2024 #################    V2


# In[44]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

from pygam import LogisticGAM, s

def calcular_pesos_balanceados_estables(y, cap=8):
    """
    Calcula pesos balanceados, pero limita pesos extremos para evitar
    divergencia numérica en el GAM.
    """
    pesos = compute_sample_weight(
        class_weight="balanced",
        y=y
    )

    pesos = np.minimum(pesos, cap)
    pesos = pesos / np.mean(pesos)

    return pesos


def ajustar_gam_estable(
    X,
    y,
    weights,
    n_features,
    n_splines=4,
    lam_grid=(50, 100, 300, 500, 1000),
    max_iter=5000
):
    """
    Ajusta un LogisticGAM probando varios niveles de regularización.
    Si un lambda falla, prueba uno mayor.
    """

    ultimo_error = None

    for lam_try in lam_grid:
        try:
            terms = crear_terminos_gam(
                n_features=n_features,
                n_splines=n_splines
            )

            gam = LogisticGAM(
                terms,
                lam=lam_try,
                max_iter=max_iter,
                tol=1e-3
            )

            gam.fit(
                X,
                y,
                weights=weights
            )

            print(f"   GAM convergió con lam={lam_try}, n_splines={n_splines}")

            return gam, lam_try

        except Exception as e:
            ultimo_error = e
            print(f"   GAM falló con lam={lam_try}: {type(e).__name__}")

    raise RuntimeError(
        f"El GAM no convergió con ningún lambda. Último error: {ultimo_error}"
    )


# =====================================================
# 1) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN USADOS EN TU TESIS
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN PARA CREAR TÉRMINOS SPLINE DEL GAM
# =====================================================
def crear_terminos_gam(n_features, n_splines=5):
    terms = s(0, n_splines=n_splines)

    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)

    return terms


# =====================================================
# 5) FUNCIÓN PRINCIPAL GAM
# =====================================================
def correr_gam_basico(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    threshold_elegido=0.28,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=0.6,
    max_iter=1000
):
    print("\n" + "="*80)
    print(f"MODELO GAM PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = np.zeros(len(y_train))

    print("\n===== ENTRENANDO GAM CON VALIDACIÓN CRUZADA =====")

    for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr_fold = X_train.iloc[idx_tr].copy()
        X_te_fold = X_train.iloc[idx_te].copy()
        y_tr_fold = y_train.iloc[idx_tr].copy()
        y_te_fold = y_train.iloc[idx_te].copy()

        # -----------------------------
        # Winsorización solo con fold train
        # -----------------------------
        winsor = PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )

        X_tr_win = winsor.fit_transform(X_tr_fold)
        X_te_win = winsor.transform(X_te_fold)

        # -----------------------------
        # Yeo-Johnson solo con fold train
        # -----------------------------
        pt = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_tr_t = pt.fit_transform(X_tr_win)
        X_te_t = pt.transform(X_te_win)

        # -----------------------------
        # Pesos balanceados solo en fold train
        # -----------------------------
        pesos_fold = calcular_pesos_balanceados_estables(
            y=y_tr_fold,
            cap=8
        )

        gam, lam_usado = ajustar_gam_estable(
            X=X_tr_t,
            y=y_tr_fold.values,
            weights=pesos_fold,
            n_features=len(vars_modelo),
            n_splines=n_splines,
            lam_grid=(50, 100, 300, 500, 1000),
            max_iter=max_iter
        )

        y_prob_cv[idx_te] = gam.predict_proba(X_te_t)

        print(f"Fold {fold} terminado con lam={lam_usado}")

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [0.40, 0.35, 0.32, 0.30, 0.28, 0.27, 0.25]

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv)

        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "mcc": mcc,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== GAM - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    print("\n===== ENTRENANDO GAM FINAL CON TODO EL TRAIN =====")

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    pt_final = PowerTransformer(
        method="yeo-johnson",
        standardize=True
    )

    X_train_t = pt_final.fit_transform(X_train_win)
    X_val_t = pt_final.transform(X_val_win)

    pesos_train = calcular_pesos_balanceados_estables(
        y=y_train,
        cap=8
    )

    gam_final, lam_final_usado = ajustar_gam_estable(
        X=X_train_t,
        y=y_train.values,
        weights=pesos_train,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),
        max_iter=max_iter
    )

    print(f"GAM final convergió con lam={lam_final_usado}")

    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = gam_final.predict_proba(X_val_t)
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== GAM - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION GAM (threshold={threshold_elegido}) =====")
    print(cm_df)

    return {
        "gam_final": gam_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# =====================================================
# 6) CORRER GAM PARA riesgo_24
# =====================================================
res_gam_24 = correr_gam_basico(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.28,
    random_state=42,
    n_splits=5,
    n_splines=4,
    lam=100,
    max_iter=5000
)


# In[ ]:


###################### modelo     GAM V3 #########################


# In[39]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

from pygam import LogisticGAM, s

def calcular_pesos_balanceados_estables(y, cap=8):
    """
    Calcula pesos balanceados, pero limita pesos extremos para evitar
    divergencia numérica en el GAM.
    """
    pesos = compute_sample_weight(
        class_weight="balanced",
        y=y
    )

    pesos = np.minimum(pesos, cap)
    pesos = pesos / np.mean(pesos)

    return pesos


def ajustar_gam_estable(
    X,
    y,
    weights,
    n_features,
    n_splines=5, ######
    lam_grid=(50, 100, 200, 300, 500,800,1000),
    max_iter=5000
):
    """
    Ajusta un LogisticGAM probando varios niveles de regularización.
    Si un lambda falla, prueba uno mayor.
    """

    ultimo_error = None

    for lam_try in lam_grid:
        try:
            terms = crear_terminos_gam(
                n_features=n_features,
                n_splines=n_splines
            )

            gam = LogisticGAM(
                terms,
                lam=lam_try,
                max_iter=max_iter,
                tol=1e-3
            )

            gam.fit(
                X,
                y,
                weights=weights
            )

            print(f"   GAM convergió con lam={lam_try}, n_splines={n_splines}")

            return gam, lam_try

        except Exception as e:
            ultimo_error = e
            print(f"   GAM falló con lam={lam_try}: {type(e).__name__}")

    raise RuntimeError(
        f"El GAM no convergió con ningún lambda. Último error: {ultimo_error}"
    )


# =====================================================
# 1) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN USADOS EN TU TESIS
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN PARA CREAR TÉRMINOS SPLINE DEL GAM
# =====================================================
def crear_terminos_gam(n_features, n_splines=5):
    terms = s(0, n_splines=n_splines)

    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)

    return terms


# =====================================================
# 5) FUNCIÓN PRINCIPAL GAM
# =====================================================
def correr_gam_basico(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    threshold_elegido=0.28,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=0.6,
    max_iter=1000
):
    print("\n" + "="*80)
    print(f"MODELO GAM PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = np.zeros(len(y_train))

    print("\n===== ENTRENANDO GAM CON VALIDACIÓN CRUZADA =====")

    for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr_fold = X_train.iloc[idx_tr].copy()
        X_te_fold = X_train.iloc[idx_te].copy()
        y_tr_fold = y_train.iloc[idx_tr].copy()
        y_te_fold = y_train.iloc[idx_te].copy()

        # -----------------------------
        # Winsorización solo con fold train
        # -----------------------------
        winsor = PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )

        X_tr_win = winsor.fit_transform(X_tr_fold)
        X_te_win = winsor.transform(X_te_fold)

        # -----------------------------
        # Yeo-Johnson solo con fold train
        # -----------------------------
        pt = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_tr_t = pt.fit_transform(X_tr_win)
        X_te_t = pt.transform(X_te_win)

        # -----------------------------
        # Pesos balanceados solo en fold train
        # -----------------------------
        pesos_fold = calcular_pesos_balanceados_estables(
            y=y_tr_fold,
            cap=8
        )

        gam, lam_usado = ajustar_gam_estable(
            X=X_tr_t,
            y=y_tr_fold.values,
            weights=pesos_fold,
            n_features=len(vars_modelo),
            n_splines=n_splines,
            lam_grid=(50, 100, 300, 500, 1000),
            max_iter=max_iter
        )

        y_prob_cv[idx_te] = gam.predict_proba(X_te_t)

        print(f"Fold {fold} terminado con lam={lam_usado}")

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [0.40, 0.35, 0.32, 0.30, 0.28, 0.27, 0.25]

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv)

        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "mcc": mcc,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== GAM - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    print("\n===== ENTRENANDO GAM FINAL CON TODO EL TRAIN =====")

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    pt_final = PowerTransformer(
        method="yeo-johnson",
        standardize=True
    )

    X_train_t = pt_final.fit_transform(X_train_win)
    X_val_t = pt_final.transform(X_val_win)

    pesos_train = calcular_pesos_balanceados_estables(
        y=y_train,
        cap=8
    )

    gam_final, lam_final_usado = ajustar_gam_estable(
        X=X_train_t,
        y=y_train.values,
        weights=pesos_train,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),
        max_iter=max_iter
    )

    print(f"GAM final convergió con lam={lam_final_usado}")

    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = gam_final.predict_proba(X_val_t)
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== GAM - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION GAM (threshold={threshold_elegido}) =====")
    print(cm_df)

    return {
        "gam_final": gam_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# =====================================================
# 6) CORRER GAM PARA riesgo_24
# =====================================================
# =====================================================
# 6) CORRER GAM PARA riesgo_24 - PRUEBA MEJORADA
# =====================================================
res_gam_24_v3_thr32 = correr_gam_basico(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.30,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=200,
    max_iter=5000
)


# In[ ]:


################################# MODELO GAM FINAL V7 2025 ################################################


# In[45]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

from pygam import LogisticGAM, s

def calcular_pesos_balanceados_estables(y, cap=8):
    """
    Calcula pesos balanceados, pero limita pesos extremos para evitar
    divergencia numérica en el GAM.
    """
    pesos = compute_sample_weight(
        class_weight="balanced",
        y=y
    )

    pesos = np.minimum(pesos, cap)
    pesos = pesos / np.mean(pesos)

    return pesos


def ajustar_gam_estable(
    X,
    y,
    weights,
    n_features,
    n_splines=4,
    lam_grid=(50, 100, 300, 500, 1000), ####### ANTES (100, 500, 1000, 5000, 10000)
    max_iter=5000
):
    """
    Ajusta un LogisticGAM probando varios niveles de regularización.
    Si un lambda falla, prueba uno mayor.
    """

    ultimo_error = None

    for lam_try in lam_grid:
        try:
            terms = crear_terminos_gam(
                n_features=n_features,
                n_splines=n_splines
            )

            gam = LogisticGAM(
                terms,
                lam=lam_try,
                max_iter=max_iter,
                tol=1e-3
            )

            gam.fit(
                X,
                y,
                weights=weights
            )

            print(f"   GAM convergió con lam={lam_try}, n_splines={n_splines}")

            return gam, lam_try

        except Exception as e:
            ultimo_error = e
            print(f"   GAM falló con lam={lam_try}: {type(e).__name__}")

    raise RuntimeError(
        f"El GAM no convergió con ningún lambda. Último error: {ultimo_error}"
    )



# =====================================================
# 1) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 2) VARIABLES CRUDAS
# =====================================================
vars_modelo = [
    "raz",
    "teso",
    "rota",
    "margenb",
    "margen",
    "margen_operacional",
    "ractiv",
    "rpatri",
    "activos_pasivos",
    "niven",
    "apalc",
    "apaltot",
    "pasivo_corto_pasivo_total",
    "ctno_ventas_preciso"
]


# =====================================================
# 3) CORTES DE WINSORIZACIÓN USADOS EN TU TESIS
# =====================================================
vars_1_99 = [
    "raz",
    "teso",
    "margen",
    "margen_operacional",
    "activos_pasivos",
    "apalc",
    "apaltot",
    "ctno_ventas_preciso"
]

vars_05_995 = [
    "rota",
    "margenb",
    "ractiv",
    "rpatri",
    "niven",
    "pasivo_corto_pasivo_total"
]

cuts_winsor = {}

for col in vars_1_99:
    cuts_winsor[col] = (0.01, 0.99)

for col in vars_05_995:
    cuts_winsor[col] = (0.005, 0.995)


# =====================================================
# 4) FUNCIÓN PARA CREAR TÉRMINOS SPLINE DEL GAM
# =====================================================
def crear_terminos_gam(n_features, n_splines=5):
    terms = s(0, n_splines=n_splines)

    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)

    return terms


# =====================================================
# 5) FUNCIÓN PRINCIPAL GAM
# =====================================================
def correr_gam_basico(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    threshold_elegido=0.30,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=0.6,
    max_iter=1000
):
    print("\n" + "="*80)
    print(f"MODELO GAM PARA TARGET: {target}")
    print("="*80)

    # -------------------------------------------------
    # Base final sin NaN ni infinitos
    # -------------------------------------------------
    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== TAMAÑO BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TAMAÑOS TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Validación cruzada estratificada en train
    # -------------------------------------------------

    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    y_prob_cv = np.zeros(len(y_train))

    print("\n===== ENTRENANDO GAM CON VALIDACIÓN CRUZADA =====")

    for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr_fold = X_train.iloc[idx_tr].copy()
        X_te_fold = X_train.iloc[idx_te].copy()
        y_tr_fold = y_train.iloc[idx_tr].copy()
        y_te_fold = y_train.iloc[idx_te].copy()

        # -----------------------------
        # Winsorización solo con fold train
        # -----------------------------
        winsor = PercentileWinsorizer(
            feature_names=vars_modelo,
            cuts_by_variable=cuts_winsor,
            default_limits=(0.01, 0.99)
        )

        X_tr_win = winsor.fit_transform(X_tr_fold)
        X_te_win = winsor.transform(X_te_fold)

        # -----------------------------
        # Yeo-Johnson solo con fold train
        # -----------------------------
        pt = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_tr_t = pt.fit_transform(X_tr_win)
        X_te_t = pt.transform(X_te_win)

        # -----------------------------
        # Pesos balanceados solo en fold train
        # -----------------------------
        pesos_fold = calcular_pesos_balanceados_estables(
        y=y_tr_fold,
            cap=8
        )

        gam, lam_usado = ajustar_gam_estable(
        X=X_tr_t,
        y=y_tr_fold.values,
        weights=pesos_fold,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),####### 100, 500, 1000, 5000, 10000)
        max_iter=max_iter
        )

        y_prob_cv[idx_te] = gam.predict_proba(X_te_t)

        print(f"Fold {fold} terminado con lam={lam_usado}")

    # -------------------------------------------------
    # Métricas por umbral en CV
    # -------------------------------------------------
    thresholds = [0.50, 0.40, 0.35, 0.30,0.28, 0.25, 0.20]

    resultados = []

    for t in thresholds:
        y_pred_cv = (y_prob_cv >= t).astype(int)

        cm = confusion_matrix(y_train, y_pred_cv, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_train, y_pred_cv, zero_division=0)
        recall = recall_score(y_train, y_pred_cv, zero_division=0)
        f1 = f1_score(y_train, y_pred_cv, zero_division=0)
        mcc = matthews_corrcoef(y_train, y_pred_cv) ################################################################################################

        # Convención usada en tus artículos base:
        # Error Tipo I: empresa sana clasificada como insolvente
        # Error Tipo II: empresa insolvente clasificada como sana
        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "mcc": mcc,######################################################################################################
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    resultados_df = pd.DataFrame(resultados)

    print("\n===== GAM - MÉTRICAS POR UMBRAL EN CV (TRAIN) =====")
    print(resultados_df.round(4))

    # -------------------------------------------------
    # Ajuste final con todo el train
    # -------------------------------------------------
    print("\n===== ENTRENANDO GAM FINAL CON TODO EL TRAIN =====")

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    pt_final = PowerTransformer(
        method="yeo-johnson",
        standardize=True
    )

    X_train_t = pt_final.fit_transform(X_train_win)
    X_val_t = pt_final.transform(X_val_win)
##############
    pesos_train = calcular_pesos_balanceados_estables(
    y=y_train,
    cap=8
    )

    gam_final, lam_final_usado = ajustar_gam_estable(
        X=X_train_t,
        y=y_train.values,
        weights=pesos_train,
        n_features=len(vars_modelo),
        n_splines=n_splines,
        lam_grid=(50, 100, 300, 500, 1000),##############100, 500, 1000, 5000, 10000
        max_iter=max_iter
    )



    print(f"GAM final convergió con lam={lam_final_usado}")
    # -------------------------------------------------
    # Evaluación final en validation
    # -------------------------------------------------
    y_prob_val = gam_final.predict_proba(X_val_t)
    y_pred_val = (y_prob_val >= threshold_elegido).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)##############################################################################3

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== GAM - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido:", threshold_elegido)
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("MCC:", round(mcc_val, 4)) ########################################################################
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print(f"\n===== MATRIZ DE CONFUSIÓN VALIDATION GAM (threshold={threshold_elegido}) =====")
    print(cm_df)

    return {
        "gam_final": gam_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "resultados_cv": resultados_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# =====================================================
# 6) CORRER GAM PARA riesgo_24
# =====================================================


# =====================================================
# 7) CORRER GAM PARA riesgo_2425
# =====================================================
res_gam_2425 = correr_gam_basico(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.40,
    random_state=42,
    n_splits=5,
    n_splines=4,
    lam=100,
    max_iter=5000
)


# In[ ]:


################################################################ BOOSTING ################################


# In[18]:


get_ipython().system('pip install xgboost')


# In[ ]:


############################ CODIGO BOOSTING


# In[ ]:


####################### MEJOR V1 con variables #######   NO TOCAR V1   FINAL AMBOS AÑOS 


# In[33]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    """
    Genera candidatos de scale_pos_weight.
    Incluye opciones menos agresivas que negativos/positivos.
    """
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_10": 10.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    # Evitar valores absurdamente altos
    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESDE CV
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.25
):
    """
    Selecciona umbral por target y por modelo.
    Prioriza equilibrio: MCC, F1, precisión y control de FP.
    Mantiene un recall mínimo para alerta temprana.
    """

    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    # Si ningún umbral cumple ambas restricciones, relajar error tipo I
    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    # Si aún no hay candidatos, usar todos
    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=50,
    recall_min=0.70,
    errorI_max=0.25,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
        thresholds = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
                      0.35, 0.30, 0.28, 0.25, 0.20, 0.15, 0.10]

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla aleatoria
    # -------------------------------------------------
    param_dist = {
        "n_estimators": [100, 150, 250, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10, 20, 30],
        "gamma": [0, 1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5],
        "reg_lambda": [1, 3, 5, 10, 20, 50],
        "subsample": [0.50, 0.65, 0.80, 0.95],
        "colsample_bytree": [0.40, 0.60, 0.80, 1.00],
        "max_delta_step": [0, 1, 3, 5],
        "spw_key": list(spw_candidatos.keys()),

        # Para árboles no es obligatorio. Lo dejamos como hiperparámetro.
        "usar_yeojohnson": [False, True]
    }

    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr_fold = X_train.iloc[idx_tr].copy()
            X_te_fold = X_train.iloc[idx_te].copy()
            y_tr_fold = y_train.iloc[idx_tr].copy()

            # -----------------------------
            # Winsorización solo con fold train
            # -----------------------------
            winsor = PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )

            X_tr_win = winsor.fit_transform(X_tr_fold)
            X_te_win = winsor.transform(X_te_fold)

            # -----------------------------
            # Yeo-Johnson opcional
            # -----------------------------
            if params["usar_yeojohnson"]:
                pt = PowerTransformer(
                    method="yeo-johnson",
                    standardize=True
                )

                X_tr_model = pt.fit_transform(X_tr_win)
                X_te_model = pt.transform(X_te_win)
            else:
                X_tr_model = X_tr_win.values
                X_te_model = X_te_win.values

            # -----------------------------
            # scale_pos_weight desde fold train
            # -----------------------------
            spw_fold_dict = generar_spw_candidatos(y_tr_fold)
            spw_fold = spw_fold_dict[params["spw_key"]]

            modelo = crear_xgb_modelo_v2(
                params=params,
                scale_pos_weight=spw_fold,
                random_state=random_state
            )

            modelo.fit(
                X_tr_model,
                y_tr_fold.values
            )

            y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        # -----------------------------
        # Métricas CV
        # -----------------------------
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        threshold_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score compuesto orientado a equilibrio
        score = (
            1.00 * fila_thr["mcc"]
            + 0.60 * fila_thr["f1_score"]
            + 0.30 * pr_cv
            + 0.10 * fila_thr["recall"]
            - 0.15 * fila_thr["error_tipo_I"]
            - 0.10 * fila_thr["error_tipo_II"]
        )

        resumen_modelos.append({
            "config_id": i,
            "score": score,
            "threshold_cv": threshold_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_cv": fila_thr["precision"],
            "recall_cv": fila_thr["recall"],
            "f1_cv": fila_thr["f1_score"],
            "mcc_cv": fila_thr["mcc"],
            "error_tipo_I_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_cv": fila_thr["error_tipo_II"],
            "TP_cv": fila_thr["TP"],
            "FN_cv": fila_thr["FN"],
            "FP_cv": fila_thr["FP"],
            "TN_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold CV:", threshold_cv)
        print("Precision CV:", round(fila_thr["precision"], 4))
        print("Recall CV:", round(fila_thr["recall"], 4))
        print("F1 CV:", round(fila_thr["f1_score"], 4))
        print("MCC CV:", round(fila_thr["mcc"], 4))
        print("Error I CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II CV:", round(fila_thr["error_tipo_II"], 4))
        print("Score:", round(score, 4))

        if score > mejor_score:
            mejor_score = score
            mejor_config = params.copy()
            mejor_config["threshold_cv"] = threshold_cv
            mejor_config["score"] = score
            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
        by="score",
        ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN =====")
    print(mejor_config)

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# In[34]:


# =====================================================
# XGBOOST V2 PARA riesgo_24
# =====================================================
res_xgb_v2_24 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25
)


# =====================================================
# XGBOOST V2 PARA riesgo_2425
# =====================================================
res_xgb_v2_2425 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25
)


# In[ ]:


################# BOOSTINF 2024 V4 VIENE DE V1 MEJORADO ############################


# In[37]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    """
    Genera candidatos de scale_pos_weight.
    Incluye opciones menos agresivas que negativos/positivos.
    """
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_10": 10.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    # Evitar valores absurdamente altos
    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESDE CV
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.15
):
    """
    Selecciona umbral por target y por modelo.
    Prioriza equilibrio: MCC, F1, precisión y control de FP.
    Mantiene un recall mínimo para alerta temprana.
    """

    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    # Si ningún umbral cumple ambas restricciones, relajar error tipo I
    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    # Si aún no hay candidatos, usar todos
    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "error_tipo_I", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.15,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
        thresholds = [
        0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
        0.35, 0.30, 0.28, 0.27, 0.26, 0.25,
        0.24, 0.23, 0.22, 0.21, 0.20,
        0.18, 0.15, 0.10
    ]

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split 80/20 estratificado
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla aleatoria
    # -------------------------------------------------
    param_dist = {
        "n_estimators": [100, 150, 250, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10, 20, 30],
        "gamma": [0, 1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5],
        "reg_lambda": [1, 3, 5, 10, 20, 50],
        "subsample": [0.50, 0.65, 0.80, 0.95],
        "colsample_bytree": [0.40, 0.60, 0.80, 1.00],
        "max_delta_step": [0, 1, 3, 5],
        "spw_key": list(spw_candidatos.keys()),

        # Para árboles no es obligatorio. Lo dejamos como hiperparámetro.
        "usar_yeojohnson": [False, True]
    }

    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr_fold = X_train.iloc[idx_tr].copy()
            X_te_fold = X_train.iloc[idx_te].copy()
            y_tr_fold = y_train.iloc[idx_tr].copy()

            # -----------------------------
            # Winsorización solo con fold train
            # -----------------------------
            winsor = PercentileWinsorizer(
                feature_names=vars_modelo,
                cuts_by_variable=cuts_winsor,
                default_limits=(0.01, 0.99)
            )

            X_tr_win = winsor.fit_transform(X_tr_fold)
            X_te_win = winsor.transform(X_te_fold)

            # -----------------------------
            # Yeo-Johnson opcional
            # -----------------------------
            if params["usar_yeojohnson"]:
                pt = PowerTransformer(
                    method="yeo-johnson",
                    standardize=True
                )

                X_tr_model = pt.fit_transform(X_tr_win)
                X_te_model = pt.transform(X_te_win)
            else:
                X_tr_model = X_tr_win.values
                X_te_model = X_te_win.values

            # -----------------------------
            # scale_pos_weight desde fold train
            # -----------------------------
            spw_fold_dict = generar_spw_candidatos(y_tr_fold)
            spw_fold = spw_fold_dict[params["spw_key"]]

            modelo = crear_xgb_modelo_v2(
                params=params,
                scale_pos_weight=spw_fold,
                random_state=random_state
            )

            modelo.fit(
                X_tr_model,
                y_tr_fold.values
            )

            y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        # -----------------------------
        # Métricas CV
        # -----------------------------
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        threshold_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score compuesto orientado a equilibrio
        score = (
            1.00 * fila_thr["mcc"]
            + 0.60 * fila_thr["f1_score"]
            + 0.30 * pr_cv
            + 0.10 * fila_thr["recall"]
            - 0.15 * fila_thr["error_tipo_I"]
            - 0.10 * fila_thr["error_tipo_II"]
        )

        resumen_modelos.append({
            "config_id": i,
            "score": score,
            "threshold_cv": threshold_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_cv": fila_thr["precision"],
            "recall_cv": fila_thr["recall"],
            "f1_cv": fila_thr["f1_score"],
            "mcc_cv": fila_thr["mcc"],
            "error_tipo_I_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_cv": fila_thr["error_tipo_II"],
            "TP_cv": fila_thr["TP"],
            "FN_cv": fila_thr["FN"],
            "FP_cv": fila_thr["FP"],
            "TN_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold CV:", threshold_cv)
        print("Precision CV:", round(fila_thr["precision"], 4))
        print("Recall CV:", round(fila_thr["recall"], 4))
        print("F1 CV:", round(fila_thr["f1_score"], 4))
        print("MCC CV:", round(fila_thr["mcc"], 4))
        print("Error I CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II CV:", round(fila_thr["error_tipo_II"], 4))
        print("Score:", round(score, 4))

        if score > mejor_score:
            mejor_score = score
            mejor_config = params.copy()
            mejor_config["threshold_cv"] = threshold_cv
            mejor_config["score"] = score
            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
        by="score",
        ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN =====")
    print(mejor_config)

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }



# In[38]:


# =====================================================
# XGBOOST V2 PARA riesgo_24
# =====================================================
# =====================================================
# XGBOOST V2 PARA riesgo_24 - REFINADO
# =====================================================
res_xgb_v2_24_refinado = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.15
)


# In[ ]:


######################## boosting 2024 V3 viene de V1 MEJORADO ####################


# In[45]:


import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

# =====================================================
# 1) FUNCIÓN DE MÉTRICAS POR THRESHOLD
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        especificidad = 1 - error_tipo_I if not np.isnan(error_tipo_I) else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "especificidad": especificidad,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) CORRER EL BOOSTING V1 / V2 QUE YA TIENES DEFINIDO
# =====================================================
res_xgb_v2_24 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25
)

# =====================================================
# 3) PROBAR THRESHOLDS MÁS PRUDENTES
#    (Aquí es donde puedes ajustar)
# =====================================================
thresholds_test = [0.28, 0.30, 0.32, 0.35]

resultados_thr_val = evaluar_por_umbrales(
    y_true=res_xgb_v2_24["y_val"],
    y_prob=res_xgb_v2_24["y_prob_validation"],
    thresholds=thresholds_test
)

print("\n===== THRESHOLDS REEVALUADOS EN VALIDATION =====")
print(resultados_thr_val.round(4))

# =====================================================
# 4) SELECCIONAR EL MEJOR THRESHOLD
#    PRIORIDAD:
#    Recall -> Error Tipo II -> Error Tipo I -> Especificidad -> MCC -> F1 -> Precision
# =====================================================
candidatos = resultados_thr_val[
    (resultados_thr_val["recall"] >= 0.72) &
    (resultados_thr_val["error_tipo_I"] <= 0.15)
].copy()

if candidatos.empty:
    candidatos = resultados_thr_val[
        (resultados_thr_val["recall"] >= 0.68) &
        (resultados_thr_val["error_tipo_I"] <= 0.15)
    ].copy()

if candidatos.empty:
    candidatos = resultados_thr_val.copy()

mejor_thr = candidatos.sort_values(
    by=[
        "error_tipo_II",
        "recall",
        "error_tipo_I",
        "especificidad",
        "mcc",
        "f1_score",
        "precision"
    ],
    ascending=[True, False, True, False, False, False, False]
).iloc[0]

threshold_final_nuevo = float(mejor_thr["threshold"])

print("\n===== MEJOR THRESHOLD RESELECCIONADO =====")
print(mejor_thr.round(4))

# =====================================================
# 5) MATRIZ FINAL Y MÉTRICAS FINALES CON EL NUEVO THRESHOLD
# =====================================================
y_pred_val_nuevo = (res_xgb_v2_24["y_prob_validation"] >= threshold_final_nuevo).astype(int)

cm_nueva = confusion_matrix(
    res_xgb_v2_24["y_val"],
    y_pred_val_nuevo,
    labels=[1, 0]
)

cm_df_nueva = pd.DataFrame(
    cm_nueva,
    index=["Real 1", "Real 0"],
    columns=["Predicho 1", "Predicho 0"]
)

roc_auc_val = roc_auc_score(res_xgb_v2_24["y_val"], res_xgb_v2_24["y_prob_validation"])
pr_auc_val = average_precision_score(res_xgb_v2_24["y_val"], res_xgb_v2_24["y_prob_validation"])

print(f"\n===== MATRIZ DE CONFUSIÓN NUEVA (threshold={threshold_final_nuevo}) =====")
print(cm_df_nueva)

print("\n===== MÉTRICAS FINALES CON THRESHOLD RESELECCIONADO =====")
print("Threshold final nuevo:", round(threshold_final_nuevo, 4))
print("Precision:", round(precision_score(res_xgb_v2_24["y_val"], y_pred_val_nuevo, zero_division=0), 4))
print("Recall:", round(recall_score(res_xgb_v2_24["y_val"], y_pred_val_nuevo, zero_division=0), 4))
print("F1-score:", round(f1_score(res_xgb_v2_24["y_val"], y_pred_val_nuevo, zero_division=0), 4))
print("MCC:", round(matthews_corrcoef(res_xgb_v2_24["y_val"], y_pred_val_nuevo), 4))
print("Error tipo I:", round(mejor_thr["error_tipo_I"], 4))
print("Error tipo II:", round(mejor_thr["error_tipo_II"], 4))
print("Especificidad:", round(mejor_thr["especificidad"], 4))
print("ROC-AUC:", round(roc_auc_val, 4))
print("PR-AUC:", round(pr_auc_val, 4))


# In[ ]:


#########################3 MEJORA BOOSTING DE ESTE EL MEJOR ES 2425 N     ######   ESTE ES EL V2 AMBOS


# In[33]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 0) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_10": 10.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESPUÉS DE ELEGIR MODELO
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.25
):
    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) CREAR MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2 CORREGIDA
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
        thresholds = [
            0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
            0.35, 0.30, 0.28, 0.25, 0.20, 0.15, 0.10
        ]

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla XGBoost V2
    # -------------------------------------------------
    param_dist = {
        "n_estimators": [100, 150, 250, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10, 20, 30],
        "gamma": [0, 1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5],
        "reg_lambda": [1, 3, 5, 10, 20, 50],
        "subsample": [0.50, 0.65, 0.80, 0.95],
        "colsample_bytree": [0.40, 0.60, 0.80, 1.00],
        "max_delta_step": [0, 1, 3, 5],
        "spw_key": list(spw_candidatos.keys()),
        "usar_yeojohnson": [False, True]
    }

    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_roc_cv = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        try:
            for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
                X_tr_fold = X_train.iloc[idx_tr].copy()
                X_te_fold = X_train.iloc[idx_te].copy()
                y_tr_fold = y_train.iloc[idx_tr].copy()

                # Winsorización solo con fold train
                winsor = PercentileWinsorizer(
                    feature_names=vars_modelo,
                    cuts_by_variable=cuts_winsor,
                    default_limits=(0.01, 0.99)
                )

                X_tr_win = winsor.fit_transform(X_tr_fold)
                X_te_win = winsor.transform(X_te_fold)

                # Yeo-Johnson opcional
                if params["usar_yeojohnson"]:
                    pt = PowerTransformer(
                        method="yeo-johnson",
                        standardize=True
                    )

                    X_tr_model = pt.fit_transform(X_tr_win)
                    X_te_model = pt.transform(X_te_win)
                else:
                    X_tr_model = X_tr_win.values
                    X_te_model = X_te_win.values

                # scale_pos_weight calculado solo con fold train
                spw_fold_dict = generar_spw_candidatos(y_tr_fold)
                spw_fold = spw_fold_dict[params["spw_key"]]

                modelo = crear_xgb_modelo_v2(
                    params=params,
                    scale_pos_weight=spw_fold,
                    random_state=random_state
                )

                modelo.fit(
                    X_tr_model,
                    y_tr_fold.values
                )

                y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        except Exception as e:
            print(f"Configuración {i} falló: {type(e).__name__} - {e}")
            continue

        # ETAPA 1: OPTIMIZACIÓN POR SCORE HÍBRIDO EN CV
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        # Threshold provisional solo diagnóstico
        threshold_tmp_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score independiente del threshold
        score = pr_cv

        if not np.isfinite(score):
            print("Score inválido. Configuración omitida.")
            continue

        resumen_modelos.append({
            "config_id": i,
            "score_pr_auc": score,
            "threshold_tmp_cv": threshold_tmp_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_tmp_cv": fila_thr["precision"],
            "recall_tmp_cv": fila_thr["recall"],
            "f1_tmp_cv": fila_thr["f1_score"],
            "mcc_tmp_cv": fila_thr["mcc"],
            "error_tipo_I_tmp_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_tmp_cv": fila_thr["error_tipo_II"],
            "TP_tmp_cv": fila_thr["TP"],
            "FN_tmp_cv": fila_thr["FN"],
            "FP_tmp_cv": fila_thr["FP"],
            "TN_tmp_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold provisional CV:", threshold_tmp_cv)
        print("Precision provisional CV:", round(fila_thr["precision"], 4))
        print("Recall provisional CV:", round(fila_thr["recall"], 4))
        print("F1 provisional CV:", round(fila_thr["f1_score"], 4))
        print("MCC provisional CV:", round(fila_thr["mcc"], 4))
        print("Error I provisional CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II provisional CV:", round(fila_thr["error_tipo_II"], 4))
        print("Score PR-AUC:", round(score, 4))

        # Guardar mejor configuración por PR-AUC
        if (
            mejor_config is None
            or score > mejor_score
            or (np.isclose(score, mejor_score) and roc_cv > mejor_roc_cv)
        ):
            mejor_score = score
            mejor_roc_cv = roc_cv

            mejor_config = params.copy()
            mejor_config["score_pr_auc"] = score
            mejor_config["roc_auc_cv"] = roc_cv
            mejor_config["pr_auc_cv"] = pr_cv

            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    if len(resumen_modelos) == 0:
        raise RuntimeError(
            "No se pudo ajustar ninguna configuración válida de XGBoost."
        )

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
        by=["score_hibrido", "roc_auc_cv"],
        ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN POR PR-AUC =====")
    print(mejor_config)

    if mejor_config is None or mejor_thresholds_df is None:
        raise RuntimeError(
            "No se guardó ninguna configuración válida."
        )

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # =====================================================
    # ETAPA 2: SELECCIÓN DEL THRESHOLD DESPUÉS DEL MODELO
    # =====================================================
    threshold_final_cv, fila_threshold_final = seleccionar_umbral_balanceado(
        resultados_df=mejor_thresholds_df,
        recall_min=recall_min,
        errorI_max=errorI_max
    )

    mejor_config["threshold_cv"] = threshold_final_cv

    print("\n===== THRESHOLD FINAL SELECCIONADO DESDE CV =====")
    print("Threshold final:", threshold_final_cv)
    print(fila_threshold_final.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# In[34]:


res_xgb_v2_24 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25
)

res_xgb_v2_2425 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25
)


# In[ ]:


############################ MEJORA V2 2024 ~@@@@@@@@@@@@@     V2 2024 


# In[29]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 0) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_10": 10.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESPUÉS DE ELEGIR MODELO
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.25
):
    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) CREAR MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2 CORREGIDA
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
         thresholds = [
        0.20, 0.19, 0.18, 0.17, 0.16, 0.15
    ]

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla XGBoost V2
    # -------------------------------------------------
    param_dist = {
        "n_estimators": [100, 150, 250, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10, 20, 30],
        "gamma": [0, 1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5],
        "reg_lambda": [1, 3, 5, 10, 20, 50],
        "subsample": [0.50, 0.65, 0.80, 0.95],
        "colsample_bytree": [0.40, 0.60, 0.80, 1.00],
        "max_delta_step": [0, 1, 3, 5],
        "spw_key": list(spw_candidatos.keys()),
        "usar_yeojohnson": [False, True]
    }

    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_roc_cv = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        try:
            for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
                X_tr_fold = X_train.iloc[idx_tr].copy()
                X_te_fold = X_train.iloc[idx_te].copy()
                y_tr_fold = y_train.iloc[idx_tr].copy()

                # Winsorización solo con fold train
                winsor = PercentileWinsorizer(
                    feature_names=vars_modelo,
                    cuts_by_variable=cuts_winsor,
                    default_limits=(0.01, 0.99)
                )

                X_tr_win = winsor.fit_transform(X_tr_fold)
                X_te_win = winsor.transform(X_te_fold)

                # Yeo-Johnson opcional
                if params["usar_yeojohnson"]:
                    pt = PowerTransformer(
                        method="yeo-johnson",
                        standardize=True
                    )

                    X_tr_model = pt.fit_transform(X_tr_win)
                    X_te_model = pt.transform(X_te_win)
                else:
                    X_tr_model = X_tr_win.values
                    X_te_model = X_te_win.values

                # scale_pos_weight calculado solo con fold train
                spw_fold_dict = generar_spw_candidatos(y_tr_fold)
                spw_fold = spw_fold_dict[params["spw_key"]]

                modelo = crear_xgb_modelo_v2(
                    params=params,
                    scale_pos_weight=spw_fold,
                    random_state=random_state
                )

                modelo.fit(
                    X_tr_model,
                    y_tr_fold.values
                )

                y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        except Exception as e:
            print(f"Configuración {i} falló: {type(e).__name__} - {e}")
            continue

        # ETAPA 1: OPTIMIZACIÓN POR SCORE HÍBRIDO EN CV
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        # Threshold provisional solo diagnóstico
        threshold_tmp_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score independiente del threshold
        score = pr_cv

        if not np.isfinite(score):
            print("Score inválido. Configuración omitida.")
            continue

        resumen_modelos.append({
            "config_id": i,
            "score_pr_auc": score,
            "threshold_tmp_cv": threshold_tmp_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_tmp_cv": fila_thr["precision"],
            "recall_tmp_cv": fila_thr["recall"],
            "f1_tmp_cv": fila_thr["f1_score"],
            "mcc_tmp_cv": fila_thr["mcc"],
            "error_tipo_I_tmp_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_tmp_cv": fila_thr["error_tipo_II"],
            "TP_tmp_cv": fila_thr["TP"],
            "FN_tmp_cv": fila_thr["FN"],
            "FP_tmp_cv": fila_thr["FP"],
            "TN_tmp_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold provisional CV:", threshold_tmp_cv)
        print("Precision provisional CV:", round(fila_thr["precision"], 4))
        print("Recall provisional CV:", round(fila_thr["recall"], 4))
        print("F1 provisional CV:", round(fila_thr["f1_score"], 4))
        print("MCC provisional CV:", round(fila_thr["mcc"], 4))
        print("Error I provisional CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II provisional CV:", round(fila_thr["error_tipo_II"], 4))
        print("Score PR-AUC:", round(score, 4))

        # Guardar mejor configuración por PR-AUC
        if (
            mejor_config is None
            or score > mejor_score
            or (np.isclose(score, mejor_score) and roc_cv > mejor_roc_cv)
        ):
            mejor_score = score
            mejor_roc_cv = roc_cv

            mejor_config = params.copy()
            mejor_config["score_pr_auc"] = score
            mejor_config["roc_auc_cv"] = roc_cv
            mejor_config["pr_auc_cv"] = pr_cv

            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    if len(resumen_modelos) == 0:
        raise RuntimeError(
            "No se pudo ajustar ninguna configuración válida de XGBoost."
        )

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
        by=["score_pr_auc", "roc_auc_cv"],
        ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN POR PR-AUC =====")
    print(mejor_config)

    if mejor_config is None or mejor_thresholds_df is None:
        raise RuntimeError(
            "No se guardó ninguna configuración válida."
        )

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # =====================================================
    # ETAPA 2: SELECCIÓN DEL THRESHOLD DESPUÉS DEL MODELO
    # =====================================================
    threshold_final_cv, fila_threshold_final = seleccionar_umbral_balanceado(
        resultados_df=mejor_thresholds_df,
        recall_min=recall_min,
        errorI_max=errorI_max
    )

    mejor_config["threshold_cv"] = threshold_final_cv

    print("\n===== THRESHOLD FINAL SELECCIONADO DESDE CV =====")
    print("Threshold final:", threshold_final_cv)
    print(fila_threshold_final.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }

res_xgb_24_v2_recall074 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.74,
    errorI_max=0.16
)


# In[ ]:


################################ MEJORA DE 2425 boosting V2   ##############################    2425 V3


# In[ ]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 0) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_10": 10.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESPUÉS DE ELEGIR MODELO
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.25
):
    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) CREAR MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2 CORREGIDA
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
        thresholds = [
            0.50, 0.48, 0.46,0.45,
            0.44, 0.42, 0.41, 0.40, 0.38
        ]

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla XGBoost V2
    # -------------------------------------------------
    param_dist = {
        "n_estimators": [100, 150, 250, 400, 600, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 3, 5, 10, 20, 30],
        "gamma": [0, 1, 3, 5, 8, 10],
        "reg_alpha": [0, 0.1, 0.5, 1, 3, 5],
        "reg_lambda": [1, 3, 5, 10, 20, 50],
        "subsample": [0.50, 0.65, 0.80, 0.95],
        "colsample_bytree": [0.40, 0.60, 0.80, 1.00],
        "max_delta_step": [0, 1, 3, 5],
        "spw_key": list(spw_candidatos.keys()),
        "usar_yeojohnson": [False, True]
    }

    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_roc_cv = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        try:
            for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
                X_tr_fold = X_train.iloc[idx_tr].copy()
                X_te_fold = X_train.iloc[idx_te].copy()
                y_tr_fold = y_train.iloc[idx_tr].copy()

                # Winsorización solo con fold train
                winsor = PercentileWinsorizer(
                    feature_names=vars_modelo,
                    cuts_by_variable=cuts_winsor,
                    default_limits=(0.01, 0.99)
                )

                X_tr_win = winsor.fit_transform(X_tr_fold)
                X_te_win = winsor.transform(X_te_fold)

                # Yeo-Johnson opcional
                if params["usar_yeojohnson"]:
                    pt = PowerTransformer(
                        method="yeo-johnson",
                        standardize=True
                    )

                    X_tr_model = pt.fit_transform(X_tr_win)
                    X_te_model = pt.transform(X_te_win)
                else:
                    X_tr_model = X_tr_win.values
                    X_te_model = X_te_win.values

                # scale_pos_weight calculado solo con fold train
                spw_fold_dict = generar_spw_candidatos(y_tr_fold)
                spw_fold = spw_fold_dict[params["spw_key"]]

                modelo = crear_xgb_modelo_v2(
                    params=params,
                    scale_pos_weight=spw_fold,
                    random_state=random_state
                )

                modelo.fit(
                    X_tr_model,
                    y_tr_fold.values
                )

                y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        except Exception as e:
            print(f"Configuración {i} falló: {type(e).__name__} - {e}")
            continue

        # =====================================================
        # ETAPA 1: OPTIMIZACIÓN SOLO CON PROBABILIDADES
        # =====================================================
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        # Threshold provisional solo diagnóstico
        threshold_tmp_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score PR-AUC independiente del threshold
        score = pr_cv

        if not np.isfinite(score):
            print("Score inválido. Configuración omitida.")
            continue

        resumen_modelos.append({
            "config_id": i,
            "score_pr_auc": score,
            "threshold_tmp_cv": threshold_tmp_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_tmp_cv": fila_thr["precision"],
            "recall_tmp_cv": fila_thr["recall"],
            "f1_tmp_cv": fila_thr["f1_score"],
            "mcc_tmp_cv": fila_thr["mcc"],
            "error_tipo_I_tmp_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_tmp_cv": fila_thr["error_tipo_II"],
            "TP_tmp_cv": fila_thr["TP"],
            "FN_tmp_cv": fila_thr["FN"],
            "FP_tmp_cv": fila_thr["FP"],
            "TN_tmp_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold provisional CV:", threshold_tmp_cv)
        print("Precision provisional CV:", round(fila_thr["precision"], 4))
        print("Recall provisional CV:", round(fila_thr["recall"], 4))
        print("F1 provisional CV:", round(fila_thr["f1_score"], 4))
        print("MCC provisional CV:", round(fila_thr["mcc"], 4))
        print("Error I provisional CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II provisional CV:", round(fila_thr["error_tipo_II"], 4))
        print("Score PR-AUC:", round(score, 4))

        # Guardar mejor configuración por PR-AUC                  VIERNRES    # Guardar mejor configuración por score híbrido
        if (
            mejor_config is None
            or score > mejor_score
            or (np.isclose(score, mejor_score) and roc_cv > mejor_roc_cv)
        ):
            mejor_score = score
            mejor_roc_cv = roc_cv

            mejor_config = params.copy()
            mejor_config["score_pr_auc"] = score    ##### mejor_config["score_hibrido"] = score VIERNES
            mejor_config["roc_auc_cv"] = roc_cv
            mejor_config["pr_auc_cv"] = pr_cv

            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    if len(resumen_modelos) == 0:
        raise RuntimeError(
            "No se pudo ajustar ninguna configuración válida de XGBoost."
        )

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
        by=["score_pr_auc", "roc_auc_cv"],
        ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN POR PR-AUC =====")
    print(mejor_config)

    if mejor_config is None or mejor_thresholds_df is None:
        raise RuntimeError(
            "No se guardó ninguna configuración válida."
        )

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # =====================================================
    # ETAPA 2: SELECCIÓN DEL THRESHOLD DESPUÉS DEL MODELO
    # =====================================================
    threshold_final_cv, fila_threshold_final = seleccionar_umbral_balanceado(
        resultados_df=mejor_thresholds_df,
        recall_min=recall_min,
        errorI_max=errorI_max
    )

    mejor_config["threshold_cv"] = threshold_final_cv

    print("\n===== THRESHOLD FINAL SELECCIONADO DESDE CV =====")
    print("Threshold final:", threshold_final_cv)
    print(fila_threshold_final.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# In[34]:


res_xgb_v2_2425 = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_2425",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.60,
    errorI_max=0.12
)


# In[ ]:


############### prueba 2024 mejor ############## V3


# In[35]:


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# =====================================================
# 0) WINSORIZADOR SIN FILTRACIÓN
# =====================================================
class PercentileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, cuts_by_variable=None, default_limits=(0.01, 0.99)):
        self.feature_names = feature_names
        self.cuts_by_variable = cuts_by_variable
        self.default_limits = default_limits

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.feature_names:
            lower_q, upper_q = self.default_limits

            if self.cuts_by_variable is not None and col in self.cuts_by_variable:
                lower_q, upper_q = self.cuts_by_variable[col]

            self.lower_bounds_[col] = X_df[col].quantile(lower_q)
            self.upper_bounds_[col] = X_df[col].quantile(upper_q)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()

        for col in self.feature_names:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_[col],
                upper=self.upper_bounds_[col]
            )

        return X_df


# =====================================================
# 1) MÉTRICAS POR UMBRAL
# =====================================================
def evaluar_por_umbrales(y_true, y_prob, thresholds):
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        tp = cm[0, 0]
        fn = cm[0, 1]
        fp = cm[1, 0]
        tn = cm[1, 1]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        error_tipo_I = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        error_tipo_II = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        resultados.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mcc": mcc,
            "error_tipo_I": error_tipo_I,
            "error_tipo_II": error_tipo_II,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn
        })

    return pd.DataFrame(resultados)


# =====================================================
# 2) SCALE_POS_WEIGHT MENOS AGRESIVO
# =====================================================
def generar_spw_candidatos(y):
    y = pd.Series(y)
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())

    if positivos == 0:
        return {"spw_1": 1.0}

    ratio = negativos / positivos
    sqrt_ratio = np.sqrt(ratio)

    candidatos = {
        "spw_1": 1.0,
        "spw_5": 5.0,
        "spw_8": 8.0,
        "spw_10": 10.0,
        "spw_12": 12.0,
        "spw_15": 15.0,
        "spw_20": 20.0,
        "spw_30": 30.0,
        "spw_sqrt": sqrt_ratio,
        "spw_010_ratio": 0.10 * ratio,
        "spw_015_ratio": 0.15 * ratio,
        "spw_020_ratio": 0.20 * ratio,
        "spw_025_ratio": 0.25 * ratio,
        "spw_035_ratio": 0.35 * ratio,
        "spw_050_ratio": 0.50 * ratio,
        "spw_075_ratio": 0.75 * ratio,
        "spw_100_ratio": ratio
    }

    candidatos = {k: float(min(v, ratio)) for k, v in candidatos.items()}

    return candidatos


# =====================================================
# 3) SELECCIÓN DE THRESHOLD DESPUÉS DE ELEGIR MODELO
# =====================================================
def seleccionar_umbral_balanceado(
    resultados_df,
    recall_min=0.70,
    errorI_max=0.25
):
    candidatos = resultados_df[
        (resultados_df["recall"] >= recall_min) &
        (resultados_df["error_tipo_I"] <= errorI_max)
    ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df[
            resultados_df["recall"] >= recall_min
        ].copy()

    if len(candidatos) == 0:
        candidatos = resultados_df.copy()

    mejor = candidatos.sort_values(
        by=["mcc", "f1_score", "precision", "recall"],
        ascending=False
    ).iloc[0]

    return float(mejor["threshold"]), mejor


# =====================================================
# 4) CREAR MODELO XGBOOST
# =====================================================
def crear_xgb_modelo_v2(params, scale_pos_weight, random_state=42):
    modelo = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        max_delta_step=params["max_delta_step"],
        scale_pos_weight=scale_pos_weight,

        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    return modelo


# =====================================================
# 5) FUNCIÓN PRINCIPAL XGBOOST V2 CORREGIDA
# =====================================================
def correr_xgboost_tuning_v2(
    df,
    target,
    vars_modelo,
    cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=60,
    recall_min=0.70,
    errorI_max=0.25,
    thresholds=None
):
    print("\n" + "="*90)
    print(f"XGBOOST V2 TUNING PARA TARGET: {target}")
    print("="*90)

    if thresholds is None:
             thresholds = [0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15]     ####### treshold

    columnas_necesarias = vars_modelo + [target]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if len(faltantes) > 0:
        raise ValueError(f"Faltan estas columnas en el DataFrame: {faltantes}")

    base = (
        df[columnas_necesarias]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = base[vars_modelo].copy()
    y = base[target].astype(int).copy()

    print("\n===== BASE FINAL =====")
    print("Shape:", base.shape)
    print("Positivos:", int(y.sum()))
    print("Negativos:", int((y == 0).sum()))
    print("Tasa de eventos:", round(y.mean(), 4))

    # -------------------------------------------------
    # Split train / validation
    # -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state
    )

    print("\n===== TRAIN / VALIDATION =====")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("Positivos train:", int(y_train.sum()))
    print("Positivos validation:", int(y_val.sum()))

    # -------------------------------------------------
    # Candidatos scale_pos_weight
    # -------------------------------------------------
    spw_candidatos = generar_spw_candidatos(y_train)

    print("\n===== CANDIDATOS SCALE_POS_WEIGHT =====")
    for k, v in spw_candidatos.items():
        print(k, "=", round(v, 4))

    # -------------------------------------------------
    # Grilla XGBoost V2                                 ##################3 param dist ##
    # -------------------------------------------------
    param_dist = {
    # Cerca del mejor previo: 400 árboles y learning_rate 0.03
    "n_estimators": [300, 400, 500, 600],
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],

    # El mejor previo fue max_depth = 2
    "max_depth": [2, 3],

    # El mejor previo fue min_child_weight = 3
    "min_child_weight": [1, 3, 5, 10],

    # El mejor previo fue gamma = 8
    "gamma": [5, 8, 10, 12],

    # El mejor previo fue reg_alpha = 5
    "reg_alpha": [3, 5, 7, 10],

    # El mejor previo fue reg_lambda = 3
    "reg_lambda": [1, 3, 5, 10, 20],

    # El mejor previo fue subsample = 0.5
    "subsample": [0.50, 0.65, 0.80],

    # El mejor previo fue colsample_bytree = 0.4
    "colsample_bytree": [0.40, 0.60, 0.80],

    # El mejor previo fue max_delta_step = 0
    "max_delta_step": [0, 1, 3],

    # El mejor previo fue spw_015_ratio
    "spw_key": [
        "spw_10",
        "spw_12",
        "spw_15",
        "spw_015_ratio",
        "spw_020_ratio"
    ],

    # El mejor previo usaba Yeo-Johnson
    "usar_yeojohnson": [True]
    }   
    sampler = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=n_iter,
        random_state=random_state
    ))

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    resumen_modelos = []

    mejor_score = -np.inf
    mejor_roc_cv = -np.inf
    mejor_config = None
    mejor_thresholds_df = None
    mejor_y_prob_cv = None

    # -------------------------------------------------
    # Tuning CV
    # -------------------------------------------------
    for i, params in enumerate(sampler, start=1):
        print("\n" + "-"*80)
        print(f"Configuración {i}/{n_iter}")
        print(params)

        y_prob_cv = np.zeros(len(y_train))

        try:
            for fold, (idx_tr, idx_te) in enumerate(cv.split(X_train, y_train), start=1):
                X_tr_fold = X_train.iloc[idx_tr].copy()
                X_te_fold = X_train.iloc[idx_te].copy()
                y_tr_fold = y_train.iloc[idx_tr].copy()

                # Winsorización solo con fold train
                winsor = PercentileWinsorizer(
                    feature_names=vars_modelo,
                    cuts_by_variable=cuts_winsor,
                    default_limits=(0.01, 0.99)
                )

                X_tr_win = winsor.fit_transform(X_tr_fold)
                X_te_win = winsor.transform(X_te_fold)

                # Yeo-Johnson opcional
                if params["usar_yeojohnson"]:
                    pt = PowerTransformer(
                        method="yeo-johnson",
                        standardize=True
                    )

                    X_tr_model = pt.fit_transform(X_tr_win)
                    X_te_model = pt.transform(X_te_win)
                else:
                    X_tr_model = X_tr_win.values
                    X_te_model = X_te_win.values

                # scale_pos_weight calculado solo con fold train
                spw_fold_dict = generar_spw_candidatos(y_tr_fold)
                spw_fold = spw_fold_dict[params["spw_key"]]

                modelo = crear_xgb_modelo_v2(
                    params=params,
                    scale_pos_weight=spw_fold,
                    random_state=random_state
                )

                modelo.fit(
                    X_tr_model,
                    y_tr_fold.values
                )

                y_prob_cv[idx_te] = modelo.predict_proba(X_te_model)[:, 1]

        except Exception as e:
            print(f"Configuración {i} falló: {type(e).__name__} - {e}")
            continue

        # =====================================================
              # ETAPA 1: OPTIMIZACIÓN POR SCORE HÍBRIDO EN CV
# Combina PR-AUC con métricas operativas por threshold
# =====================================================
        roc_cv = roc_auc_score(y_train, y_prob_cv)
        pr_cv = average_precision_score(y_train, y_prob_cv)

        thresholds_df = evaluar_por_umbrales(
            y_true=y_train,
            y_prob=y_prob_cv,
            thresholds=thresholds
        )

        # Threshold provisional solo diagnóstico
        threshold_tmp_cv, fila_thr = seleccionar_umbral_balanceado(
            resultados_df=thresholds_df,
            recall_min=recall_min,
            errorI_max=errorI_max
        )

        # Score independiente del threshold  ###################### score
        score = (
                1.00 * fila_thr["mcc"]
                + 0.60 * fila_thr["f1_score"]
                + 0.30 * pr_cv
                + 0.10 * fila_thr["recall"]
                - 0.15 * fila_thr["error_tipo_I"]
                - 0.10 * fila_thr["error_tipo_II"]
        )

        if not np.isfinite(score):
            print("Score inválido. Configuración omitida.")
            continue

        resumen_modelos.append({
            "config_id": i,
            "score_hibrido": score,
            "threshold_tmp_cv": threshold_tmp_cv,
            "roc_auc_cv": roc_cv,
            "pr_auc_cv": pr_cv,
            "precision_tmp_cv": fila_thr["precision"],
            "recall_tmp_cv": fila_thr["recall"],
            "f1_tmp_cv": fila_thr["f1_score"],
            "mcc_tmp_cv": fila_thr["mcc"],
            "error_tipo_I_tmp_cv": fila_thr["error_tipo_I"],
            "error_tipo_II_tmp_cv": fila_thr["error_tipo_II"],
            "TP_tmp_cv": fila_thr["TP"],
            "FN_tmp_cv": fila_thr["FN"],
            "FP_tmp_cv": fila_thr["FP"],
            "TN_tmp_cv": fila_thr["TN"],
            **params
        })

        print("ROC-AUC CV:", round(roc_cv, 4))
        print("PR-AUC CV:", round(pr_cv, 4))
        print("Threshold provisional CV:", threshold_tmp_cv)
        print("Precision provisional CV:", round(fila_thr["precision"], 4))
        print("Recall provisional CV:", round(fila_thr["recall"], 4))
        print("F1 provisional CV:", round(fila_thr["f1_score"], 4))
        print("MCC provisional CV:", round(fila_thr["mcc"], 4))
        print("Error I provisional CV:", round(fila_thr["error_tipo_I"], 4))
        print("Error II provisional CV:", round(fila_thr["error_tipo_II"], 4))
        print("score_hibrido:", round(score, 4))

        # Guardar mejor configuración por score híbrido
        if (
            mejor_config is None
            or score > mejor_score
            or (np.isclose(score, mejor_score) and roc_cv > mejor_roc_cv)
        ):
            mejor_score = score
            mejor_roc_cv = roc_cv

            mejor_config = params.copy()
            mejor_config["score_hibrido"] = score
            mejor_config["roc_auc_cv"] = roc_cv
            mejor_config["pr_auc_cv"] = pr_cv

            mejor_thresholds_df = thresholds_df.copy()
            mejor_y_prob_cv = y_prob_cv.copy()

    if len(resumen_modelos) == 0:
        raise RuntimeError(
            "No se pudo ajustar ninguna configuración válida de XGBoost."
        )

    resumen_df = pd.DataFrame(resumen_modelos).sort_values(
    by=["score_hibrido", "roc_auc_cv"],
    ascending=False
    )

    print("\n" + "="*90)
    print("TOP 10 CONFIGURACIONES XGBOOST V2")
    print("="*90)
    print(resumen_df.head(10).round(4))

    print("\n===== MEJOR CONFIGURACIÓN POR SCORE HÍBRIDO =====")
    print(mejor_config)

    if mejor_config is None or mejor_thresholds_df is None:
        raise RuntimeError(
            "No se guardó ninguna configuración válida."
        )

    print("\n===== MÉTRICAS POR UMBRAL DE LA MEJOR CONFIGURACIÓN EN CV =====")
    print(mejor_thresholds_df.round(4))

    # =====================================================
    # ETAPA 2: SELECCIÓN DEL THRESHOLD DESPUÉS DEL MODELO
    # =====================================================
    threshold_final_cv, fila_threshold_final = seleccionar_umbral_balanceado(
        resultados_df=mejor_thresholds_df,
        recall_min=recall_min,
        errorI_max=errorI_max
    )

    mejor_config["threshold_cv"] = threshold_final_cv

    print("\n===== THRESHOLD FINAL SELECCIONADO DESDE CV =====")
    print("Threshold final:", threshold_final_cv)
    print(fila_threshold_final.round(4))

    # -------------------------------------------------
    # Entrenar modelo final con todo train
    # -------------------------------------------------
    print("\n" + "="*90)
    print("ENTRENANDO XGBOOST FINAL CON MEJOR CONFIGURACIÓN")
    print("="*90)

    winsor_final = PercentileWinsorizer(
        feature_names=vars_modelo,
        cuts_by_variable=cuts_winsor,
        default_limits=(0.01, 0.99)
    )

    X_train_win = winsor_final.fit_transform(X_train)
    X_val_win = winsor_final.transform(X_val)

    if mejor_config["usar_yeojohnson"]:
        pt_final = PowerTransformer(
            method="yeo-johnson",
            standardize=True
        )

        X_train_model = pt_final.fit_transform(X_train_win)
        X_val_model = pt_final.transform(X_val_win)
    else:
        pt_final = None
        X_train_model = X_train_win.values
        X_val_model = X_val_win.values

    spw_final = generar_spw_candidatos(y_train)[mejor_config["spw_key"]]

    modelo_final = crear_xgb_modelo_v2(
        params=mejor_config,
        scale_pos_weight=spw_final,
        random_state=random_state
    )

    modelo_final.fit(
        X_train_model,
        y_train.values
    )

    threshold_final = mejor_config["threshold_cv"]

    y_prob_val = modelo_final.predict_proba(X_val_model)[:, 1]
    y_pred_val = (y_prob_val >= threshold_final).astype(int)

    cm_val = confusion_matrix(y_val, y_pred_val, labels=[1, 0])

    tp = cm_val[0, 0]
    fn = cm_val[0, 1]
    fp = cm_val[1, 0]
    tn = cm_val[1, 1]

    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_prob_val)
    pr_auc_val = average_precision_score(y_val, y_prob_val)

    error_tipo_I_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    error_tipo_II_val = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    print("\n===== XGBOOST V2 FINAL - MÉTRICAS EN VALIDATION =====")
    print("Threshold elegido desde CV:", threshold_final)
    print("scale_pos_weight final:", round(spw_final, 4))
    print("spw_key:", mejor_config["spw_key"])
    print("Usa Yeo-Johnson:", mejor_config["usar_yeojohnson"])
    print("Precision:", round(precision_val, 4))
    print("Recall:", round(recall_val, 4))
    print("F1-score:", round(f1_val, 4))
    print("MCC:", round(mcc_val, 4))
    print("Error tipo I:", round(error_tipo_I_val, 4))
    print("Error tipo II:", round(error_tipo_II_val, 4))
    print("ROC-AUC:", round(roc_auc_val, 4))
    print("PR-AUC:", round(pr_auc_val, 4))

    cm_df = pd.DataFrame(
        cm_val,
        index=["Real 1", "Real 0"],
        columns=["Predicho 1", "Predicho 0"]
    )

    print("\n===== MATRIZ DE CONFUSIÓN VALIDATION XGBOOST V2 =====")
    print(cm_df)

    return {
        "modelo_final": modelo_final,
        "winsor_final": winsor_final,
        "yeojohnson_final": pt_final,
        "mejor_config": mejor_config,
        "resumen_tuning": resumen_df,
        "resultados_thresholds_cv": mejor_thresholds_df,
        "cm_validation": cm_df,
        "y_prob_validation": y_prob_val,
        "y_pred_validation": y_pred_val,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "variables": vars_modelo
    }


# In[36]:


res_xgb_24_ensayo_conservador = correr_xgboost_tuning_v2(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    random_state=42,
    n_splits=5,
    n_iter=50,
    recall_min=0.70,
    errorI_max=0.17
)


# In[ ]:


#################################### #################### 


# In[ ]:


if thresholds is None:
    thresholds = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
                  0.35, 0.30, 0.28, 0.25, 0.20, 0.15, 0.10]


# In[ ]:


res_gam_24_v3 = correr_gam_basico(
    df=pre_23,
    target="riesgo_24",
    vars_modelo=vars_modelo,
    cuts_winsor=cuts_winsor,
    threshold_elegido=0.30,
    random_state=42,
    n_splits=5,
    n_splines=5,
    lam=200,
    max_iter=5000
)

