# ============================================================
# PART 1 : Multi-Omics Data Acquisition & Harmonization
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Dataset Path
# ------------------------------------------------------------

DATASET_PATH = "Dataset"

TRANSCRIPT = os.path.join(DATASET_PATH,"transcriptomics.csv")
CNA         = os.path.join(DATASET_PATH,"cna.csv")
MUTATION    = os.path.join(DATASET_PATH,"mutation.csv")
PROTEIN     = os.path.join(DATASET_PATH,"proteomics.csv")
LABEL        = os.path.join(DATASET_PATH,"labels.csv")

# ------------------------------------------------------------
# Read Dataset
# ------------------------------------------------------------

print("Loading Dataset...")

rna = pd.read_csv(TRANSCRIPT)
cna = pd.read_csv(CNA)
mut = pd.read_csv(MUTATION)
pro = pd.read_csv(PROTEIN)
label = pd.read_csv(LABEL)

print("RNA :",rna.shape)
print("CNA :",cna.shape)
print("Mutation :",mut.shape)
print("Protein :",pro.shape)

# ------------------------------------------------------------
# Remove duplicated samples
# ------------------------------------------------------------

rna = rna.drop_duplicates()
cna = cna.drop_duplicates()
mut = mut.drop_duplicates()
pro = pro.drop_duplicates()

# ------------------------------------------------------------
# Missing Value Imputation
# (Paper uses MissForest; SimpleImputer can be replaced later)
# ------------------------------------------------------------

print("Imputing Missing Values...")

imputer = SimpleImputer(strategy='mean')

rna = pd.DataFrame(imputer.fit_transform(rna))
cna = pd.DataFrame(imputer.fit_transform(cna))
mut = pd.DataFrame(imputer.fit_transform(mut))
pro = pd.DataFrame(imputer.fit_transform(pro))

# ------------------------------------------------------------
# Quantile Normalization
# ------------------------------------------------------------

print("Quantile Normalization...")

qt = QuantileTransformer(output_distribution='normal')

rna = qt.fit_transform(rna)
cna = qt.fit_transform(cna)
mut = qt.fit_transform(mut)
pro = qt.fit_transform(pro)

# ------------------------------------------------------------
# Z-score Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

rna = scaler.fit_transform(rna)
cna = scaler.fit_transform(cna)
mut = scaler.fit_transform(mut)
pro = scaler.fit_transform(pro)

# ------------------------------------------------------------
# Feature Fusion
# ------------------------------------------------------------

print("Fusing Multi-Omics Features...")

X = np.concatenate([rna,cna,mut,pro],axis=1)

y = label.values.ravel()

print("Final Feature Shape :",X.shape)

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train :",X_train.shape)
print("Test  :",X_test.shape)

print("\nPart 1 Completed Successfully.")

# ============================================================
# PART 2 : Multi-Omics Perturbation Intelligence Engine (MPIE)
# ============================================================

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Split fused feature matrix into modalities
# ------------------------------------------------------------

rna_dim = rna.shape[1]
cna_dim = cna.shape[1]
mut_dim = mut.shape[1]
pro_dim = pro.shape[1]

def split_modalities(X):

    rna_x = X[:,0:rna_dim]

    cna_x = X[:,rna_dim:rna_dim+cna_dim]

    mut_x = X[:,rna_dim+cna_dim:
                rna_dim+cna_dim+mut_dim]

    pro_x = X[:,rna_dim+cna_dim+mut_dim:]

    return rna_x,cna_x,mut_x,pro_x


# ------------------------------------------------------------
# MOSD
# ------------------------------------------------------------

class MOSD:

    def __init__(self):

        self.pca_gene = PCA(n_components=0.95)

        self.pca_module = PCA(n_components=0.90)

        self.pca_pathway = PCA(n_components=0.85)

    def extract(self,data):

        gene = self.pca_gene.fit_transform(data)

        module = self.pca_module.fit_transform(data)

        pathway = self.pca_pathway.fit_transform(data)

        g = np.mean(np.abs(gene),axis=1)

        m = np.mean(np.abs(module),axis=1)

        p = np.mean(np.abs(pathway),axis=1)

        perturbation = np.vstack((g,m,p)).T

        return perturbation


# ------------------------------------------------------------
# CRDM
# ------------------------------------------------------------

class CRDM:

    def dependency(self,a,b):

        score=[]

        for i in range(a.shape[0]):

            c=np.corrcoef(a[i],b[i])[0,1]

            if np.isnan(c):
                c=0

            score.append(abs(c))

        return np.array(score)


# ------------------------------------------------------------
# ABIS
# ------------------------------------------------------------

class ABIS:

    def instability(self,perturbation,discordance):

        scaler=MinMaxScaler()

        p=scaler.fit_transform(perturbation)

        d=scaler.fit_transform(
            discordance.reshape(-1,1)
        )

        instability=np.mean(p,axis=1)+d.flatten()

        instability=scaler.fit_transform(
            instability.reshape(-1,1)
        )

        return instability.flatten()


# ------------------------------------------------------------
# Initialize MPIE
# ------------------------------------------------------------

mosd=MOSD()

crdm=CRDM()

abis=ABIS()

# ------------------------------------------------------------
# Split Modalities
# ------------------------------------------------------------

rna_train,cna_train,mut_train,pro_train=split_modalities(X_train)

rna_test,cna_test,mut_test,pro_test=split_modalities(X_test)

print("Running MOSD...")

rna_p=mosd.extract(rna_train)

cna_p=mosd.extract(cna_train)

mut_p=mosd.extract(mut_train)

pro_p=mosd.extract(pro_train)

print("Running CRDM...")

d1=crdm.dependency(rna_train,cna_train)

d2=crdm.dependency(rna_train,mut_train)

d3=crdm.dependency(rna_train,pro_train)

d4=crdm.dependency(cna_train,mut_train)

d5=crdm.dependency(cna_train,pro_train)

d6=crdm.dependency(mut_train,pro_train)

discordance=np.vstack(
    (d1,d2,d3,d4,d5,d6)
).mean(axis=0)

print("Computing ABIS...")

perturbation=np.concatenate(

    [rna_p,cna_p,mut_p,pro_p],

    axis=1

)

instability=abis.instability(

    perturbation,

    discordance

)

# ------------------------------------------------------------
# MPIE Feature Representation
# ------------------------------------------------------------

MPIE_train=np.concatenate(

    [

        perturbation,

        discordance.reshape(-1,1),

        instability.reshape(-1,1)

    ],

    axis=1

)

print("MPIE Train Shape :",MPIE_train.shape)

# ------------------------------------------------------------
# Generate MPIE Features for Test Set
# ------------------------------------------------------------

rna_p=mosd.extract(rna_test)

cna_p=mosd.extract(cna_test)

mut_p=mosd.extract(mut_test)

pro_p=mosd.extract(pro_test)

d1=crdm.dependency(rna_test,cna_test)

d2=crdm.dependency(rna_test,mut_test)

d3=crdm.dependency(rna_test,pro_test)

d4=crdm.dependency(cna_test,mut_test)

d5=crdm.dependency(cna_test,pro_test)

d6=crdm.dependency(mut_test,pro_test)

discordance=np.vstack(

    (d1,d2,d3,d4,d5,d6)

).mean(axis=0)

perturbation=np.concatenate(

    [rna_p,cna_p,mut_p,pro_p],

    axis=1

)

instability=abis.instability(

    perturbation,

    discordance

)

MPIE_test=np.concatenate(

    [

        perturbation,

        discordance.reshape(-1,1),

        instability.reshape(-1,1)

    ],

    axis=1

)

print("MPIE Test Shape :",MPIE_test.shape)

print("\nMPIE Completed Successfully.")
# ============================================================
# PART 3 : Hypergraph Multi-Omics Dependency Network (HMDN)
# ============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# PGHT
# ------------------------------------------------------------

class PGHT:

    def __init__(self,k=10):

        self.k=k

    def build(self,X):

        sim=cosine_similarity(X)

        incidence=np.zeros(sim.shape)

        for i in range(sim.shape[0]):

            idx=np.argsort(sim[i])[-self.k:]

            incidence[i,idx]=1

        return incidence


# ------------------------------------------------------------
# BCHO
# ------------------------------------------------------------

class BCHO:

    def optimize(self,H,X):

        weight=[]

        scaler=MinMaxScaler()

        for i in range(H.shape[0]):

            node=np.where(H[i]==1)[0]

            if len(node)==0:

                weight.append(0)

                continue

            feature=np.mean(X[node],axis=0)

            score=np.mean(np.abs(feature))

            weight.append(score)

        weight=np.array(weight)

        weight=scaler.fit_transform(

            weight.reshape(-1,1)

        ).flatten()

        return np.diag(weight)


# ------------------------------------------------------------
# PAHA
# ------------------------------------------------------------

class PAHA:

    def attention(self,H,W,X):

        Dv=np.diag(H.sum(axis=1)+1e-6)

        De=np.diag(H.sum(axis=0)+1e-6)

        Dv_inv=np.linalg.inv(Dv)

        De_inv=np.linalg.inv(De)

        propagation=(
            Dv_inv@
            H@
            W@
            De_inv@
            H.T
        )

        embedding=propagation@X

        score=np.mean(

            np.abs(embedding),

            axis=1

        )

        score=np.exp(score)

        score=score/np.sum(score)

        embedding=embedding*score[:,None]

        return embedding


# ------------------------------------------------------------
# Initialize HMDN
# ------------------------------------------------------------

pght=PGHT(k=12)

bcho=BCHO()

paha=PAHA()

# ------------------------------------------------------------
# Hypergraph Construction
# ------------------------------------------------------------

print("Constructing Hypergraph...")

H_train=pght.build(MPIE_train)

H_test=pght.build(MPIE_test)

print("Hypergraph Shape :",H_train.shape)

# ------------------------------------------------------------
# Hyperedge Optimization
# ------------------------------------------------------------

print("Optimizing Hyperedges...")

W_train=bcho.optimize(

    H_train,

    MPIE_train

)

W_test=bcho.optimize(

    H_test,

    MPIE_test

)

# ------------------------------------------------------------
# Hypergraph Attention Embedding
# ------------------------------------------------------------

print("Learning Hypergraph Embedding...")

HMDN_train=paha.attention(

    H_train,

    W_train,

    MPIE_train

)

HMDN_test=paha.attention(

    H_test,

    W_test,

    MPIE_test

)

print("HMDN Train :",HMDN_train.shape)

print("HMDN Test :",HMDN_test.shape)

# ------------------------------------------------------------
# Fuse MPIE + HMDN Representation
# ------------------------------------------------------------

Final_train=np.concatenate(

    [

        MPIE_train,

        HMDN_train

    ],

    axis=1

)

Final_test=np.concatenate(

    [

        MPIE_test,

        HMDN_test

    ],

    axis=1

)

print("Final Train Shape :",Final_train.shape)

print("Final Test Shape :",Final_test.shape)

print("\nHMDN Completed Successfully.")
# ============================================================
# PART 4 : QBDC-Net (PCBI + QGBO)
# ============================================================

import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# PCBI
# ------------------------------------------------------------

class PCBI:

    def initialize(self,X):

        scaler=MinMaxScaler()

        X=scaler.fit_transform(X)

        perturb=np.mean(np.abs(X),axis=0)

        pathway=np.var(X,axis=0)

        stability=1/(1+np.std(X,axis=0))

        descriptor=np.vstack(

            (

                perturb,

                pathway,

                stability

            )

        ).T

        return descriptor


# ------------------------------------------------------------
# Quantum Particle
# ------------------------------------------------------------

class Particle:

    def __init__(self,n_features):

        self.position=np.random.randint(

            0,

            2,

            n_features

        )

        self.best=self.position.copy()

        self.score=0


# ------------------------------------------------------------
# Quantum PSO
# ------------------------------------------------------------

class QPSO:

    def __init__(

            self,

            particles=20,

            iterations=25

    ):

        self.particles=particles

        self.iterations=iterations


    def fitness(

            self,

            mask,

            X,

            y

    ):

        idx=np.where(mask==1)[0]

        if len(idx)<5:

            return 0

        model=RandomForestClassifier(

            n_estimators=100,

            random_state=42

        )

        score=np.mean(

            cross_val_score(

                model,

                X[:,idx],

                y,

                cv=5,

                scoring="accuracy"

            )

        )

        return score


    def optimize(

            self,

            X,

            y

    ):

        n=X.shape[1]

        swarm=[

            Particle(n)

            for _ in range(self.particles)

        ]

        global_best=None

        global_score=0


        for itr in range(self.iterations):

            print(

                "Iteration",

                itr+1,

                "/",

                self.iterations

            )


            for p in swarm:

                fit=self.fitness(

                    p.position,

                    X,

                    y

                )

                if fit>p.score:

                    p.score=fit

                    p.best=p.position.copy()

                if fit>global_score:

                    global_score=fit

                    global_best=p.position.copy()


            mbest=np.mean(

                [s.best for s in swarm],

                axis=0

            )


            for p in swarm:

                u=np.random.rand(n)

                beta=0.75

                direction=np.where(

                    np.random.rand(n)>0.5,

                    1,

                    -1

                )

                step=beta*np.abs(

                    mbest-p.position

                )*np.log(

                    1/u

                )

                pos=global_best+direction*step

                p.position=(

                    pos>

                    np.mean(pos)

                ).astype(int)


        return global_best,global_score


# ------------------------------------------------------------
# Initialize Candidate Biomarkers
# ------------------------------------------------------------

print("Initializing Biomarkers...")

pcbi=PCBI()

descriptor=pcbi.initialize(

    Final_train

)

print(

    "Descriptor Shape :",

    descriptor.shape

)

# ------------------------------------------------------------
# Quantum Optimization
# ------------------------------------------------------------

print("\nRunning QPSO...\n")

optimizer=QPSO(

    particles=20,

    iterations=25

)

best_mask,best_score=optimizer.optimize(

    Final_train,

    y_train

)

print(

    "\nBest Accuracy :",

    best_score

)

selected=np.where(

    best_mask==1

)[0]

print(

    "Selected Features :",

    len(selected)

)

# ------------------------------------------------------------
# Reduced Biomarker Dataset
# ------------------------------------------------------------

QBDC_train=Final_train[:,selected]

QBDC_test=Final_test[:,selected]

print(

    "Optimized Train :",

    QBDC_train.shape

)

print(

    "Optimized Test :",

    QBDC_test.shape

)

print("\nQGBO Completed Successfully.")
# ============================================================
# PART 5 : APSL + PADC + CCDF
# ============================================================

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC

# ------------------------------------------------------------
# APSL
# ------------------------------------------------------------

class APSL:

    def __init__(self,threshold=0.50):

        self.threshold=threshold

    def fit(self,X):

        stability=[]

        for i in range(X.shape[1]):

            feature=X[:,i]

            cv=np.std(feature)/(np.mean(np.abs(feature))+1e-8)

            score=1/(1+cv)

            stability.append(score)

        stability=np.array(stability)

        stability=MinMaxScaler().fit_transform(

            stability.reshape(-1,1)

        ).flatten()

        mask=stability>=self.threshold

        return mask,stability

# ------------------------------------------------------------
# Stability Learning
# ------------------------------------------------------------

print("Running APSL...")

apsl=APSL(threshold=0.50)

mask,stability=apsl.fit(QBDC_train)

QBDC_train=QBDC_train[:,mask]

QBDC_test=QBDC_test[:,mask]

print("Stable Biomarkers :",QBDC_train.shape[1])

# ------------------------------------------------------------
# Graph-Aware Deep Forest
# ------------------------------------------------------------

print("\nTraining Deep Forest...")

deepforest=RandomForestClassifier(

    n_estimators=500,

    max_depth=18,

    random_state=42,

    n_jobs=-1

)

deepforest.fit(

    QBDC_train,

    y_train

)

prob_df=deepforest.predict_proba(

    QBDC_test

)

# ------------------------------------------------------------
# Kernel Extreme Learning Machine
# (Implemented using RBF Kernel SVM Probability)
# ------------------------------------------------------------

print("Training KELM...")

kelm=SVC(

    kernel="rbf",

    probability=True,

    gamma="scale",

    C=5

)

kelm.fit(

    QBDC_train,

    y_train

)

prob_kelm=kelm.predict_proba(

    QBDC_test

)

# ------------------------------------------------------------
# Confidence Calibrated Decision Fusion
# ------------------------------------------------------------

print("\nConfidence Fusion...")

confidence_df=np.max(

    prob_df,

    axis=1

)

confidence_kelm=np.max(

    prob_kelm,

    axis=1

)

alpha=confidence_df/(

    confidence_df+

    confidence_kelm+

    1e-8

)

alpha=alpha.reshape(-1,1)

final_prob=(

    alpha*prob_df+

    (1-alpha)*prob_kelm

)

prediction=np.argmax(

    final_prob,

    axis=1

)

confidence=np.max(

    final_prob,

    axis=1

)

accuracy=accuracy_score(

    y_test,

    prediction

)

print("\nClassification Accuracy :",accuracy)

print("Average Confidence :",confidence.mean())

# ------------------------------------------------------------
# Store Outputs
# ------------------------------------------------------------

Predicted_Class=prediction

Prediction_Probability=final_prob

Prediction_Confidence=confidence

print("\nPADC + CCDF Completed Successfully.")
# ============================================================
# PART 6 : Evaluation + Explainability + Risk Stratification
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef,
    cohen_kappa_score
)

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------

acc = accuracy_score(y_test, Predicted_Class)

pre = precision_score(
    y_test,
    Predicted_Class,
    average='weighted'
)

rec = recall_score(
    y_test,
    Predicted_Class,
    average='weighted'
)

f1 = f1_score(
    y_test,
    Predicted_Class,
    average='weighted'
)

mcc = matthews_corrcoef(
    y_test,
    Predicted_Class
)

kappa = cohen_kappa_score(
    y_test,
    Predicted_Class
)

cm = confusion_matrix(
    y_test,
    Predicted_Class
)

TN,FP,FN,TP = cm.ravel()

specificity = TN/(TN+FP)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy     :",acc)
print("Precision    :",pre)
print("Recall       :",rec)
print("F1 Score     :",f1)
print("Specificity  :",specificity)
print("MCC          :",mcc)
print("Kappa        :",kappa)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

plt.figure(figsize=(5,5))

plt.imshow(cm,cmap='Blues')

plt.title("Confusion Matrix")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i,j],
            ha='center',
            fontsize=14
        )

plt.xlabel("Predicted")

plt.ylabel("True")

plt.colorbar()

plt.show()

# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

prob = Prediction_Probability[:,1]

fpr,tpr,_ = roc_curve(
    y_test,
    prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    linewidth=3,
    label="AUC=%.4f"%roc_auc
)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid()

plt.show()

print("ROC AUC :",roc_auc)

# ------------------------------------------------------------
# Precision Recall Curve
# ------------------------------------------------------------

precision,recall,_=precision_recall_curve(
    y_test,
    prob
)

plt.figure(figsize=(6,5))

plt.plot(
    recall,
    precision,
    linewidth=3
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid()

plt.show()

# ------------------------------------------------------------
# SHAP Explainability
# ------------------------------------------------------------

print("\nGenerating SHAP Values...")

explainer=shap.TreeExplainer(
    deepforest
)

shap_values=explainer.shap_values(
    QBDC_test
)

shap.summary_plot(
    shap_values,
    QBDC_test,
    show=False
)

plt.show()

# ------------------------------------------------------------
# Biomarker Ranking
# ------------------------------------------------------------

importance=np.mean(
    np.abs(shap_values[1]),
    axis=0
)

rank=np.argsort(
    importance
)[::-1]

print("\nTop Biomarkers")

for i in range(min(20,len(rank))):

    print(
        i+1,
        "Feature",
        selected[rank[i]],
        "Importance :",
        round(
            importance[rank[i]],
            4
        )
    )

# ------------------------------------------------------------
# Risk Stratification
# ------------------------------------------------------------

risk=[]

for p in prob:

    if p<0.30:

        risk.append("Healthy")

    elif p<0.70:

        risk.append("High Risk")

    else:

        risk.append("Breast Cancer")

report=pd.DataFrame({

    "True Label":y_test,

    "Prediction":Predicted_Class,

    "Probability":prob,

    "Confidence":Prediction_Confidence,

    "Risk":risk

})

print("\nSample Report")

print(report.head())

report.to_csv(

    "Prediction_Report.csv",

    index=False

)

print("\nPrediction_Report.csv Saved Successfully.")

print("\n==============================")
print("IMPLEMENTATION COMPLETED")
print("==============================")