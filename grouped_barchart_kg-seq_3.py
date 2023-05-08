import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

#THIS IS USING PANDAS FOR KG + SEQ; MODEL ABLATION (KG)

# Plot configuration
mpl.style.use("seaborn-pastel")
mpl.rcParams.update(
    {
        "font.size": 14,
        "figure.facecolor": "w",
        "axes.facecolor": "w",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "xtick.top": False,
        "xtick.bottom": False,
        "ytick.right": False,
        "ytick.left": False,
    }
)






#THIS IS AUC and AUPR; 
df = pd.DataFrame({
    "": ['AUC','AUPR'],
    #"": ['AUC'],
    "ORIG_BOTH-EPOCH-100": [0.89522, 0.86919],
    "DEEPPURPOSE_RELU_NORM-EPOCH-100": [0.52597, 0.46154],
    "ALL_GRAPH_ASSOCIATIONS_W/O_SELF.PROTEIN/DRUG_EMBEDDING-EPOCH-100": [0.90002, 0.83380],

    "DRUG-DRUG/PROTEIN-PROTEIN-EPOCH-100": [0.63417, 0.60490],
    "DRUG-CHEMICAL/PROTEIN-SEQUENCE-EPOCH-100": [0.56982, 0.48365], 
    "DRUG-DISEASE/PROTEIN-DISEASE-EPOCH-100": [0.70192, 0.67805],
    "DRUG-SIDE_EFFECT/PROTEIN-SEQUENCE-EPOCH-100": [0.63198, 0.60230],
    "DRUG-PROTEIN/PROTEIN-DRUG-EPOCH-100": [0.92431, 0.92315],

    "DEEPPURPOSE + DRUG-DRUG/PROTEIN-PROTEIN-EPOCH-100": [0.67077, 0.65398],
    "DEEPPURPOSE + DRUG-CHEMICAL/PROTEIN-SEQUENCE-EPOCH-100": [0.57218, 0.48607],
    "DEEPPURPOSE + DRUG-DISEASE/PROTEIN-DISEASE-EPOCH-100": [0.72772, 0.72853],
    "DEEPPURPOSE + DRUG-SIDE_EFFECT/PROTEIN-SEQUENCE-EPOCH-100": [0.630430, 0.59460],
    "DEEPPURPOSE + DRUG-PROTEIN/PROTEIN-DRUG-EPOCH-100": [0.92790, 0.91578],
}).set_index("")


##THIS IS AUC and AUPR; 
#df = pd.DataFrame({
#    "": ['AUC', 'AUPR'],
#    #"": ['AUC'],
#    "ORIG_BOTH-EPOCH-100": [0.0, 0.0],
#    "DEEPPURPOSE_RELU_NORM-EPOCH-100": [0.0, 0.0],
#    "ALL_GRAPH_ASSOCIATIONS_W/O_SELF.PROTEIN/DRUG_EMBEDDING-EPOCH-100": [0.0, 0.0],

#    "DRUG-DRUG/PROTEIN-PROTEIN-EPOCH-100": [0.0, 0.0],
#    "DRUG-CHEMICAL/PROTEIN-SEQUENCE-EPOCH-100": [0.0, 0.0], 
#    "DRUG-DISEASE/PROTEIN-DISEASE-EPOCH-100": [0.0, 0.0],
#    "DRUG-SIDE_EFFECT/PROTEIN-SEQUENCE-EPOCH-100": [0.0, 0.0],
#    "DRUG-PROTEIN/PROTEIN-DRUG-EPOCH-100": [0.0, 0.0],

#    "DEEPPURPOSE + DRUG-DRUG/PROTEIN-PROTEIN-EPOCH-100": [0.0, 0.0],
#    "DEEPPURPOSE + DRUG-CHEMICAL/PROTEIN-SEQUENCE-EPOCH-100": [0.0, 0.0],
#    "DEEPPURPOSE + DRUG-DISEASE/PROTEIN-DISEASE-EPOCH-100": [0.0, 0.0],
#    "DEEPPURPOSE + DRUG-SIDE_EFFECT/PROTEIN-SEQUENCE-EPOCH-100": [0.0, 0.0],
#    "DEEPPURPOSE + DRUG-PROTEIN/PROTEIN-DRUG-EPOCH-100": [0.0, 0.0],
#}).set_index("")





fig = plt.figure(figsize=(10,4))
#fig = plt.figure(figsize=(20,20))
#fig = plt.figure(figsize=(15,10))
#fig = plt.figure(figsize=(15,15))

#ax = df.plot(kind='bar', figsize= (14,11), width=0.75, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B', '#CFCFCF', '#4878D0', '#956CB4', '#797979', '#D65F5F', '#FFC400']); #this
#ax = df.plot(kind='line', figsize= (14,11), lw=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B', '#CFCFCF', '#4878D0', '#956CB4', '#797979', '#D65F5F', '#FFC400']); #this
#ax = df.plot(kind='line', figsize= (14,11), lw=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6']); #this
#ax = df.plot(kind='bar', figsize= (14,11), width=0.75, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6']); #this
#ax = df.plot(kind='bar', figsize= (14,11), width=0.9, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6']); #this

#ax = df.plot(kind='bar', figsize= (20,20), width=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6']); #this
#ax = df.plot(kind='bar', figsize= (20,20), width=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF']); #this
#ax = df.plot(kind='bar', figsize= (20,20), width=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B']); #this
#ax = df.plot(kind='bar', figsize= (14,11), width=4, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B']); #this
#ax = df.plot(kind='bar', figsize= (15,15), width=5, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B', '#CFCFCF', '#4878D0', '#956CB4', '#797979']); #this

ax = df.plot(kind='bar', figsize= (15,15), width=0.75, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B', '#CFCFCF', '#4878D0', '#956CB4', '#797979', '#D65F5F', '#FFC400']); #this
#ax = df.plot(kind='bar', figsize= (15,15), width=0.75, color = ['w']); #this


for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


#ax.legend(loc='center left', bbox_to_anchor=(0.3, 1))
#ax.legend(loc='center left', bbox_to_anchor=(0.2, 0.9))
#ax.legend(loc='center left', bbox_to_anchor=(0.4, 1))
#ax.legend(loc='center left', bbox_to_anchor=(0.1, 1))

##legend = plt.legend(frameon = 1)
##frame = legend.get_frame()
#frame.set_color('white')
##frame.set_facecolor('white')
#frame.set_edgecolor('red')
#ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.05))

#plt.show()

#ax.set_ylabel("test AUC", fontsize=22)
#ax.set_xlabel("test AUPR", fontsize=22)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

#plt.figure(figsize=(20,5))

#ax.legend_ = None

#plt.ylim(0.02, 0.09)

plt.ylim(0.4)

file_name = 'figures/lab_kg-seq_1'
#plt.savefig(file_name + '_test_aupr.png')
plt.savefig(file_name + '_test_auc.png')

plt.show()
