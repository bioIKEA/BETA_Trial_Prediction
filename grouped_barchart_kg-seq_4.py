import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

#THIS IS USING PANDAS FOR KG + SEQ; MODEL FUSION

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
    "DEEPPURPOSE_RELU_NORM-EPOCH-100": [0.52597, 0.46154],
    "ALL_GRAPH_ASSOCIATIONS_W/O_SELF.PROTEIN/DRUG_EMBEDDING-EPOCH-100": [0.90002, 0.83380],
    "ORIG_BOTH-EPOCH-100": [0.89522, 0.86919],

    "MEAN_BOTH-EPOCH-100": [0.91773, 0.90073],
    "SUM_BOTH-EPOCH-100": [0.86026, 0.77233], 
    "CONC_BOTH-EPOCH-100": [0.86745, 0.78779],
    "SVD_BOTH-EPOCH-100": [0.61465, 0.49559],
 

}).set_index("")





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

ax = df.plot(kind='bar', figsize= (15,15), width=0.75, color = ['#92C6FF', '#97F0AA', '#FF9F9A', '#D0BBFF', '#FFFEA3', '#B0E0E6', '#DEBB9B']); #this


for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


#ax.legend(loc='center left', bbox_to_anchor=(0.3, 1))
#ax.legend(loc='center left', bbox_to_anchor=(0.2, 0.9))
#ax.legend(loc='center left', bbox_to_anchor=(0.4, 1))
#ax.legend(loc='center left', bbox_to_anchor=(0.1, 1))

#legend = plt.legend(frameon = 1)
#frame = legend.get_frame()
#frame.set_color('white')
#frame.set_facecolor('white')
#frame.set_edgecolor('red')

ax.legend(loc='center left', bbox_to_anchor=(0.1, 1.05))

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
