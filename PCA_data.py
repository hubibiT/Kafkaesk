
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data_path = '/Users/hubitragenap/Documents/Uni/Python/DIHU1/'
filen_name = 'data_synAnalysis21.npy'


work_names=['doeblin_butterblume', 'heinrichmann_rassen' , 'heinrichmann_unrat', 'heinrichmann_untertan','hesse_demian', 'hesse_klingsors', 'hesse_steppenwolf', 'hesse_untermrad', 'kafka_prozess', 'kafka_schloss', 'kafka_strafkolonie', 'kafka_verwandlung', 'mann_buddenbrooks1', 'mann_buddenbrooks2', 'mann_kroeger', 'musil_verwirrungen', 'rilke_pragergeschichten', 'schnitzler_casanova', 'schnitzler_taenzerin', 'schnitzler_traumnovelle', 'zweig_geheimnis']# 'modvvg1908'   #law
 



colors= ['C0']*1 + ['C1'] * 3 + ['C2']*4 + ['C4']*4 + ['C5']*3 + ['C6']*2 + ['C7']*3 +['C8']*1

data_wc =np.load(data_path+filen_name)
data_wordcount =  StandardScaler().fit_transform(data_wc)

#plot data as image (similar to line plots)
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
plt.imshow(data_wordcount, aspect='auto', interpolation='none')
plt.colorbar(label='Z-Score')
plt.xlabel(' Z-scored Features')
plt.ylabel('Werk')

x1 = [0,1,2,3,4,5,6,7]
features = ['ASL', 'APN' , 'ASN' , 'VPF' , 'ZPF' , 'KIF' , 'KIIF' , 'RWF']
plt.xticks(x1, features, rotation=45)
x2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19, 20]
plt.yticks(x2, work_names)

plt.show()


#%% PCA
pca_words= PCA()
pca_words = pca_words.fit(data_wordcount)   #trainiert PCA (finds components)
projection_wordcount=pca_words.transform(data_wordcount)



fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
for idx, (color, work) in enumerate(zip(colors, work_names)):
    plt.scatter(projection_wordcount[idx,0], projection_wordcount[idx,1],
                color=color, marker='x',
            )

    plt.text(projection_wordcount[idx,0], projection_wordcount[idx,1],
             work,
                color=color, ha='center', va='bottom')

plt.xlabel('PC  1')
plt.ylabel('PC 2')
plt.margins(x=0.25, y=0.05)

plt.show()

#in PC1/PC2 kann man gut zwischen Verfassungen und Literatur unterscheiden. Warum?
#--> look at which words are important for PC1 and PC2

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
plt.imshow(pca_words.components_, aspect='auto', interpolation='none')
plt.colorbar(label='magnitude')
plt.xlabel(' Z-Scored Features')
# plt.xticks(x1, features, rotation=45)
plt.ylabel('Principal component')
plt.show()

##where has PC1 largerst contribution:
all_pc_components= pca_words.components_
word_indices_sorted = np.argsort(all_pc_components[0])[::-1]  #sort by largest values first
#words 67 and 96


print(word_indices_sorted[:30], end=',' )
####################################



