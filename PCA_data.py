
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



#%% Linear discriminant analysis to distinguish between authors
#Wenn es die option #multi_class='ovr' gibt, dann kann man mit der Methode lernen,
#wie man diesen Author am besten von allen anderen unterscheiden kann.

# author_classes = np.asarray([1]*1 + [2]*4 + [3]*3 + [4]*4 + [5]*6)


# LDA_authors = da.LinearDiscriminantAnalysis(    )
# LDA_authors = LDA_authors.fit(data_wordcount, author_classes)
# projections_lda = LDA_authors.transform(data_wordcount)


# idx_dim1=0
# idx_dim2=1
# fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(5,5))
# for idx, (color, work) in enumerate(zip(colors, work_names)):
#     plt.scatter(projections_lda[idx,idx_dim1], projections_lda[idx,idx_dim2],
#                 color=color, marker='x',
#             )

#     plt.text(projections_lda[idx,idx_dim1], projections_lda[idx,idx_dim2],
#              work,
#                 color=color, ha='center', va='bottom')

# plt.xlabel('LDA {}'.format(idx_dim1+1))
# plt.ylabel('LDA {}'.format(idx_dim2+1))
# plt.margins(x=0.25, y=0.05)
# plt.show()

# #What are the words based on which we can discrimiinate authors?
# all_ld_components= LDA_authors.scalings_
# fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
# plt.imshow(all_ld_components.T, aspect='auto', interpolation='none')
# plt.colorbar(label='magnitude')
# plt.xlabel('words')
# plt.ylabel('Discrimination dimensions')
# plt.show()

# ##where has PC1 largerst contribution:

# word_indices_sorted = np.argsort(all_ld_components[:,2])[::-1]  #sort by largest values first
# print(word_indices_sorted[:15])
# #words 78, 95, 87, 75, 20, #[66 78 32 39 18], #[66 58 47 73 78]
