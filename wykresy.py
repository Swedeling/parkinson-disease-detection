# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# dir_path = "C:/Users/User/Desktop/wyniki końcowe/a"
# #
# # for model in ["xception", "vgg16", "resnet", "mobile", "inception"]:
# print("==== Xception ====")
# filename = f"history_xception_a.xlsx"
# history_path = os.path.join(dir_path, filename)
# df = pd.read_excel(history_path)
# #
# loss_xception = df["loss"]
# val_loss_xception = df["val_loss"]
# acc_xception = df["loss"]
# val_acc_xception = df["val_accuracy"]
#
# print("==== VGG16 ====")
# filename = f"history_vgg16_a.xlsx"
# history_path = os.path.join(dir_path, filename)
# df = pd.read_excel(history_path)
#
# loss_vgg = df["loss"]
# val_loss_vgg = df["val_loss"]
# acc_vgg = df["loss"]
# val_acc_vgg = df["val_accuracy"]
#
# print("==== ResNet50 ====")
# filename = f"history_resnet50_a_3.xlsx"
# history_path = os.path.join(dir_path, filename)
# df = pd.read_excel(history_path)
#
# loss_resnet = df["loss"]
# val_loss_resnet = df["val_loss"]
# acc_resnet = df["loss"]
# val_acc_resnet = df["val_accuracy"]
# #
# print("==== MobileNet ====")
# filename = f"history_mobile_a.csv"
# history_path = os.path.join(dir_path, filename)
# df = pd.read_csv(history_path)
# #
# loss_mobile= df["loss"]
# val_loss_mobile = df["val_loss"]
# acc_mobile = df["loss"]
# val_acc_mobile = df["val_accuracy"]
# #
# print("==== Inception ====")
# filename = f"history_inception_a.xlsx"
# history_path = os.path.join(dir_path, filename)
# df = pd.read_excel(history_path)
#
# loss_inception= df["loss"]
# val_loss_inception = df["val_loss"]
# acc_inception = df["loss"]
# val_acc_inception = df["val_accuracy"]
#
#
# COLORS = sns.color_palette('flare')
# #
#
# plt.figure(figsize=(8, 6))
#
# lista1 = val_loss_xception
# lista2 = val_loss_inception
# lista3 = val_loss_mobile
# lista4 = val_loss_resnet
# lista5 = val_loss_vgg
#
# plt.plot(lista1, label='Xception', color=COLORS[0])
# plt.plot(lista2, label='InceptionV3', color=COLORS[1])
# # plt.plot(lista3, label='MobileNet', color=COLORS[2])
# plt.plot(lista4, label='ResNet50', color=COLORS[3])
# plt.plot(lista5, label='VGG16', color=COLORS[4])
#
# plt.ylim([0.3, 0.8])
# plt.xlabel('Epoka')
# plt.ylabel('Strata')
# plt.legend()
# plt.show()
#
# #
# plt.figure(figsize=(8, 6))
#
# lista1 = val_acc_xception
# lista2 = val_acc_inception
# lista3 = val_acc_mobile
# lista4 = val_acc_resnet
# lista5 = val_acc_vgg
#
# plt.plot(lista1, label='Xception', color=COLORS[0])
# plt.plot(lista2, label='InceptionV3', color=COLORS[1])
# # plt.plot(lista3, label='MobileNet', color=COLORS[2])
# plt.plot(lista4, label='ResNet50', color=COLORS[3])
# plt.plot(lista5, label='VGG16', color=COLORS[4])
#
# plt.ylim([0.0, 0.9])
# plt.xlabel('Epoka')
# plt.ylabel('Dokładność')
# plt.legend()
# plt.show()


# plt.plot(df['accuracy'], label='Dokładność treningowa', color=COLORS[2])
# plt.plot(df['val_accuracy'], label='Dokładność walidacyjna', color=COLORS[0])
# plt.title('Dokładność')
# plt.xlabel('Epoka')
# plt.ylabel('Dokładność')
# plt.legend()
# plt.show()
#
# # Tworzenie wykresu funkcji straty treningowej i walidacyjnej
# plt.plot(df['loss'], label='Strata treningowa', color=COLORS[2])
# plt.plot(df['val_loss'], label='Strata walidacyjna', color=COLORS[0])
# plt.title('Funkcja Straty')
# plt.xlabel('Epoka')
# plt.ylabel('Strata')
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_excel("db.xlsx")
df.dropna(inplace=True)
# df['class'] = df['label'].replace({1: 'PD', 0: 'HS'})
df['Klasa'] = df["class"]
df['Płeć'] = df['gender'].replace({"K": "Kobiety", "M": "Mężczyźni"})
df['Język'] = df['language'].replace({"spanish": "hiszpański", "polish": "polski", "italian": "włoski"})

print(df.head())
plt.style.use('seaborn-whitegrid')
colors = sns.color_palette('flare')

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# fig.suptitle("Summary of dataset", fontsize=16)

# Language distribution
language_counts = df['Język'].value_counts()
axs[0, 0].pie(language_counts, labels=language_counts.index, colors=[colors[1], colors[3], colors[4]], autopct='%1.1f%%', startangle=140)
axs[0, 0].set_title("Rozkład języków ojczystych wśród osób w bazie danych")
for wedge in axs[0, 0].patches:
    wedge.set_edgecolor('black')

# axs[0, 0].title('Rozkład języków')
# axs[0, 0].axis('equal')  # Ustala równy rozmiar osi X i Y, aby uzyskać wykres kołowy

    # # Gender distribution
    # axs[0, 0].bar(df['gender'].value_counts().index, df['gender'].value_counts().values,
    #               color=[colors[0], colors[1]], edgecolor='black', linewidth=1.2)
    # axs[0, 0].set_xlabel('Gender')
    # axs[0, 0].set_ylabel('Count')
    # axs[0, 0].set_title("Gender distribution")
    #
    # # Class distribution
    # axs[1, 0].bar(df['class'].value_counts().index, df['class'].value_counts().values,
    #               color=[colors[2], colors[3]], edgecolor='black', linewidth=1.2)
    # axs[1, 0].set_xlabel('Class')
    # axs[1, 0].set_ylabel('Count')
    # axs[1, 0].set_title("Class distribution")

# Class and gender distribution
counts = df.groupby(['class', 'Płeć']).size().unstack()
counts.plot(kind='bar', stacked=False, ax=axs[0, 1], color=[colors[1], colors[3]], edgecolor='black', linewidth=1.2)
for p in axs[0, 1].patches:
    axs[0, 1].annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                       (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                       textcoords='offset points')

axs[0, 1].set_xlabel('Klasa')
axs[0, 1].set_ylabel('Liczba osób')
axs[0, 1].set_title("Rozkład płci w klasach")
axs[0, 1].set_ylim(0, 90)


# Age and class distribution
bins = np.arange(50, 100, 10)
age_labels = ['50-59', '60-69', '70-79', '80-89', '90+']
df['category'] = np.digitize(df.age, bins, right=True)
df.loc[df['age'] < 50, 'category'] = -1

age_labels.insert(0, "< 50")

print(df.head(50))
counts = df.groupby(['category', 'Klasa']).age.count().unstack()
counts.plot(kind='bar', stacked=False, ax=axs[1, 1], color=[colors[1], colors[3]], edgecolor='black', linewidth=1.2)
for p in axs[1, 1].patches:
    axs[1, 1].annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                       (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                       textcoords='offset points')

axs[1, 1].set_xlabel('Grupa wiekowa')
axs[1, 1].set_ylabel('Liczba osób')
axs[1, 1].set_title("Rozkład klas w grupach wiekowych")
axs[1, 1].set_xticklabels(age_labels)
axs[1, 1].set_ylim(0, 50)
fig.tight_layout()
plt.show()


fig, ax = plt.subplots()
counts.plot(kind='bar', stacked=False, ax=ax, color=[colors[1], colors[3]], edgecolor='black', linewidth=1.2)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                       (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                       textcoords='offset points')
ax.set_xticklabels(age_labels)
ax.set_ylim(0, 50)
ax.set_xlabel('Grupa wiekowa')
ax.set_ylabel('Liczba osób')
# axs[1, 1].set_xlabel('Grupa wiekowa')
# axs[1, 1].set_ylabel('Liczba osób')
# axs[1, 1].set_title("Rozkład klas w grupach wiekowych")
# axs[1, 1].set_xticklabels(age_labels)
# axs[1, 1].set_ylim(0, 50)
fig.tight_layout()
plt.savefig("latex/img/database stats/age_distribution.png", dpi=300)
plt.show()



# plt.figure()
# plt.bar(language_counts, labels=language_counts.index, colors=[colors[1], colors[3], colors[4]], autopct='%1.1f%%', startangle=140)
# # plt.title("Rozkład języków ojczystych wśród osób w bazie danych")
# plt.savefig("latex/img/database stats/languages_distribution.png", dpi=300)
# plt.show()
# plt.close()
# # for wedge in axs[0, 0].patches:
# #     wedge.set_edgecolor('black')


fig, ax = plt.subplots()
counts = df.groupby(['class', 'Płeć']).size().unstack()
counts.plot(kind='bar', stacked=False, ax=ax, color=[colors[1], colors[3]], edgecolor='black', linewidth=1.2)

for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                textcoords='offset points')

ax.set_xlabel('Klasa')
ax.set_ylabel('Liczba osób')
# ax.set_title("Rozkład płci w klasach")
ax.set_ylim(0, 80)
plt.savefig("latex/img/database stats/gender_distribution.png", dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots()
counts = df.groupby(['class', 'Język']).size().unstack()
counts.plot(kind='bar', stacked=False, ax=ax, color=[colors[1], colors[3], colors[4]], edgecolor='black', linewidth=1.2)

for p in ax.patches:
    ax.annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                textcoords='offset points')

ax.set_xlabel('Klasa')
ax.set_ylabel('Liczba osób')
# ax.set_title("Rozkład płci w klasach")
ax.set_ylim(0, 80)
plt.savefig("latex/img/database stats/languages_distribution.png", dpi=300)
plt.show()
plt.close()

