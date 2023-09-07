import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dir_path = "C:/Users/User/Desktop/wyniki końcowe/a"
#
# for model in ["xception", "vgg16", "resnet", "mobile", "inception"]:
print("==== Xception ====")
filename = f"history_xception_a.xlsx"
history_path = os.path.join(dir_path, filename)
df = pd.read_excel(history_path)
#
loss_xception = df["loss"]
val_loss_xception = df["val_loss"]
acc_xception = df["loss"]
val_acc_xception = df["val_accuracy"]

print("==== VGG16 ====")
filename = f"history_vgg16_a.xlsx"
history_path = os.path.join(dir_path, filename)
df = pd.read_excel(history_path)

loss_vgg = df["loss"]
val_loss_vgg = df["val_loss"]
acc_vgg = df["loss"]
val_acc_vgg = df["val_accuracy"]

print("==== ResNet50 ====")
filename = f"history_resnet50_a_3.xlsx"
history_path = os.path.join(dir_path, filename)
df = pd.read_excel(history_path)

loss_resnet = df["loss"]
val_loss_resnet = df["val_loss"]
acc_resnet = df["loss"]
val_acc_resnet = df["val_accuracy"]
#
print("==== MobileNet ====")
filename = f"history_mobile_a.csv"
history_path = os.path.join(dir_path, filename)
df = pd.read_csv(history_path)
#
loss_mobile= df["loss"]
val_loss_mobile = df["val_loss"]
acc_mobile = df["loss"]
val_acc_mobile = df["val_accuracy"]
#
print("==== Inception ====")
filename = f"history_inception_a.xlsx"
history_path = os.path.join(dir_path, filename)
df = pd.read_excel(history_path)

loss_inception= df["loss"]
val_loss_inception = df["val_loss"]
acc_inception = df["loss"]
val_acc_inception = df["val_accuracy"]


COLORS = sns.color_palette('flare')
#

plt.figure(figsize=(8, 6))

lista1 = val_loss_xception
lista2 = val_loss_inception
lista3 = val_loss_mobile
lista4 = val_loss_resnet
lista5 = val_loss_vgg

plt.plot(lista1, label='Xception', color=COLORS[0])
plt.plot(lista2, label='InceptionV3', color=COLORS[1])
# plt.plot(lista3, label='MobileNet', color=COLORS[2])
plt.plot(lista4, label='ResNet50', color=COLORS[3])
plt.plot(lista5, label='VGG16', color=COLORS[4])

plt.ylim([0.3, 0.8])
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()

#
plt.figure(figsize=(8, 6))

lista1 = val_acc_xception
lista2 = val_acc_inception
lista3 = val_acc_mobile
lista4 = val_acc_resnet
lista5 = val_acc_vgg

plt.plot(lista1, label='Xception', color=COLORS[0])
plt.plot(lista2, label='InceptionV3', color=COLORS[1])
# plt.plot(lista3, label='MobileNet', color=COLORS[2])
plt.plot(lista4, label='ResNet50', color=COLORS[3])
plt.plot(lista5, label='VGG16', color=COLORS[4])

plt.ylim([0.0, 0.9])
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.show()


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