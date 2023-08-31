import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fashion_mnist-Datensatz wird aus keras.datasets geladen
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
plt.figure()
plt.imshow(train_images[100])
plt.colorbar()
plt.show

# Labels mit den folgenden Namen benennen
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_labels)
train_images.shape 
len(train_labels) 
len(test_labels)

# Vorverarbeitung der Daten

train_images=train_images/255
test_images=test_images/255

plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.xlabel(class_names[train_labels[0]])
plt.show()

# Das Modell wird erstellt

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print("accuracy", test_acc)
print("loss", test_loss)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(test_images)
predictions = probability_model.predict(test_images)
print("prediction: ", predictions[0])
print("correct label", test_labels[0])

np.argmax(predictions[0])

print("correct label", test_labels[0])

# plot_image-Funktion definieren, um die Vorhersage darzustellen
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    
    # Binäres Bild auf dem Plot anzeigen
    plt.imshow(img, cmap = plt.cm.binary)
    
    # Wenn "predicted label" gleich "true_label" ist, setzt man die Farbe blau, sonst setzt man die Farbe rot.
    # Diese Farbe benutzen, um xlabel für den vorhergesagten Klassennamen und den wahren Klassennamen zu schreiben.
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                              100*np.max(predictions_array),
                              class_names[true_label]),
                              color=color)

# Vorhersagen für 25 Bilder aus test_images zeigen
for i in range(25):
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.show()