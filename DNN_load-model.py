from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

model = load_model('DNN_9292.h5')

y_pred = model.predict(test_X)
y_pred = np.argmax(y_pred, axis=1)

matrix = confusion_matrix(integer_encoded_test_y, y_pred)

accuracy = accuracy_score(integer_encoded_test_y, y_pred)
precision_score = precision_score(integer_encoded_test_y, y_pred, average='weighted')
recall_score = recall_score(integer_encoded_test_y, y_pred, average='macro')
f1_score = f1_score(integer_encoded_test_y, y_pred, average='weighted')

print("Accuracy : " + str(accuracy))
print("Precision : " + str(precision_score))
print("Recall : " + str(recall_score))
print("f1_score : " + str(f1_score))


plt.clf()
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('P-0s', 'P-1s','P-2s'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('A-0s', 'A-1s','A-2s'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='red')

plt.savefig('CM_LR_Predicted_class3.png')
plt.show()
