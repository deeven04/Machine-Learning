import numpy as np


X_seen=np.load('X_seen.npy', encoding = 'bytes', allow_pickle= True) 
Xtest=np.load('Xtest.npy', encoding = 'bytes', allow_pickle= True)
Ytest=np.load('Ytest.npy', encoding = 'bytes', allow_pickle= True)
class_attributes_seen=np.load('class_attributes_seen.npy', encoding = 'bytes', allow_pickle= True)
class_attributes_unseen=np.load('class_attributes_unseen.npy', encoding = 'bytes', allow_pickle= True)


#function to predict the classes
def predict(mean_UnseenClass, Xtest, Ytest, weight):
    acc = 0. #accuracy
    dist = np.zeros((Ytest.shape[0], mean_UnseenClass.shape[0]))
    for i in range(mean_UnseenClass.shape[0]):
        diff = mean_UnseenClass[i] - Xtest
        sq = np.square(diff)
        d = np.dot(sq, weight)
        dist[:, i] = d.reshape(d.shape[0],)

    predicted_y = np.argmin(dist, axis=1)
    predicted_y = predicted_y.reshape(predicted_y.shape[0],1)
    predicted_y+=1
    acc = 1 - np.count_nonzero(predicted_y-Ytest)/float(Ytest.shape[0])
    return predicted_y, acc


#finding weights
def find_weights(u, x_seen, weight):
    diff = u - x_seen
    sq = np.sum(np.square(diff), axis=0).reshape(diff.shape[1], 1)
    weight/=np.sum(weight)

    return weight



#calculating mean of seen classes
mean_SeenClass = np.zeros((X_seen.shape[0], X_seen[0].shape[1]))
for i in range(0, X_seen.shape[0]):
    mean_SeenClass[i] = (np.mean(X_seen[i], axis=0)).reshape(1, X_seen[0].shape[1])

weight = np.ones((mean_SeenClass.shape[1], 1))
weight/=np.sum(weight)

for i in range(0, 40):
    weight = find_weights(mean_SeenClass[i], X_seen[i], weight)

# Computing the similarity of each unseen class with each of the seen classes
dot_product = np.dot(class_attributes_unseen, class_attributes_seen.T)
row_sum = np.sum(dot_product, axis=1, keepdims=True)
similarity_matrix = dot_product / row_sum

#calculating mean of unseen classes
mean_UnseenClass = np.dot(similarity_matrix, mean_SeenClass)

predicted_y, acc = predict(mean_UnseenClass, Xtest, Ytest, weight)

print(100*acc)
