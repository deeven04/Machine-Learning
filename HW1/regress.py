import numpy as np


X_seen=np.load('X_seen.npy', encoding = 'bytes', allow_pickle= True) 
Xtest=np.load('Xtest.npy', encoding = 'bytes', allow_pickle= True)
Ytest=np.load('Ytest.npy', encoding = 'bytes', allow_pickle= True)
class_attributes_seen=np.load('class_attributes_seen.npy', encoding = 'bytes', allow_pickle= True)
class_attributes_unseen=np.load('class_attributes_unseen.npy', encoding = 'bytes', allow_pickle= True)


def predict(mean_UnseenClass, Xtest, Ytest, weight):
    acc = 0.  #accuracy
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



#calculating mean of seen classes
mean_SeenClass = np.zeros((X_seen.shape[0], X_seen[0].shape[1]))
for i in range(0, X_seen.shape[0]):
    mean_SeenClass[i] = (np.mean(X_seen[i], axis=0)).reshape(1, X_seen[0].shape[1])

weight = np.ones((mean_SeenClass.shape[1], 1))

# Test Class
for i in [0.01, 0.1, 1, 10, 20, 50, 100]:
    #calculating mean of unseen classes
    W1 = np.dot(class_attributes_seen.T, class_attributes_seen) + i*(np.eye(class_attributes_seen.shape[1]))
    W2 = np.dot(class_attributes_seen.T, mean_SeenClass)
    W = np.dot(np.linalg.inv(W1), W2)
    mean_UnseenClass = np.dot(class_attributes_unseen, W)

    predicted_y, acc = predict(mean_UnseenClass, Xtest, Ytest, weight)

    print("Test accuracy for lambda = " + str(i) + " is: " + str(100*acc))