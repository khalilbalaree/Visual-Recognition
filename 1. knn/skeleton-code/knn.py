def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    # Convert data from numpy arrays to pytorch tensors
    x = torch.tensor(x_train, dtype=torch.float, device=device)
    tests = torch.tensor(x_test, dtype=torch.float, device=device)
    y = torch.tensor(y_train, dtype=torch.uint8, device=device)
    
    # Choose a distance function and a value for k
    k = 3

    # Initial the final result
    y_test = np.zeros(x_test.shape[0])
    count = 0

    # For each test image:
    for test in tests:
      # Calculate the distance between the test image and all training images.
      distance = torch.norm(test-x, p=2, dim=1)
      # Find indices of k training images with the smallest distances
      _, index = torch.topk(distance, k, largest=False)
      # Get classes of the corresponding training images
      classes = torch.gather(y, 0, index)

      # Represent classes as one-hot vectors and stack into a 𝑘 × 10 array
      hot_array = np.zeros((k, n_classes))
      for i in range(k):
        hot_array[i][classes[i]] = 1
      # Compute column-wise sum of this array
      hot_one = np.argmax(hot_array.sum(axis=0))
      # Take the column with the maximum sum
      y_test[count] = hot_one

      count += 1

    return y_test