import csv
import math
from collections import Counter

class DataPoint:
    def __init__(self, label, features):
        self.label = label
        self.features = features


# Load CSV file
def load_file(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features = list(map(float, row[:-1]))
            label = row[-1]
            data.append(DataPoint(label, features))
    return data


# Calculate Euclidean distance
def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# KNN classification
def knn_classify(train_set, test_vector, k):
    distances = []
    for dp in train_set:
        dist = euclidean_distance(dp.features, test_vector)
        distances.append((dist, dp.label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    k_nearest = [label for _, label in distances[:k]]

    # Find most common label
    most_common = Counter(k_nearest).most_common(1)[0][0]
    return most_common


# Calculate accuracy
def accuracy(train_set, test_set, k):
    correct = 0
    for dp in test_set:
        predicted = knn_classify(train_set, dp.features, k)
        if predicted == dp.label:
            correct += 1
    return (correct / len(test_set)) * 100


def main():
    import sys

    if len(sys.argv) != 4:
        print("Usage: python knn.py <k> <train_file> <test_file>")
        return

    try:
        k = int(sys.argv[1])
        train_file = sys.argv[2]
        test_file = sys.argv[3]

        train_set = load_file(train_file)
        test_set = load_file(test_file)

        acc = accuracy(train_set, test_set, k)
        print(f"The accuracy is: {acc:.2f}%")

        # Interactive testing loop
        while True:
            user_input = input("Enter a new feature vector separated by commas: ")
            test_vector = list(map(float, user_input.strip().split(',')))
            classification = knn_classify(train_set, test_vector, k)
            print(f"The class of the feature vector is: {classification}")

            again = input("Do you want to test another feature vector? (y/n): ").strip().lower()
            if again != 'y':
                print("Exiting...")
                break

    except ValueError:
        print("Please enter numeric values correctly.")
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
