import numpy as np
import matplotlib.pyplot as plt

def test_gaussian_histogram():
    # Generate 1000 random samples from a Gaussian distribution
    data = np.random.normal(loc=0, scale=1, size=1000)
    
    # Create a histogram
    plt.figure()
    plt.hist(data, bins=30, color='blue', alpha=0.7)
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    # Save the plot to a file
    plt.savefig("gaussian_histogram.png")
    plt.close()

if __name__ == '__main__':
    test_gaussian_histogram()