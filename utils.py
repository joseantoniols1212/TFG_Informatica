import numpy as np

def proyection(point: np.array):
    return point[1:]


def map_to_sphere(point: np.array) -> np.array:
    first_cordinate = 1-sum(point)
    return np.append([first_cordinate], point)

# Function to parse the data file
def parse_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract number of assets
    num_assets = int(lines[0])
    
    # Initialize arrays
    mean_returns = np.zeros(num_assets)
    std_deviations = np.zeros(num_assets)
    correlations = np.zeros((num_assets, num_assets))
    
    # Parse mean returns and standard deviations
    for i in range(1, num_assets + 1):
        mean_returns[i - 1], std_deviations[i - 1] = map(float, lines[i].split())
    
    # Parse correlations
    for line in lines[num_assets + 1:]:
        i, j, corr = map(float, line.split())
        correlations[int(i) - 1, int(j) - 1] = corr
        correlations[int(j) - 1, int(i) - 1] = corr  # Since correlation matrix is symmetric
    
    return num_assets, mean_returns, std_deviations, correlations

# Function to parse the solution file
def parse_solution(filename):
    print(filename)
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize arrays
    num_points = len(lines)
    return_ = np.zeros(num_points)
    variance = np.zeros(num_points)

    print(num_points)
    for i in range(num_points):
        return_[i], variance[i] = map(float, lines[i].split())
    
    return return_, variance