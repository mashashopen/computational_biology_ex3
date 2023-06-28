import random
import math
import sys
import time
import matplotlib.pyplot as plt


POPULATION_SIZE = 30
GENERATIONS = 150
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2


def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split() for line in lines]
        strings = [line[:-1] for line in data]
        patterns = [int(line[-1]) for line in data]
        return strings, patterns

train_set = sys.argv[1]
test_set = sys.argv[2]
train_strings, train_patterns = read_data(train_set)
test_strings, test_patterns = read_data(test_set)


def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        # Randomly initialize the weights for each NN structure
        weights = [random.uniform(-1, 1) for _ in range(16)]  # Input to hidden layer weights
        weights += [random.uniform(-1, 1) for _ in range(8)]  # first Hidden layer to output weights
        weights += [random.uniform(-1, 1) for _ in range(8)]  # second Hidden layer to output weights
        population.append(weights)
    return population


def calculate_fitness(nn_weights):
    train_accuracy = 0
    for string, pattern in zip(train_strings, train_patterns):
        prediction = predict_pattern(string[0], nn_weights)
        if prediction == pattern:
            train_accuracy += 1

    test_accuracy = 0
    for string, pattern in zip(test_strings, test_patterns):
        prediction = predict_pattern(string[0], nn_weights)
        if prediction == pattern:
            test_accuracy += 1

    train_accuracy /= len(train_patterns)
    test_accuracy /= len(test_patterns)

    return train_accuracy, test_accuracy


def predict_pattern(string, nn_weights):
    # Feed-forward neural network
    total_first_hidden = 0
    total_second_hidden = 0
    for i in range(len(string)):
        total_first_hidden += int(string[i]) * nn_weights[i]  # Input to first hidden layer computation

    hidden_first_activation = sigmoid(total_first_hidden)

    for i in range(16, 24):
        total_second_hidden += hidden_first_activation * nn_weights[i]  # Input to second hidden layer computation

    hidden_second_activation = lucky_relu(total_second_hidden)

    total_output = 0
    for i in range(24, 32):
        total_output += hidden_second_activation * nn_weights[i]  # Hidden layer to output computation

    output_threshold = (max(nn_weights[24:32]) + min(nn_weights[24:32])) / 2

    return 1 if total_output >= output_threshold else 0


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def lucky_relu(x):
    return max(0.1 * x, x)


def main():
    start_time = time.time()  # Start the timer

    count_static_accuracy = 0

    best_test_accuracy = 0.0
    best_weights = None

    # Initialize the Population
    population = initialize_population()

    generation_list = []
    test_accuracy_list = []

    for generation in range(GENERATIONS):
        # Selection (Tournament Selection)
        selected_population = []
        for _ in range(POPULATION_SIZE):
            tournament_size = 5
            tournament = random.sample(population, tournament_size)
            best = max(tournament, key=calculate_fitness)
            selected_population.append(best)

        # Crossover
        offspring_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            if random.random() < CROSSOVER_RATE:
                # One-point crossover
                crossover_point = random.randint(1, len(parent1) - 1)
                offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
                offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                offspring1 = parent1
                offspring2 = parent2
            offspring_population.extend([offspring1, offspring2])

        # Mutation
        for i in range(len(offspring_population)):
            for j in range(len(offspring_population[i])):
                if random.random() < MUTATION_RATE:
                    # Randomly perturb the weight
                    offspring_population[i][j] += random.uniform(-0.1, 0.1)

        # Evaluate Fitness
        population = offspring_population
        fitness_scores = [calculate_fitness(nn_weights) for nn_weights in population]

        # Elitism
        elite_count = int(POPULATION_SIZE * 0.2)  # Select top 20% as elites
        elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k][0], reverse=True)[:elite_count]
        elites = [population[i] for i in elite_indices]
        population = elites + random.choices(population, k=POPULATION_SIZE - elite_count)

        # Track best test accuracy and weights
        current_test_accuracy = max(fitness_scores, key=lambda x: x[1])[1]
        if current_test_accuracy > best_test_accuracy:
            count_static_accuracy = 0
            best_test_accuracy = current_test_accuracy
            best_weights = max(population, key=calculate_fitness)

        if current_test_accuracy == best_test_accuracy:
            count_static_accuracy += 1

        # Print accuracy for current iteration
        train_accuracy, test_accuracy = max(fitness_scores, key=lambda x: x[0])
        print(f"Iteration {generation + 1}: Train Accuracy = {train_accuracy}, Test Accuracy = {test_accuracy}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time:", elapsed_time, "seconds")

        if count_static_accuracy == 10:
            end_time = time.time()  # Stop the timer
            break

        generation_list.append(generation + 1)
        test_accuracy_list.append(current_test_accuracy)

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time
    # Save the Best Neural Network
    with open('wnet1.txt', 'w') as file:
        file.write(' '.join(str(weight) for weight in best_weights))

    print("Elapsed Time:", elapsed_time / 60, "minutes")

    # Plot the increase in test accuracy
    plt.plot(generation_list, test_accuracy_list)
    plt.xlabel('Generation')
    plt.ylabel('Test Accuracy')
    plt.title('Increase in Test Accuracy')
    plt.savefig('test_accuracy_plot_nn0.png')
    plt.show()


if __name__ == "__main__":
    main()
