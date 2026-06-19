import itertools
import math


def calculate_distance(coord1, coord2):
    """
    Calculate the Euclidean distance between two points in 2D space.
    """
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


# @test_optimize_vrp
def optimize_vrp(depot, customers, vehicle_capacity, num_vehicles):
    """
    Optimize the Vehicle Routing Problem using Brute Force.
    """
    best_distance = float("inf")
    best_routes = []

    # Створюємо індекси клієнтів
    customer_indices = list(range(len(customers)))

    # Генеруємо всі можливі перестановки порядку відвідування клієнтів
    for p in itertools.permutations(customer_indices):
        routes = [[] for _ in range(num_vehicles)]
        current_capacities = [vehicle_capacity] * num_vehicles
        total_dist = 0
        possible = True

        # Розподіляємо клієнтів по машинах (спрощений жадібний підхід до перебору)
        for customer_idx in p:
            assigned = False
            for v in range(num_vehicles):
                # Припускаємо, що кожен клієнт має вимогу 1 (якщо інше не задано)
                if current_capacities[v] >= 1:
                    routes[v].append(customers[customer_idx])
                    current_capacities[v] -= 1
                    assigned = True
                    break
            if not assigned:
                possible = False
                break

        if not possible:
            continue

        # Розрахунок дистанції для кожного маршруту
        route_dist = 0
        for route in routes:
            if not route:
                continue
            # Від депо до першого клієнта
            route_dist += calculate_distance(depot, route[0])
            # Між клієнтами
            for i in range(len(route) - 1):
                route_dist += calculate_distance(route[i], route[i + 1])
            # Від останнього клієнта назад у депо
            route_dist += calculate_distance(route[-1], depot)

        if route_dist < best_distance:
            best_distance = route_dist
            best_routes = routes

    return best_routes


# Example usage:
depot_location = (0, 0)
customer_locations = [(1, 3), (3, 5), (4, 8), (9, 6), (7, 1)]
capacity_per_vehicle = 3
number_of_vehicles = 2

optimized_routes = optimize_vrp(
    depot_location, customer_locations, capacity_per_vehicle, number_of_vehicles
)
print(optimized_routes)
