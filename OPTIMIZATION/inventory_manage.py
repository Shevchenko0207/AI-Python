import pulp


# @test_optimize_inventory_management
def optimize_inventory_management(
    demand, holding_cost, ordering_cost, initial_inventory, reorder_point
):
    periods = range(len(demand))

    # Ініціалізація моделі
    model = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

    # Змінні рішення
    inventory = pulp.LpVariable.dicts("Inv", periods, lowBound=0, cat="Continuous")
    order_quantity = pulp.LpVariable.dicts(
        "Order", periods, lowBound=0, cat="Continuous"
    )

    # Цільова функція: мінімізація витрат (утримання + замовлення)
    # Примітка: для спрощення замовлення вважається як лінійна вартість
    model += pulp.lpSum(
        [
            inventory[p] * holding_cost + order_quantity[p] * ordering_cost
            for p in periods
        ]
    )

    # Обмеження
    model += inventory[0] == initial_inventory + order_quantity[0] - demand[0]

    for p in periods[1:]:
        model += inventory[p] == inventory[p - 1] + order_quantity[p] - demand[p]
        model += inventory[p] >= reorder_point

    # Розв'язання
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Збір результатів
    optimal_inventory_levels = [pulp.value(inventory[p]) for p in periods]
    return optimal_inventory_levels


# Приклад використання:
demand = [10, 15, 10, 20]
holding_cost = 2
ordering_cost = 50
initial_inventory = 20
reorder_point = 5

result = optimize_inventory_management(
    demand, holding_cost, ordering_cost, initial_inventory, reorder_point
)
print(result)
