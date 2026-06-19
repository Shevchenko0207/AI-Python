# @test_schedule_tasks
def schedule_tasks(tasks, resources, deadline):
    # Ініціалізація
    assigned_tasks = {res: [] for res in resources}
    task_times = {}
    resource_capacities = resources.copy()

    # Сортуємо завдання за тривалістю (за зростанням)
    sorted_tasks = sorted(tasks, key=lambda x: x[1])

    for task_name, duration, requirements in sorted_tasks:
        assigned = False

        # Шукаємо відповідний ресурс
        for res_name, req_cap in requirements.items():
            if (
                res_name in resource_capacities
                and resource_capacities[res_name] >= req_cap
            ):
                assigned_tasks[res_name].append(task_name)
                task_times[task_name] = duration
                resource_capacities[res_name] -= req_cap
                assigned = True
                break

        # Якщо завдання не вдалося призначити
        if not assigned:
            # Згідно з умовою, якщо не призначено, ігноруємо або "розширюємо"
            continue

    # Об'єднуємо результати
    result = assigned_tasks.copy()
    result.update(task_times)
    return result


# Example usage:
tasks_list = [
    ("TaskA", 4.5, {"Resource1": 2, "Resource2": 1}),
    ("TaskB", 7.2, {"Resource2": 3}),
    ("TaskC", 5.0, {"Resource1": 1}),
]

resources_dict = {"Resource1": 10, "Resource2": 15}
deadline_time = 12.0

result = schedule_tasks(tasks_list, resources_dict, deadline_time)
print(result)
