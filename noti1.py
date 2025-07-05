# To-Do List App

class ToDoList:
    """Class to represent a To-Do List"""

    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        """Method to add a task to the list"""
        self.tasks.append(task)

    def remove_task(self, task):
        """Method to remove a task from the list"""
        if task in self.tasks:
            self.tasks.remove(task)
        else:
            print("Task not found!")

    def view_tasks(self):
        """Method to view all tasks in the list"""
        print("Tasks:")
        for i, task in enumerate(self.tasks, start=1):
            print(f"{i}. {task}")

# Test the class
todo_list = ToDoList()
todo_list.add_task("Walk dog")
todo_list.add_task("Wash dishes")
todo_list.add_task("Finish homework")
todo_list.view_tasks()
todo_list.remove_task("Wash dishes")
todo_list.view_tasks()
