# Basic Calculator with Report Logging

from datetime import datetime

REPORT_FILE = "calc_report.txt"

# Function to log report
def log_report(expression, result):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(REPORT_FILE, "a") as file:
        file.write(f"{time} | {expression} = {result}\n")


# Calculator operations
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error (Division by zero)"
    return a / b


# Main calculator menu
def main():
    while True:
        print("\n====== BASIC CALCULATOR ======")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "5":
            print("👋 Calculator closed.")
            break

        if choice not in ["1", "2", "3", "4"]:
            print("⚠️ Invalid choice!")
            continue

        a = float(input("Enter first number: "))
        b = float(input("Enter second number: "))

        if choice == "1":
            result = add(a, b)
            expression = f"{a} + {b}"

        elif choice == "2":
            result = subtract(a, b)
            expression = f"{a} - {b}"

        elif choice == "3":
            result = multiply(a, b)
            expression = f"{a} * {b}"

        elif choice == "4":
            result = divide(a, b)
            expression = f"{a} / {b}"

        print("Result:", result)

        # Log the calculation
        log_report(expression, result)


# Run the program
main()
