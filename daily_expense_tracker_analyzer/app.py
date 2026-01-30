# Daily Expense Tracker and Analyzer

import pandas as pd
from datetime import datetime

FILE_NAME = "expenses.csv"


# 1. Add new expense
def add_expense():
    date = datetime.now().strftime("%Y-%m-%d")
    category = input("Enter category (Food/Travel/etc): ")
    description = input("Enter description: ")
    amount = float(input("Enter amount: "))

    new_data = {
        "Date": date,
        "Category": category,
        "Description": description,
        "Amount": amount
    }

    df = pd.read_csv(FILE_NAME)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(FILE_NAME, index=False)

    print("Expense added successfully!\n")


# 2. View all expenses
def view_expenses():
    df = pd.read_csv(FILE_NAME)
    print("\nAll Expenses:")
    print(df)
    print()


# 3. Analyze expenses
def analyze_expenses():
    df = pd.read_csv(FILE_NAME)

    print("\nExpense Analysis")

    # Total expense
    total = df["Amount"].sum()
    print("Total Expense: ₹", total)

    # Category-wise expense
    category_sum = df.groupby("Category")["Amount"].sum()
    print("\nCategory-wise Expense:")
    print(category_sum)
    print()


# 4. Main menu
def main():
    while True:
        print("====== DAILY EXPENSE TRACKER ======")
        print("1. Add Expense")
        print("2. View Expenses")
        print("3. Analyze Expenses")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            analyze_expenses()
        elif choice == "4":
            print("Exiting Expense Tracker")
            break
        else:
            print("Invalid choice\n")


# Run the program
main()
