# File Handling Based Student Record System

FILE_NAME = "students.txt"

# 1. Add student record
def add_student():
    roll = input("Enter Roll Number: ")
    name = input("Enter Name: ")
    clas = input("Enter Class: ")
    marks = input("Enter Marks: ")

    with open(FILE_NAME, "a") as file:
        file.write(f"{roll},{name},{clas},{marks}\n")

    print("Student record added successfully!\n")


# 2. View all student records
def view_students():
    try:
        with open(FILE_NAME, "r") as file:
            records = file.readlines()

            if not records:
                print("No records found.\n")
                return

            print("\n--- Student Records ---")
            for record in records:
                roll, name, clas, marks = record.strip().split(",")
                print(f"Roll: {roll}, Name: {name}, Class: {clas}, Marks: {marks}")
            print()
    except FileNotFoundError:
        print("File not found.\n")


# 3. Search student by roll number
def search_student():
    search_roll = input("Enter Roll Number to search: ")

    with open(FILE_NAME, "r") as file:
        for record in file:
            roll, name, clas, marks = record.strip().split(",")
            if roll == search_roll:
                print("\nStudent Found")
                print(f"Roll: {roll}, Name: {name}, Class: {clas}, Marks: {marks}\n")
                return

    print("Student not found.\n")


# 4. Delete student record
def delete_student():
    delete_roll = input("Enter Roll Number to delete: ")
    found = False

    with open(FILE_NAME, "r") as file:
        records = file.readlines()

    with open(FILE_NAME, "w") as file:
        for record in records:
            roll = record.split(",")[0]
            if roll != delete_roll:
                file.write(record)
            else:
                found = True

    if found:
        print("Student record deleted successfully!\n")
    else:
        print("Student not found.\n")


# 5. Main menu
def main():
    while True:
        print("====== Student Record System ======")
        print("1. Add Student")
        print("2. View Students")
        print("3. Search Student")
        print("4. Delete Student")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            add_student()
        elif choice == "2":
            view_students()
        elif choice == "3":
            search_student()
        elif choice == "4":
            delete_student()
        elif choice == "5":
            print("Exiting program. Thank you!")
            break
        else:
            print("Invalid choice. Try again.\n")


# Run the program
main()