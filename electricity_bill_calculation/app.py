# Electricity Bill Calculation System

# Function to calculate bill amount
def calculate_bill(units):
    bill = 0

    if units <= 100:
        bill = units * 1.5

    elif units <= 200:
        bill = (100 * 1.5) + (units - 100) * 2.5

    elif units <= 300:
        bill = (100 * 1.5) + (100 * 2.5) + (units - 200) * 4.0

    else:
        bill = (100 * 1.5) + (100 * 2.5) + (100 * 4.0) + (units - 300) * 6.0

    return bill


# Main program
def main():
    print("ELECTRICITY BILL SYSTEM")

    name = input("Enter Customer Name: ")
    customer_no = input("Enter Customer Number: ")
    units = int(input("Enter Units Consumed: "))

    amount = calculate_bill(units)

    print("\n------ Electricity Bill ------")
    print("Customer Name   :", name)
    print("Customer Number :", customer_no)
    print("Units Consumed  :", units)
    print("Total Amount    : ₹", amount)
    print("------------------------------")


# Run the program
main()
