# =========================================
# Employee Leave Management Analysis
# =========================================

import pandas as pd

# -------------------------------
# STEP 1: Create Employee Leave Data
# -------------------------------
data = {
    "Employee_ID": [101, 102, 103, 104, 105],
    "Employee_Name": ["Ravi", "Anita", "Kumar", "Priya", "Arjun"],
    "Department": ["HR", "IT", "Finance", "IT", "HR"],
    "Leave_Type": ["Sick", "Casual", "Sick", "Earned", "Casual"],
    "Leave_Days": [2, 1, 3, 5, 2],
    "Month": ["Jan", "Jan", "Feb", "Mar", "Mar"]
}

df = pd.DataFrame(data)

print("\n=== Employee Leave Data ===")
print(df)

# -------------------------------
# STEP 2: Dataset Information
# -------------------------------
print("\n=== Dataset Info ===")
print(df.info())

# -------------------------------
# STEP 3: Check Missing Values
# -------------------------------
print("\n=== Missing Values ===")
print(df.isnull().sum())

# -------------------------------
# STEP 4: Total Leave per Employee
# -------------------------------
print("\n=== Total Leave Days per Employee ===")
total_leave_employee = df.groupby("Employee_Name")["Leave_Days"].sum()
print(total_leave_employee)

# -------------------------------
# STEP 5: Leave Taken by Department
# -------------------------------
print("\n=== Leave Days by Department ===")
department_leave = df.groupby("Department")["Leave_Days"].sum()
print(department_leave)

# -------------------------------
# STEP 6: Most Common Leave Type
# -------------------------------
print("\n=== Leave Type Count ===")
leave_type_count = df["Leave_Type"].value_counts()
print(leave_type_count)

# -------------------------------
# STEP 7: Monthly Leave Analysis
# -------------------------------
print("\n=== Monthly Leave Analysis ===")
monthly_leave = df.groupby("Month")["Leave_Days"].sum()
print(monthly_leave)

# -------------------------------
# STEP 8: Employees with High Leave (>3 days)
# -------------------------------
print("\n=== Employees with More Than 3 Leave Days ===")
high_leave = df[df["Leave_Days"] > 3]
print(high_leave)

# -------------------------------
# STEP 9: Save Output to CSV
# -------------------------------
df.to_csv("employee_leave_analysis.csv", index=False)
print("\nData saved as employee_leave_analysis.csv")

# -------------------------------
# STEP 10: Final Insights
# -------------------------------
print("\n=== Final Insights ===")
print("- Sick leave is the most commonly used leave type")
print("- IT department has higher leave usage")
print("- March shows increased leave activity")
print("- Some employees take long-duration leaves")

print("\n=== Analysis Completed Successfully ===")
