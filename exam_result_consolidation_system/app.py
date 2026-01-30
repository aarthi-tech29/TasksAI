import pandas as pd

data = {
    "Student_ID": [1, 2, 3, 4, 5],
    "Name": ["Amit", "Sneha", "Rahul", "Priya", "Karan"],
    "Maths": [78, 85, 62, 90, 55],
    "Science": [72, 88, 60, 92, 58],
    "English": [80, 75, 65, 85, 60]
}

df = pd.DataFrame(data)

# Total and Average
df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

# Grade assignment
def assign_grade(avg):
    if avg >= 85:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 50:
        return "C"
    else:
        return "F"

df["Grade"] = df["Average"].apply(assign_grade)

# Pass / Fail
df["Result"] = df["Grade"].apply(lambda x: "Pass" if x != "F" else "Fail")

# Ranking
df["Rank"] = df["Total"].rank(ascending=False, method="dense")

# Sort by Rank
df = df.sort_values("Rank")

print("\n=== Consolidated Exam Results ===")
print(df)

print("\n=== Subject-wise Class Average ===")
print(df[["Maths", "Science", "English"]].mean())

# Save to CSV
df.to_csv("exam_results.csv", index=False)

print("\nResults saved as exam_results.csv")
