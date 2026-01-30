# Text File Word and Character Analyzer

# Step 1: Open the file
file = open("sample.txt", "r")

# Step 2: Read all content
text = file.read()

# Step 3: Close the file
file.close()

# Step 4: Count characters
char_count = len(text)

# Step 5: Count words
words = text.split()
word_count = len(words)

# Step 6: Count lines
file = open("sample.txt", "r")
lines = file.readlines()
line_count = len(lines)
file.close()

# Step 7: Display results
print("Text File Analysis")
print("----------------------")
print("Total Characters:", char_count)
print("Total Words     :", word_count)
print("Total Lines     :", line_count)
