import pandas as pd

# Load the confusion matrix from the csv file
confusion_matrix = pd.read_csv("cprobs.csv", index_col=0)

# Loop to allow multiple queries
while True:
    # Get input from the user
    class1 = input("Enter conditional class: ")
    class2 = input("Enter probability class: ")
    
    # Check if the input is valid
    if class1 not in confusion_matrix.columns or class2 not in confusion_matrix.columns:
        print("Invalid input, please try again.")
        continue
    
    # Get the conditional probability
    cprob = confusion_matrix.at[class1, class2]
    
    # Print the result
    print(f"The conditional probability of {class2} given {class1} is {cprob}\n")