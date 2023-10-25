import pandas as pd

penguins = pd.read_csv("Assignment I/penguins.csv")
abalones = pd.read_csv("Assignment I/abalone.csv")

def main():
    #penguins_numerical = 
    #penguin_attributes = 
    print(pd.get_dummies(penguins))



##############################################################################################################

if __name__ == "__main__":
    main()