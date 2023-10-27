import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("Assignment I/penguins.csv")
abalones = pd.read_csv("Assignment I/abalone.csv")

def main():
    # POINT 1 OF ASSIGNMENT INSTRUCTIONS
    #############################################
    penguin_dummy_num_attributes = pd.get_dummies(penguins.iloc[:,1:]) # Dummy-coded numerical penguin data

    # Self categorized numerical penguin data
    penguin_num_attributes = penguins.iloc[:,1:]
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Torgersen", "island"] = -1
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Dream", "island"] = 0
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Biscoe", "island"] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "MALE", "sex"] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "FEMALE", "sex"] = 0

    abalone_num_attributes = abalones.iloc[:,1:] # Numerical abalone data

    # POINT 2 OF ASSIGNMENT INSTRUCTIONS
    #############################################
    save_graphic(penguins, "penguin-classes")
    save_graphic(abalones, "abalone-classes")

    # POINT 3 OF ASSIGNMENT INSTRUCTIONS
    # Switch between the two lines below depending if you want dummy-data or self-categoried data for Penguins
    X_penguins_train, X_penguins_test, y_penguins_train, y_penguins_test = train_test_split(penguin_dummy_num_attributes, penguins.iloc[:, 0])
    #X_penguins_train, X_penguins_test, y_penguins_train, y_penguins_test = train_test_split(penguin_num_attributes, penguins.iloc[:, 0])
    
    print(f"PENGUINS:X_train={X_penguins_train.shape}; X_test={X_penguins_test.shape}; y_train={y_penguins_train.shape}; y_test={y_penguins_test.shape};\n")
    X_abalones_train, X_abalones_test, y_abalones_train, y_abalones_test = train_test_split(abalone_num_attributes, abalones.iloc[:, 0])
    print(f"ABALONES:X_train={X_abalones_train.shape}; X_test={X_abalones_test.shape}; y_train={y_abalones_train.shape}; y_test={y_abalones_test.shape};\n")

def save_graphic(df : pd.DataFrame, type):
    df_output = tuple(df.iloc[:, 0].unique())
    output_percent = []
    for output in df_output:
        output_percent.append(round(float(len(df[df.iloc[:, 0]==output]))/df.shape[0]*100,2))
    fig, ax = plt.subplots()
    ax.bar(df_output, output_percent, color ='blue', width = 0.4)
    ax.set_ylabel("Instance Percentage (%)")
    ax.set_xlabel(df.keys()[0])
    if type=="penguin-classes":
        ax.set_title("Percentage of the instances in each output class of Penguins data set")
    else:
        ax.set_title("Percentage of the instances in each output class of Abalone data set")
    
    type = f"Assignment I/{type}"
    type += ".png"
    fig.savefig(type)
    plt.close(fig)

##############################################################################################################

if __name__ == "__main__":
    main()