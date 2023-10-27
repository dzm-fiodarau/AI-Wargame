import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

penguins = pd.read_csv("Assignment I/penguins.csv")
abalones = pd.read_csv("Assignment I/abalone.csv")

def main():
    # penguin_species = tuple(penguins["species"].unique())
    # penguin_species_percent = []
    # for species in penguin_species:
    #     penguin_species_percent.append(round(float(len(penguins[penguins["species"]==species]))/penguins.shape[0]*100,2))

    penguin_dummy_num_attributes = pd.get_dummies(penguins.iloc[:,1:]) # Dummy-coded numerical penguin data

    penguin_dummy_num_attributes = pd.get_dummies(penguins.iloc[:,1:]) # Dummy-coded numerical penguin data

    # Self categorized numerical penguin data
    penguin_num_attributes = penguins.iloc[:,1:]
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Torgersen", "island"] = -1
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Dream", "island"] = 0
    penguin_num_attributes.loc[penguin_num_attributes["island"] == "Biscoe", "island"] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "MALE", "sex"] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "FEMALE", "sex"] = 0

    abalone_num_attributes = abalones.iloc[:,1:] # Numerical abalone data

    # print(penguin_species_percent)
    # plt.bar(penguin_species, penguin_species_percent, color ='blue', width = 0.4)
    # plt.ylabel("Instance Percentage (%)")
    # plt.title("Percentage of the instances in each output class of Penguins data set")

    # plt.show()
    save_graphic(penguins, "penguin-classes")
    save_graphic(abalones, "abalone-classes")

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