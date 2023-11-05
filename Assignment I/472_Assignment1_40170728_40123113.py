import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    v_measure_score,
)
from sklearn.model_selection import GridSearchCV

penguins = pd.read_csv("Assignment I/penguins.csv")
abalones = pd.read_csv("Assignment I/abalone.csv")


def main():
    # POINT 1 OF ASSIGNMENT INSTRUCTIONS
    #############################################
    penguin_dummy_num_attributes = pd.get_dummies(
        penguins.iloc[:, 1:]
    )  # Dummy-coded numerical penguin data

    # Self categorized numerical penguin data
    penguin_num_attributes = penguins.iloc[:, 1:]
    penguin_num_attributes.loc[
        penguin_num_attributes["island"] == "Torgersen", "island"
    ] = -1
    penguin_num_attributes.loc[
        penguin_num_attributes["island"] == "Dream", "island"
    ] = 0
    penguin_num_attributes.loc[
        penguin_num_attributes["island"] == "Biscoe", "island"
    ] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "MALE", "sex"] = 1
    penguin_num_attributes.loc[penguin_num_attributes["sex"] == "FEMALE", "sex"] = 0

    abalone_num_attributes = abalones.iloc[:, 1:]  # Numerical abalone data

    # POINT 2 OF ASSIGNMENT INSTRUCTIONS
    #############################################
    penguins_output_percent = save_graphic(penguins, "penguin-classes")
    abalones_output_percent = save_graphic(abalones, "abalone-classes")

    # POINT 3 OF ASSIGNMENT INSTRUCTIONS
    # Switch between the two lines below depending if you want dummy-data or self-categorized data for Penguins
    (
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
    ) = train_test_split(penguin_dummy_num_attributes, penguins.iloc[:, 0])
    # X_penguins_train, X_penguins_test, y_penguins_train, y_penguins_test = train_test_split(penguin_num_attributes, penguins.iloc[:, 0])
    (
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
    ) = train_test_split(abalone_num_attributes, abalones.iloc[:, 0])

    # print(f"PENGUINS:X_train={X_penguins_train.shape}; X_test={X_penguins_test.shape}; y_train={y_penguins_train.shape}; y_test={y_penguins_test.shape};\n")
    # print(f"ABALONES:X_train={X_abalones_train.shape}; X_test={X_abalones_test.shape}; y_train={y_abalones_train.shape}; y_test={y_abalones_test.shape};\n")

    # DECIDE ON FEATURES
    dtc_penguins = tree.DecisionTreeClassifier(criterion="entropy")
    dtc_penguins.fit(X_penguins_train, y_penguins_train)
    dot_data = tree.export_graphviz(
        dtc_penguins,
        out_file=None,
        feature_names=penguin_dummy_num_attributes.keys(),
        class_names=penguins.iloc[:, 0].unique(),
        filled=True,
        rounded=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("mytree")

    # 4C: Train and evaluate the base MLP classifier
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
    )

    train_and_evaluate_classifier(
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        base_mlp,
        "Base-MLP",
        "abalones",
        "best_param_BASE_MLP",
    )

    train_and_evaluate_classifier(
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        base_mlp,
        "Base-MLP",
        "penguins",
        "best_param_BASE_MLP",
    )

    # 4D: Perform grid search to find the top MLP
    top_mlp_Penguins = perform_grid_search(X_penguins_train, y_penguins_train)
    # 4D: Train and evaluate the top MLP classifier with the best parameters found
    train_and_evaluate_classifier(
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        top_mlp_Penguins,
        "Top-MLP",
        "penguins",
        best_param_TOP_MLP,
    )

    # 4D: Perform grid search to find the top MLP
    top_mlp_Abalones = perform_grid_search(X_penguins_train, y_penguins_train)
    # 4D: Train and evaluate the top MLP classifier with the best parameters found
    train_and_evaluate_classifier(
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        top_mlp_Abalones,
        "Top-MLP",
        "abalones",
        best_param_TOP_MLP,
    )


def save_graphic(df: pd.DataFrame, type):
    df_output = tuple(df.iloc[:, 0].unique())
    output_percent = []
    for output in df_output:
        output_percent.append(
            round(float(len(df[df.iloc[:, 0] == output])) / df.shape[0] * 100, 2)
        )
    fig, ax = plt.subplots()
    ax.bar(df_output, output_percent, color="blue", width=0.4)
    ax.set_ylabel("Instance Percentage (%)")
    ax.set_xlabel(df.keys()[0])
    if type == "penguin-classes":
        ax.set_title(
            "Percentage of the instances in each output class of Penguins data set"
        )
    else:
        ax.set_title(
            "Percentage of the instances in each output class of Abalone data set"
        )

    type = f"Assignment I/{type}"
    type += ".png"
    fig.savefig(type)
    plt.close(fig)
    return output_percent


# NOT NEEDED scikit-learn does it for us
# h_penguins = 0
# h_abalones = 0
# for nbr in penguins_output_percent:
#     h_penguins += -((nbr/100)*math.log2(nbr/100))
# for nbr in abalones_output_percent:
#     h_abalones += -((nbr/100)*math.log2(nbr/100))
def get_best_gain(df: pd.DataFrame, h):
    output_key = df.keys()[0]
    info_gains = []
    for key in df.keys():
        inter_df = df.get([output_key, key])
        h_key = 0
        for value in df[key].unique():
            h_key += 0
        info_gains.append(h - h_key)


def train_and_evaluate_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
    classifier,
    classifier_name,
    category,
    hyper_parameters,
):
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    conf_matrix = confusion_matrix(y_test, predictions)

    class_report = classification_report(y_test, predictions, output_dict=True)

    accuracy = accuracy_score(y_test, predictions)

    with open(f"{category}-performance-{classifier_name}.txt", "a") as file:
        file.write(
            f"A) --- {category}_{classifier_name} --- Hyper-Parameters: {hyper_parameters}\n"
        )
        file.write("B) Confusion Matrix:\n")
        file.write(f"{conf_matrix}\n\n")
        file.write("C) Classification Report:\n")
        file.write(f"{classification_report(y_test, predictions)}\n")
        file.write(f"D)\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Macro Average F1: {class_report['macro avg']['f1-score']:.2f}\n")
        file.write(
            f"Weighted Average F1: {class_report['weighted avg']['f1-score']:.2f}\n"
        )
        file.write("**************************************************\n\n")


def perform_grid_search(X_train, y_train):
    global best_param_TOP_MLP
    # Define the parameter grid to search
    parameter_space = {
        "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
    }

    # Create MLP classifier instance
    mlp = MLPClassifier(max_iter=1000)

    # Create GridSearchCV instance
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    # Best parameter set
    print("Best parameters found using TOP_MLP:\n", clf.best_params_)
    best_param_TOP_MLP = clf.best_params_

    # All results
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_


##############################################################################################################

if __name__ == "__main__":
    main()
