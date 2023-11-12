import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
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
    # (
    #     X_penguins_train,
    #     X_penguins_test,
    #     y_penguins_train,
    #     y_penguins_test,
    # ) = train_test_split(penguin_dummy_num_attributes, penguins.iloc[:, 0])
    (
        X_penguins_train, 
        X_penguins_test, 
        y_penguins_train, 
        y_penguins_test
    ) = train_test_split(penguin_num_attributes, penguins.iloc[:, 0])
    (
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
    ) = train_test_split(abalone_num_attributes, abalones.iloc[:, 0])

    # Deleting outdated files that will be generated
    penguins_output = "penguin-performance.txt"
    abalones_output = "abalone-performance.txt"
    if os.path.exists(penguins_output):
        os.remove(penguins_output)
    if os.path.exists(abalones_output):
        os.remove(abalones_output)

    # 4A: Train and evaluate the base DT
    dtc_penguins = tree.DecisionTreeClassifier(criterion="entropy")
    dtc_penguins.fit(X_penguins_train, y_penguins_train)

    # Visualizing the tree - limited
    plt.figure(figsize=(20, 10))  # Set figure size for better readability
    tree.plot_tree(
        dtc_penguins,
        filled=True,
        rounded=True,
        feature_names=penguin_num_attributes.columns,
        #feature_names=penguin_dummy_num_attributes.columns,
        class_names=penguins.iloc[:, 0].unique(),
    )
    plt.savefig("penguins_decision_tree.png")  # Save the figure to a file

    # 4A: Train the Base-DT for the Abalones dataset
    dtc_abalones = tree.DecisionTreeClassifier(criterion="entropy")
    dtc_abalones.fit(X_abalones_train, y_abalones_train)

    # Visualizing the tree - limited to depth for clearer visualization
    plt.figure(figsize=(20, 10))  # Set figure size for better readability
    tree.plot_tree(
        dtc_abalones,
        filled=True,
        rounded=True,
        feature_names=abalone_num_attributes.columns,
        class_names=abalones.iloc[:, 0].unique(),
        max_depth=3,
    )  # Limited depth to 3 for visualization, remove to visualize the full tree
    plt.savefig("abalone_decision_tree.png")  # Save the figure to a file

    # 4A: Train the Base-DT for the Penguins dataset
    train_and_evaluate_classifier(
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        dtc_penguins,
        "Base-DT",
        "penguins",
        "Default parameters",
        penguins_output
    )
    # 4A: Train the Base-DT for the Abalones dataset
    train_and_evaluate_classifier(
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        dtc_abalones,
        "Base-DT",
        "abalones",
        "Default parameters",
        abalones_output
    )

    # 4B: Train the Base-DT for the Penguins dataset
    top_DT_Penguins = perform_grid_search_Top_DT(
        X_penguins_train, y_penguins_train, "penguins"
    )
    train_and_evaluate_classifier(
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        top_DT_Penguins,
        "Top-DT",
        "penguins",
        best_param_TOP_DT,
        penguins_output
    )

    # 4B: Train the Base-DT for the Abalones dataset
    top_DT_abalones = perform_grid_search_Top_DT(
        X_abalones_train, y_abalones_train, "abalones"
    )
    train_and_evaluate_classifier(
        X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        top_DT_abalones,
        "Top-DT",
        "abalones",
        best_param_TOP_DT,
        abalones_output
    )

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
        "Default parameters",
        abalones_output
    )

    train_and_evaluate_classifier(
        X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        base_mlp,
        "Base-MLP",
        "penguins",
        "Default parameters",
        penguins_output
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
        penguins_output
    )

    # 4D: Perform grid search to find the top MLP
    top_mlp_Abalones = perform_grid_search(X_abalones_train, y_abalones_train)
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
        abalones_output
    )

    with open(penguins_output, "a") as file:
        file.write(
            f"\n------------------------------------ PART 6 ------------------------------------\n"
        )
    with open(abalones_output, "a") as file:
        file.write(
            f"\n------------------------------------ PART 6 ------------------------------------\n"
        )
    # 6A
    get_variance(X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        dtc_penguins,
        "Base-DT",
        "penguins",
        penguins_output)
    get_variance(X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        dtc_abalones,
        "Base-DT",
        "abalones",
        abalones_output)

    # 6B
    get_variance(X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        top_DT_Penguins,
        "Top-DT",
        "penguins",
        penguins_output)
    get_variance(X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        top_DT_abalones,
        "Top-DT",
        "abalones",
        abalones_output)

    # 6C
    get_variance(X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        base_mlp,
        "Base-MLP",
        "penguins",
        penguins_output)
    get_variance(X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        base_mlp,
        "Base-MLP",
        "abalones",
        abalones_output)

    # 6D
    get_variance(X_penguins_train,
        X_penguins_test,
        y_penguins_train,
        y_penguins_test,
        top_mlp_Penguins,
        "Top-MLP",
        "penguins",
        penguins_output)
    get_variance(X_abalones_train,
        X_abalones_test,
        y_abalones_train,
        y_abalones_test,
        top_mlp_Abalones,
        "Top-MLP",
        "abalones",
        abalones_output)


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

def train_and_evaluate_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
    classifier,
    classifier_name,
    category,
    hyper_parameters,
    path
):
    (accuracy, macro_f1, weighted_f1, predictions) = get_stats(X_train, X_test, y_train, y_test, classifier)
    conf_matrix = confusion_matrix(y_test, predictions)

    with open(path, "a") as file:
        file.write(
            f"A) --- {category}_{classifier_name} --- Hyper-Parameters: {hyper_parameters}\n"
        )
        file.write("B) Confusion Matrix:\n")
        file.write(f"{conf_matrix}\n\n")
        file.write("C) Classification Report:\n")
        file.write(f"{classification_report(y_test, predictions)}\n")
        file.write(f"D)\n")
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Macro Average F1: {macro_f1:.2f}\n")
        file.write(f"Weighted Average F1: {weighted_f1:.2f}\n")
        file.write("**************************************************\n\n")

def get_stats(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    class_report = classification_report(y_test, predictions, output_dict=True)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = class_report["macro avg"]["f1-score"]
    weighted_f1 = class_report["weighted avg"]["f1-score"]

    return (accuracy, macro_f1, weighted_f1, predictions)

def get_variance(X_train, X_test, y_train, y_test, classifier,classifier_name, category, path):
    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    averages = []
    variances = []
    for i in range(5):
        (accuracy, macro_f1, weighted_f1, predictions) = get_stats(X_train, X_test, y_train, y_test, classifier)
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        weighted_f1s.append(weighted_f1)

    averages.append(sum(accuracies)/len(accuracies))
    averages.append(sum(macro_f1s)/len(macro_f1s))
    averages.append(sum(weighted_f1s)/len(weighted_f1s))
    temp_sum = 0.0
    for a in accuracies:
        temp_sum += (a-averages[0])**2
    variances.append(100*temp_sum/len(accuracies))
    temp_sum = 0.0
    for f in macro_f1s:
        temp_sum += (a-averages[1])**2
    variances.append(100*temp_sum/len(macro_f1s))
    temp_sum = 0.0
    for f in weighted_f1s:
        temp_sum += (a-averages[2])**2
    variances.append(100*temp_sum/len(weighted_f1s))

    with open(path, "a") as file:
        file.write(
            f"--- {category}_{classifier_name} ---\n"
        )
        file.write("A) Accuracy:\n")
        file.write(f"Average = {averages[0]};\tVariance = {variances[0]} %\t Std = {math.sqrt(variances[0])} %\n\n")
        file.write("B) Macro Average F1:\n")
        file.write(f"Average = {averages[1]};\tVariance = {variances[1]} %\t Std = {math.sqrt(variances[1])} %\n\n")
        file.write(f"C) Weighted Average F1\n")
        file.write(f"Average = {averages[2]};\tVariance = {variances[2]} %\t Std = {math.sqrt(variances[2])} %\n\n")
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
    #for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_


def perform_grid_search_Top_DT(X_train, y_train, category):
    global best_param_TOP_DT
    # Define the parameter grid to search
    parameter_space = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4, 6],
    }

    scoring_function = "accuracy"
    # Create MLP classifier instance
    dt_classifier = tree.DecisionTreeClassifier(random_state=42)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(
        dt_classifier, parameter_space, scoring=scoring_function, cv=5
    )
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_

    # Best parameter set
    print("Best parameters found using TOP_MLP:\n", grid_search.best_params_)
    best_param_TOP_DT = grid_search.best_params_

    # All results
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    # for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        # Visualizing the tree - limited to depth for clearer visualization
    feature_names = X_train.columns.tolist()
    class_names = (
        y_train.unique().tolist()
    )  # Replace 'y_train' with your target series for the classes' names
    plt.figure(figsize=(20, 10))  # Set figure size for better readability
    tree.plot_tree(
        best_classifier,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=3,
    )  # Limit depth to 3 for visualization, remove to visualize the full tree
    plt.savefig(f"{category}_decision_tree_Top_DT.png")  # Save the figure to a file
    # plt.show()  # Display the figure inline

    return best_classifier


##############################################################################################################

if __name__ == "__main__":
    main()
