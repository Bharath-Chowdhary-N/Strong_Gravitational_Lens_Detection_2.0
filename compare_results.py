import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pandas as pd
import json
from utils import load_settings_yaml, smooth_curve
from Parameters import Parameters
from DataGenerator import DataGenerator
from Network import Network
import csv
import math


# Count the number of true positive, true negative, false positive and false negative in for a prediction vector relative to the label vector.
def count_TP_TN_FP_FN_and_FB(prediction_vector, y_test, threshold, beta_squarred, verbatim = False):
    TP = 0 #true positive
    TN = 0 #true negative
    FP = 0 #false positive
    FN = 0 #false negative

    for idx, pred in enumerate(prediction_vector):
        if pred >= threshold and y_test[idx] >= threshold:
            TP += 1
        if pred < threshold and y_test[idx] < threshold:
            TN += 1
        if pred >= threshold and y_test[idx] < threshold:
            FP += 1
        if pred < threshold and y_test[idx] >= threshold:
            FN += 1

    tot_count = TP + TN + FP + FN
    
    precision = TP/(TP + FP) if TP + FP != 0 else 0
    recall    = TP/(TP + FN) if TP + FN != 0 else 0
    fp_rate   = FP/(FP + TN) if FP + TN != 0 else 0
    accuracy  = (TP + TN) / len(prediction_vector) if len(prediction_vector) != 0 else 0
    F_beta    = (1+beta_squarred) * ((precision * recall) / ((beta_squarred * precision) + recall)) if ((beta_squarred * precision) + recall) else 0
    
    if verbatim:
        if tot_count != len(prediction_vector):
            print("Total count {} of (TP, TN, FP, FN) is not equal to the length of the prediction vector: {}".format(tot_count, len(prediction_vector)), flush=True)

        print("Total Count {}\n\tTP: {}, TN: {}, FP: {}, FN: {}".format(tot_count, TP, TN, FP, FN), flush=True)
        print("precision = {}".format(precision), flush=True)
        print("recall    = {}".format(recall), flush=True)
        print("fp_rate   = {}".format(fp_rate), flush=True)
        print("accuracy  = {}".format(accuracy), flush=True)
        print("F beta    = {}".format(F_beta), flush=True)

    return TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta


# Construct dataframes of csv files in the given model folders. 
# This will be used for plotting data about the model
# csv - Output Dump
def get_dataframes(models):
    if verbatim:
        print("\n\nReading in csv data:")
    dfs = []    # list of dataframes - for each csv
    model_paths_csv = []
    for idx, model in enumerate(models):
        if verbatim:
            print("Model: {} - csv data".format(idx))
        # path = os.path.join(root_models, model)
        history_csv_path = glob.glob(model + "/*history.csv")[0]
        model_paths_csv.append(history_csv_path)
        dfs.append(pd.read_csv(history_csv_path))
        if verbatim:
            print("path = {}".format(history_csv_path))
            print(dfs[idx].head())
            print("\n")
    return dfs, model_paths_csv


# Construct jsons of josn files in the given model folders.
# This will be used to get model paramters when plotting
# json - Input Parameter Dump
def get_jsons(models):
    if verbatim:
        print("\n\njson files: ")
    jsons = []
    model_paths_json = []
    for idx, model in enumerate(models):
        if verbatim:
            print("Model: {} - json data".format(idx))
        paramDump_json_path = glob.glob(model + "/*.json")[0]
        model_paths_json.append(paramDump_json_path)
        jsons.append(json.load(open(paramDump_json_path)))
        if verbatim:
            print("path = {}".format(paramDump_json_path))
            for i in jsons[idx]:
                print("\t" + i + ": " + str(jsons[idx][i]))
            print("\n")
    return jsons, model_paths_json


# For each model path in models find a .h5 file path and return it.
def get_h5s_paths(models):
    if verbatim:
        print("\n\nh5 files: ")
    paths_h5s = []
    for idx, model in enumerate(models):
        if verbatim:
            print("Model: {} - h5 file".format(idx))
        h5_path = glob.glob(model + "/*.h5")[0]
        paths_h5s.append(h5_path)
    return paths_h5s


# dstack the data to three channels instead of one
def dstack_data(data):
    dstack_data = np.empty((data.shape[0], data.shape[1], data.shape[2], 3), dtype=np.float32)
    for i in range(data.shape[0]):
        img = data[i]
        dstack_data[i] = np.dstack((img,img,img))
    return dstack_data


# Average predictions of the neural network over a 'avg_iter_counter' amount of times
def average_prediction_results(network, data, avg_iter_counter=10, verbose=True):
    avg_preds = np.zeros((data.shape[0],1), dtype=np.float32)
    for i in range(avg_iter_counter):
        if verbose:
            print("avg iter counter = {}".format(i))
        prediction_vector = network.model.predict(data)
        avg_preds = np.add(avg_preds, prediction_vector)
    avg_preds = avg_preds / avg_iter_counter
    if verbose:
        print("Length prediction vector: {}".format(len(avg_preds)), flush=True)
    return avg_preds


# Load validation chunk and calculate per model folder its model performance evaluated on f-beta score
def store_fbeta_results(models, paths_h5s, jsons, json_comp_key, f_beta_avg_count, do_eval=True):
    for idx, model_folder in enumerate(models):
        # Step 0.0 - inits
        f_beta_full_path = os.path.join(model_folder, "f_beta_results.csv")
        full_path_fBeta_figure = os.path.join(model_folder, "f_beta_graph.png")

        # Step 1.0 - Load settings of the model
        yaml_path = glob.glob(os.path.join(model_folder) + "/*.yaml")[0]
        settings_yaml = load_settings_yaml(yaml_path)
        
        # Step 2.0 - Set Parameters - and overload fraction to load sources - because not all are needed and it will just slow things down for now.
        params = Parameters(settings_yaml, yaml_path, mode="no_training")
        params.fraction_to_load_sources_vali = 0.15

        params.data_type = np.float32 if params.data_type == "np.float32" else np.float32       # must be done here, due to the json, not accepting this kind of if statement in the parameter class.
        if params.model_name == "Baseline_Enrico":
            params.img_dims = (101,101,3)
      
        # Step 3.0 - Define a DataGenerator that can generate validation chunks based on validation data.
        dg = DataGenerator(params, mode="no_training", do_shuffle_data=False)     #do not shuffle the data in the data generator
        
        # Step 4.0 - Construct a neural network with the same architecture as that it was trained with.
        network = Network(params, dg, training=False) # The network needs to know hyper-paramters from params, and needs to know how to generate data with a datagenerator object.
        network.model.trainable = False
        
        # Step 5.0 - Load weights of the neural network
        network.model.load_weights(paths_h5s[idx])
        
        # Step 6.0 - we want a nice plot with standard deviation.
        f_beta_vectors = []
        precision_data_vectors = []
        recall_data_vectors = []
        for i in range(f_beta_avg_count):
            X_validation_chunk, y_validation_chunk = dg.load_chunk_val(params.data_type, params.mock_lens_alpha_scaling)
            
            # Step 6.1 - dstack images for enrico neural network
            if params.model_name == "Baseline_Enrico":
                X_validation_chunk = dstack_data(X_validation_chunk)

            # Step 6.2 - Predict the labels of the validation chunk on the loaded neural network - averaged over 'avg_iter_counter' predictions
            preds = network.model.predict(X_validation_chunk)

            # Step 6.3 - Also calculate an evaluation based on the models evaluation metric
            if do_eval:
                results = network.model.evaluate(X_validation_chunk, y_validation_chunk, verbose=0)
                for met_idx in range(len(results)):
                    print("{} = {}".format(network.model.metrics_names[met_idx], results[met_idx]))

            # Step 6.3 - Begin f-beta calculation and store into csv file
            f_betas = []
            precision_data = []
            recall_data = []
            with open(f_beta_full_path, 'w', newline='') as f_beta_file:
                writer = csv.writer(f_beta_file)
                writer.writerow(["p_threshold", "TP", "TN", "FP", "FN", "precision", "recall", "fp_rate", "accuracy", "f_beta"])
                for p_threshold in threshold_range:
                    (TP, TN, FP, FN, precision, recall, fp_rate, accuracy, F_beta) = count_TP_TN_FP_FN_and_FB(preds, y_validation_chunk, p_threshold, beta_squarred)
                    f_betas.append(F_beta)
                    precision_data.append(precision)
                    recall_data.append(recall)
                    writer.writerow([str(p_threshold), str(TP), str(TN), str(FP), str(FN), str(precision), str(recall), str(fp_rate), str(accuracy), str(F_beta)])
            f_beta_vectors.append(f_betas)
            precision_data_vectors.append(precision_data)
            recall_data_vectors.append(recall_data)
        print("saved csv with f_beta scores to: {}".format(f_beta_full_path), flush=True)
        
        # step 7.0 - calculate std and mean - based on f_beta_vectors
        colls = list(zip(*f_beta_vectors))
        means = np.asarray(list(map(np.mean, map(np.asarray, colls))))
        stds = np.asarray(list(map(np.std, map(np.asarray, colls))))

        # step 7.1 - define upper and lower limits
        upline  = np.add(means, stds)
        lowline = np.subtract(means, stds)

        # Step 7.1.1 - Calculate mean precision and recall rates
        colls_pre = list(zip(*precision_data_vectors))
        precision_mu = np.asarray(list(map(np.mean,(map(np.asarray, colls_pre)))))
        colls_recall = list(zip(*recall_data_vectors))
        recall_mu = np.asarray(list(map(np.mean,(map(np.asarray, colls_recall)))))

        # step 7.2 - Plotting all lines
        plt.plot(list(threshold_range), precision_mu, ":", color=colors[idx], label="precision mean", alpha=0.9, linewidth=3)
        plt.plot(list(threshold_range), recall_mu, "--", color=colors[idx], label="recall mean", alpha=0.9, linewidth=3)
        plt.plot(list(threshold_range), upline, colors[idx])
        plt.plot(list(threshold_range), means, colors[idx], label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]))
        plt.plot(list(threshold_range), lowline, colors[idx])
        plt.fill_between(list(threshold_range), upline, lowline, color=colors[idx], alpha=0.5) 

        plt.xlabel("p threshold")
        plt.ylabel("F")
        plt.title("F_beta score - Beta = {0:.2f}".format(math.sqrt(beta_squarred)))
        figure = plt.gcf() # get current figure
        figure.set_size_inches(12, 8)       # (12,8), seems quite fine
        plt.savefig(full_path_fBeta_figure, dpi=100)
        print("figure saved: {}".format(full_path_fBeta_figure), flush=True)

    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()
    

# Universal plot function that plots many plots
def compare_plot_models(comparing_headerName_df, dfs, jsons, json_comp_key):
    print("------------------------------")
    print("Plotting {}...".format(comparing_headerName_df))
    for idx in range(len(dfs)):         # loop over each model dataframe        
        data = dfs[idx][comparing_headerName_df]
        
        # time needs to be formatted before it can be plotted
        if comparing_headerName_df == "time":
            data = list(data)
            formatted_time_data = []     # in minutes
            for timestamp in data:
                parts = timestamp.split(":")
                formatted_time_data.append(int(parts[0])*60 + int(parts[1]) + int(parts[2])/60)
            data = formatted_time_data

        plt.plot(data, label = str(json_comp_key) + ": " + str(jsons[idx][json_comp_key]), color=colors[idx], linewidth=1)
        print(str(str(json_comp_key) + ": " + str(jsons[idx][json_comp_key])))

    plt.title(comparing_headerName_df)
    plt.ylabel(comparing_headerName_df)
    plt.xlabel("Trained Chunks")
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()


# Plot the losses of the trained model over training time. Plot the Exponential Moving Average
def plot_losses_avg(models, dfs, jsons,  smooth_fac=0.9, do_diff_loss=False):
    print("------------------------------")
    print("Plotting average losses...")
    for j in range(len(models)):
        df_val_loss = dfs[j]["val_loss"]
        df_loss     = dfs[j]["loss"]

        val_loss_avg = []
        for i in range(len(df_val_loss)):
            val_loss_avg.append(df_val_loss[i])
            
        loss_avg = []
        for i in range(len(df_loss)):
            loss_avg.append(df_loss[i]) 

        plt.plot(smooth_curve(val_loss_avg, factor=smooth_fac), label="val loss  {}".format(jsons[j][json_comp_key]), color=colors[j], linewidth=3)
        plt.plot(smooth_curve(loss_avg, factor=smooth_fac), label="train loss  {}".format(jsons[j][json_comp_key]), color=colors[j], linewidth=1)
        plt.title("Model losses")
        plt.ylabel("loss")
        plt.xlabel("Trained Chunks")
        
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.savefig(os.path.join(models[j], "avg_loss"))
    plt.show()


# Prompt the user to fill in which experiment folder to run.
def set_experiment_folder(root_folder):
    print("------------------------------")
    print("\n\nRoot folder of experiment: {}".format(root_folder))


    cwd = os.path.join(os.getcwd(), root_folder)
    folders = sorted(os.listdir(cwd))
    local_folders = [x for x in folders if os.path.isdir(os.path.join(cwd, x))]

    print("\nSet experiment folder:")
    for idx, exp_folder in enumerate(local_folders):
        print("\t{} - {}".format(idx, exp_folder))
    exp_idx = int(input("Set number experiment folder (integer): "))
    print("Choose index: {}, {}".format(exp_idx, os.path.join(root_folder, local_folders[exp_idx])))

    return os.path.join(root_folder, local_folders[exp_idx])


# Prompt the user which models to compare against each other in the given experiment folder
def set_models_folders(experiment_folder):
    print("------------------------------")
    print("\n\nRoot folder this experiment: {}".format(experiment_folder))

    cwd = os.path.join(os.getcwd(), experiment_folder)
    folders = sorted(os.listdir(cwd))
    local_folders = [x for x in folders if os.path.isdir(os.path.join(cwd, x))]

    print("\nSet model folders:")
    for idx, folder in enumerate(local_folders):
        print("\t{} - {}".format(idx, folder))
    folder_idxs = input("Set indexes model folder(s)\n(integer)\nOr comma seperated ints: ")
    
    str_indexes = folder_idxs.split(',')
    chosen_models = [local_folders[int(string_idx)] for string_idx in str_indexes]
    
    print("\nUser Choices: ")
    for chosen_model in chosen_models:
        print(chosen_model)

    full_paths = [os.path.join(cwd, m) for m in chosen_models]
    return full_paths


# Plot the Exponential Moving Average error (in percentage) of the given models over time/chunks
def plot_errors(models, dfs, jsons, json_comp_key, smooth_fac=0.9, ylim_top = 1.0, ylim_bottom=0.0):
    print("------------------------------")
    print("Plotting errors rates...")
    for model_idx in range(len(models)):
        # Selecting columns of interest
        df_acc     = dfs[model_idx]["binary_accuracy"]
        df_acc_val = dfs[model_idx]["val_binary_accuracy"]

        train_error_avg = []
        for i in range(len(df_acc_val)):
            train_error_avg.append(100.0-100.0*df_acc[i])

        vali_error_avg = []
        for i in range(len(df_acc_val)):
            vali_error_avg.append(100.0-100.0*df_acc_val[i])
        
        plt.plot(smooth_curve(train_error_avg, factor=smooth_fac), label="Train: {}".format(jsons[model_idx][json_comp_key]), color=colors[model_idx], linewidth=1)
        plt.plot(smooth_curve(vali_error_avg, factor=smooth_fac), label="Val: {}".format(jsons[model_idx][json_comp_key]), color=colors[model_idx], linewidth=3)
        plt.title("Error Plot")
        plt.ylabel("Error (%)")
        plt.xlabel("Trained Chunks")
    
    # Maximize current figure before saving
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.ylim(top=ylim_top)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=ylim_bottom)  # adjust the bottom leaving top unchanged
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 8)       # (12,8), seems quite fine
    plt.savefig(os.path.join(models[model_idx], "error_train_vali"), dpi=100)
    plt.show()
    if verbatim:
        print("done Plotting Errors")


# plots the macro_f1-softloss (score) of given models. 
def make_f1_plot(models, dfs, jsons, json_comp_key, smooth_fac=0.9, ylim_top = 1.0, ylim_bottom=0.0):
    print("------------------------------")
    print("Plotting f1-softloss scores...")
    for model_idx in range(len(models)):
        # Selecting columns of interest
        df_macro_f1     = dfs[model_idx]["macro_f1"]
        df_macro_f1_val = dfs[model_idx]["val_macro_f1"]

        train_f1_avg = []
        for i in range(len(df_macro_f1)):
            train_f1_avg.append(df_macro_f1[i])

        val_f1_avg = []
        for idx in range(len(df_macro_f1_val)):
            val_f1_avg.append(df_macro_f1_val[idx])

        plt.plot(smooth_curve(train_f1_avg, factor=smooth_fac), label="Train: {}".format(jsons[model_idx][json_comp_key]), color=colors[model_idx], linewidth=1)
        plt.plot(smooth_curve(val_f1_avg, factor=smooth_fac), label="Val: {}".format(jsons[model_idx][json_comp_key]), color=colors[model_idx], linewidth=3)

        plt.title('Training and validation Macro F1-score')
        plt.ylabel('Macro F1-Score')
        plt.xlabel("Trained Chunks")
    
    # Maximize current figure before saving
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.ylim(top=ylim_top)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=ylim_bottom)  # adjust the bottom leaving top unchanged
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.legend()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 8)       # (12,8), seems quite fine
    plt.savefig(os.path.join(models[model_idx], "_f1-softloss_"), dpi=100)
    plt.show()
    if verbatim:
        print("done Plotting f1-softlosses")


# Loops over a set of columns and formats it for the user.
# The user can choose which columns to plot.
def which_plots_to_plot(columns):
    print("------------------------------")
    print("Set plots that you want to view:")
    for idx, folder in enumerate(columns):
        print("\t{} - {}".format(idx, folder))
    plot_idxs = input("Set indexes of plot(s)\n(integer)\nOr comma seperated ints: ")

    str_indexes = plot_idxs.split(',')
    chosen_plots = [columns[int(string_idx)] for string_idx in str_indexes]

    print("\nUser Choices: ")
    for chosen_plot in chosen_plots:
        print(chosen_plot)

    return chosen_plots


# Ask the user whether he want to see and compute the error plot.
def error_plot_dialog():
    print("------------------------------")
    show_error_plot = input("Show error plot? y/n: ")
    show_error_plot = True if show_error_plot in ["y", "yes", "Y", "1"] else False
    return show_error_plot

# Ask the user whether he want to see and compute the loss plot.
def loss_plot_dialog():
    print("------------------------------")
    show_loss_plot = input("Show loss plot? y/n: ")
    show_loss_plot = True if show_loss_plot in ["y", "yes", "Y", "1"] else False
    return show_loss_plot

# Ask the user whether he want to see and compute the fbeta plot.
def fBeta_plot_dialog():
    print("------------------------------")
    show_fBeta_plot = input("Show f_beta graphs? y/n: ")
    show_fBeta_plot = True if show_fBeta_plot in ["y", "yes", "Y", "1"] else False
    return show_fBeta_plot


# Ask the user whether he want to see and compute the f1 plot.
def f1_plot_dialog():
    print("------------------------------")
    show_f1_plot = input("Show f1-macro score plots? y/n: ")
    show_f1_plot = True if show_f1_plot in ["y", "yes", "Y", "1"] else False
    return show_f1_plot

# Ask the user whether he want to see and compute the f1 plot.
def many_plot_dialog():
    print("------------------------------")
    show_wide_array_plots = input("Show wide array of plots? y/n: ")
    show_wide_array_plots = True if show_wide_array_plots in ["y", "yes", "Y", "1"] else False
    return show_wide_array_plots


############## Parameters ##############
## Set colors to be used in all plots
colors                  = ['r', 'c', 'green', 'orange', 'lawngreen', 'b', 'plum', 'darkturquoise', 'm']
json_comp_key           = "model_name"              # is the label in generated plots
verbatim                = False

#Exponential Moving Average factor range=<0.0, 1.0>, the higher the factor the more smoothing wilload
smooth_fac = 0.999

### Error Plot of given Models
ytop                    = 100.0    # Error plot y upper-limit in percentage
ybottom                 = 0.00    # Error plot y bottom-limit in percentage

### f_beta graph and its paramters
# Shows a f_beta plot of the given models (can be time consuming)
f_beta_avg_count        = 3                                    # How many chunks should be evaluated, over which the mean and standard deviation will be calculated
beta_squarred           = 0.03                                  # For f-beta calculation
stepsize                = 0.01                                  # For f-beta calculation
threshold_range         = np.arange(stepsize, 1.0, stepsize)    # For f-beta calculation
######################################################


def main():
    if __name__== "__main__" :
        
        # Folder which experiment to run
        root_folder       = "models"
        experiment_folder = set_experiment_folder(root_folder)

        # Models to compare against each other
        models_paths_list = set_models_folders(experiment_folder)

        # Determine whether an Enrico model was chosen. if so, then we cannot plot, certain plots due to not having collected the data for the plot.
        is_enrico_model_chosen = True if len([x for x in models_paths_list if "resnet_single_newtr_last_last_weights_only" in x]) > 0 else False

        ## 1.0 - Get list dataframes
        copy_models_paths_lists = [ x for x in models_paths_list if "resnet_single_newtr_last_last_weights_only" not in x ] # we don't want enrico's model to be a dataframe, due to it not having logged statistics.
        dfs, _ = get_dataframes(copy_models_paths_lists)

        ## 2.0 - Get list of jsons
        jsons, _ = get_jsons(models_paths_list)

        ## 3.0 - get list of .h5 files
        paths_h5s = get_h5s_paths(models_paths_list)

        ## 4.0 - Plot Error for all models given
        if not is_enrico_model_chosen and error_plot_dialog():
            plot_errors(models_paths_list, dfs, jsons, json_comp_key, smooth_fac=smooth_fac, ylim_top = 60.0, ylim_bottom=0.0)

        ## 5.0 - Show the losses nicely for each model
        if not is_enrico_model_chosen and loss_plot_dialog():
            plot_losses_avg(models_paths_list, dfs, jsons, smooth_fac=smooth_fac)

        ## 6.0 - Show the macro-f1 score plot for each model:
        if not is_enrico_model_chosen and f1_plot_dialog():
            make_f1_plot(models_paths_list, dfs, jsons, json_comp_key, smooth_fac=smooth_fac, ylim_top = 1.0, ylim_bottom=0.0)

        ## 6.0 - Plot the data from the csvs - legend determined by json parameter dump file
        if not is_enrico_model_chosen and many_plot_dialog():
            plots_to_show = which_plots_to_plot(dfs[0].columns)
            for columnname in dfs[0].columns:
                if columnname not in plots_to_show:
                    continue
                compare_plot_models(columnname, dfs, jsons, json_comp_key)

        ## 7.0 - Calculate f-beta score per model - based on validation data
        if fBeta_plot_dialog():
            store_fbeta_results(models_paths_list, paths_h5s, jsons, json_comp_key, f_beta_avg_count)
        ######################################################

main()
