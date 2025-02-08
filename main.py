#!/usr/bin/env python
import os
import sys
import joblib
import torch


def main():
    print("\n[INFO] Welcome to the Model Trainer/Evaluator")
    print("[INFO] ----------------------------------------")
    print("[INFO] Please select the task:")
    print("       1. Task A (BreastMNIST)")
    print("       2. Task B (BloodMNIST)")
    task_choice = input("Enter 1 or 2: ").strip()

    # ----- TASK SELECTION -----
    if task_choice == "1":
        task = "A"
        print("\n[INFO] Task A (BreastMNIST) selected.")
        print("[INFO] Loading BreastMNIST dataset using utils_A...")
        try:
            from A.utils_A import prepare_breastmnist_data
        except ImportError:
            print("[ERROR] Could not import prepare_breastmnist_data from utils_A. Exiting.")
            sys.exit(1)
            # Download/load BreastMNIST
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_breastmnist_data()
    elif task_choice == "2":
        task = "B"
        print("\n[INFO] Task B (BloodMNIST) selected.")
        print("[INFO] Attempting to load BloodMNIST dataset using medmnist...")
        try:
            from medmnist import BloodMNIST
        except ImportError:
            print("[ERROR] medmnist module is not installed. Please install it (e.g. via pip install medmnist) and try again.")
            sys.exit(1)

        # Download/load BloodMNIST
        try:
            train_dataset = BloodMNIST(split='train', download=True)
            val_dataset   = BloodMNIST(split='val', download=True)
            test_dataset  = BloodMNIST(split='test', download=True)
        except Exception as e:
            print(f"[ERROR] Could not load BloodMNIST dataset: {e}")
            sys.exit(1)

        print("[INFO] Extracting images and labels from BloodMNIST dataset...")
        # Obtaining images and labels from dataset
        x_train = train_dataset.imgs
        y_train = train_dataset.labels
        x_val   = val_dataset.imgs
        y_val   = val_dataset.labels
        x_test  = test_dataset.imgs
        y_test  = test_dataset.labels

        # For ANN and CNN, we need the prepared tensors.
        try:
            from B.utils_B import prepare_data_bloodmnist
        except ImportError:
            print("[ERROR] Could not import prepare_data_bloodmnist from utils_B. Exiting.")
            sys.exit(1)
        # Prepare tensor data for ANN and CNN (for SVM we use the raw arrays)
        X_train_b, Y_train_b, X_val_b, Y_val_b, X_test_b, Y_test_b = prepare_data_bloodmnist(
            x_train, y_train, x_val, y_val, x_test, y_test
        )
    else:
        print("\n[ERROR] Invalid task selection. Exiting.")
        sys.exit(1)

    # ----- MODEL SELECTION -----
    print("\n[INFO] Please select the model:")
    print("       1. SVM")
    print("       2. ANN")
    print("       3. CNN")
    model_choice = input("Enter 1, 2, or 3: ").strip()
    if model_choice == "1":
        model_type = "svm"
    elif model_choice == "2":
        model_type = "ann"
    elif model_choice == "3":
        model_type = "cnn"
    else:
        print("\n[ERROR] Invalid model selection. Exiting.")
        sys.exit(1)

    # ----- OPERATION SELECTION -----
    print("\n[INFO] Please select the operation:")
    print("       1. Train and Evaluate")
    print("       2. Load and Evaluate (using pre-saved model from Results folder)")
    op_choice = input("Enter 1 or 2: ").strip()
    if op_choice not in ["1", "2"]:
        print("\n[ERROR] Invalid operation selection. Exiting.")
        sys.exit(1)

    # Define the file name for the model in the Results folder.
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_filename = f"best_{model_type}_model_{task}"
    # SVM models are saved as .joblib but ANN and CNN is .pth
    if model_type == "svm":
        model_filepath = os.path.join(results_dir, model_filename + ".joblib")
    else:
        model_filepath = os.path.join(results_dir, model_filename + ".pth")

    # ----- OPERATION: TRAIN & EVALUATE -----
    if op_choice == "1":
        print(f"\n[INFO] Starting training for {model_type.upper()} on Task {task} dataset...")
        if task == "A":
            if model_type == "svm":
                from A.SVM_A import preprocess_for_svm, train_svm_A, evaluate_svm_A
                print("[INFO] Preprocessing data for SVM (Task A)...")
                x_train_svm, y_train_svm, x_val_svm, y_val_svm, x_test_svm, y_test_svm = preprocess_for_svm(
                    x_train, y_train, x_val, y_val, x_test, y_test
                )
                print("[INFO] Training SVM model (Task A)...")
                svm_model = train_svm_A(x_train_svm, y_train_svm, use_gridsearch=False)
                print("[INFO] Evaluating SVM model on validation and test sets (Task A)...")
                eval_results = evaluate_svm_A(svm_model, x_val_svm, y_val_svm, x_test_svm, y_test_svm)

            elif model_type == "ann":
                from A.ANN_A import prepare_data_ann, train_ann_A, evaluate_ann_A
                print("[INFO] Preparing data for ANN (Task A)...")
                X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_ann(
                    x_train, y_train, x_val, y_val, x_test, y_test
                )
                print("[INFO] Training ANN model (Task A)...")
                ann_model, best_params = train_ann_A(X_train, Y_train, X_val, Y_val, use_optuna=False)
                print("[INFO] Evaluating ANN model on test set (Task A)...")
                eval_results = evaluate_ann_A(ann_model, X_test, Y_test, use_best=False)

            elif model_type == "cnn":
                from A.CNN_A import prepare_data_cnn, train_cnn_A, evaluate_cnn_A
                print("[INFO] Preparing data for CNN (Task A)...")
                X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_cnn(
                    x_train, y_train, x_val, y_val, x_test, y_test
                )
                print("[INFO] Training CNN model (Task A)...")
                cnn_model, best_params = train_cnn_A(X_train, Y_train, X_val, Y_val, use_optuna=False)
                print("[INFO] Evaluating CNN model on test set (Task A)...")
                eval_results = evaluate_cnn_A(cnn_model, X_test, Y_test, use_best=False)

        elif task == "B":
            if model_type == "svm":
                from B.SVM_B import preprocess_for_svm_B, train_svm_B, evaluate_svm_B
                print("[INFO] Preprocessing data for SVM (Task B, BloodMNIST)...")
                # For SVM we use the raw arrays (x_train, etc.)
                x_train_svm, y_train_svm, x_val_svm, y_val_svm, x_test_svm, y_test_svm = preprocess_for_svm_B(
                    x_train, y_train, x_val, y_val, x_test, y_test
                )
                print("[INFO] Training SVM model (Task B, BloodMNIST)...")
                svm_model = train_svm_B(x_train_svm, y_train_svm, use_gridsearch=False)
                print("[INFO] Evaluating SVM model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_svm_B(svm_model, x_test_svm, y_test_svm)

            elif model_type == "ann":
                from B.ANN_B import train_ann_B, evaluate_ann_B
                print("[INFO] Training ANN model (Task B, BloodMNIST)...")
                # For ANN in Task B, use the prepared tensors (X_train_b, etc.) from utils_B.
                ann_model, best_params = train_ann_B(X_train_b, Y_train_b, X_val_b, Y_val_b, use_optuna=False)
                print("[INFO] Evaluating ANN model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_ann_B(ann_model, X_test_b, Y_test_b, use_best=False)

            elif model_type == "cnn":
                from B.CNN_B import train_cnn_B, evaluate_cnn_B, prepare_data_cnn
                print("[INFO] Preparing data for CNN (Task B)...")
                X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data_cnn(
                    x_train, y_train, x_val, y_val, x_test, y_test
                )
                print("[INFO] Training CNN model (Task B, BloodMNIST)...")
                # For CNN in Task B, use the prepared tensors (X_train_b, etc.) from utils_B.
                cnn_model, best_params = train_cnn_B(X_train, Y_train, X_val, Y_val, use_optuna=False)
                print("[INFO] Evaluating CNN model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_cnn_B(cnn_model, X_test, Y_test, use_best=False)
        else:
            print("[ERROR] Invalid task selection during training. Exiting.")
            sys.exit(1)

        print("\n[INFO] Training and evaluation completed.")
        print("[INFO] (Note: The trained model was not saved to Results; to evaluate a saved model use the Load option.)")

    # ----- OPERATION: LOAD & EVALUATE -----
    else:
        print(f"\n[INFO] Loading pre-trained {model_type.upper()} model for Task {task} from:")
        print(f"       {model_filepath}")
        if not os.path.exists(model_filepath):
            print(f"[ERROR] Model file '{model_filepath}' not found in Results folder. Exiting.")
            sys.exit(1)

        if model_type == "svm":
            # Load SVM model using joblib
            svm_model = joblib.load(model_filepath)
            if task == "A":
                from A.SVM_A import evaluate_svm_A, preprocess_for_svm
                print("[INFO] Preprocessing test data for SVM (Task A)...")
                x_train_svm, y_train_svm, x_val_svm, y_val_svm, x_test_svm, y_test_svm = preprocess_for_svm(x_train, y_train, x_val, y_val, x_test, y_test)
                print("[INFO] Evaluating loaded SVM model on test set (Task A)...")
                # Using test data for both validation and test in evaluation
                eval_results = evaluate_svm_A(svm_model, x_val_svm, y_val_svm, x_test_svm, y_test_svm)
            elif task == "B":
                from B.SVM_B import evaluate_svm_B, preprocess_for_svm_B
                print("[INFO] Preprocessing test data for SVM (Task B, BloodMNIST)...")
                x_train_svm, y_train_svm, x_val_svm, y_val_svm, x_test_svm, y_test_svm = preprocess_for_svm_B(x_train, y_train, x_val, y_val, x_test, y_test)
                print("[INFO] Evaluating loaded SVM model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_svm_B(svm_model, x_test_svm, y_test_svm)

        elif model_type == "ann":
            # Load ANN model using torch.load
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ann_model = torch.load(model_filepath, weights_only=True, map_location=device)
            if task == "A":
                from A.ANN_A import prepare_data_ann, evaluate_ann_A
                print("[INFO] Preparing test data for ANN (Task A)...")
                _, _, _, _, X_test, Y_test = prepare_data_ann(x_train, y_train, x_val, y_val, x_test, y_test)
                print("[INFO] Evaluating loaded ANN model on test set (Task A)...")
                eval_results = evaluate_ann_A(ann_model, X_test, Y_test)
            elif task == "B":
                from B.ANN_B import evaluate_ann_B
                print("[INFO] Evaluating loaded ANN model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_ann_B(ann_model, X_test_b, Y_test_b)

        elif model_type == "cnn":
            # Load CNN model using torch.load
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cnn_model = torch.load(model_filepath, weights_only=True, map_location=device)
            if task == "A":
                from A.CNN_A import prepare_data_cnn, evaluate_cnn_A
                print("[INFO] Preparing test data for CNN (Task A)...")
                _, _, _, _, X_test, Y_test = prepare_data_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
                print("[INFO] Evaluating loaded CNN model on test set (Task A)...")
                eval_results = evaluate_cnn_A(cnn_model, X_test, Y_test)
            elif task == "B":
                from B.CNN_B import evaluate_cnn_B, prepare_data_cnn
                print("[INFO] Preparing test data for CNN (Task B)...")
                _, _, _, _, X_test, Y_test = prepare_data_cnn(x_train, y_train, x_val, y_val, x_test, y_test)
                print("[INFO] Evaluating loaded CNN model on test set (Task B, BloodMNIST)...")
                eval_results = evaluate_cnn_B(cnn_model, X_test, Y_test)
        else:
            print("[ERROR] Invalid model type during loading. Exiting.")
            sys.exit(1)

        print("\n[INFO] Model loading and evaluation completed.")

    print("\n[INFO] Process completed. Exiting.\n")


if __name__ == "__main__":
    main()
