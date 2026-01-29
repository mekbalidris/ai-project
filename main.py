import data_preprocessing as dp
import model_training as mt
import evaluation as ev

def main():
    # part 1: data stuff
    # load data
    df = dp.load_and_explore_data('heart.csv')
    
    # clean and scale data
    df_processed = dp.preprocess_data(df)
    
    # part 2: training stuff
    # split into train and test
    x_train, x_test = mt.split_data(df_processed)
    
    # find best k
    k = mt.find_optimal_k(x_train)
    
    # wich K we have choosed
    print(f"\nwe choose K as : {k}\n")
    
    # train the model
    model = mt.train_model(x_train, k)
    
    # get labels for test set (just to satisfy requirement)
    test_labels = model.predict(x_test)
    
    # part 3: evaluation stuff
    # check metrics
    ev.evaluate_clustering(model, x_train, k)
    
    # compare with other k values
    ev.compare_k_values(x_train)
    
    # make the final plot
    ev.visualize_results(model, x_train)

if __name__ == "__main__":
    main()