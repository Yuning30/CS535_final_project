import inference
import dataset
import train
import pdb
import pickle

if __name__ == "__main__":
    # model_path = "ddrsa_20/epoch=1297-val_loss=0.3985.ckpt"
    model_path = "ddrsa_20_less_64_full/epoch=646-val_loss=2.1397.ckpt"
    print("#### start process the data ####")
    train_dataset, valid_dataset, test_dataset = dataset.process_data(
        "../AMLWorkshop/Data/features_15h.csv"
    )
    print("#### finish process the data ####")
    with open("testdataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    L_max = 0
    for x, y in test_dataset:
        # pdb.set_trace()
        L_max = max(len(x), L_max)

    print("#### start loading the model ####")
    model = train.DDRSA.load_from_checkpoint(model_path, feature_size=16, learning_rate=0.01)
    print("#### finish loading the model ####")
    
    # pdb.set_trace()
    # for c in range(8, 268, 20):
    for c in [8, 128, 256]:
        print(f"=== inference for C_alpha = {c}", flush=True)
        inference.inference_test_data(64, test_dataset, model, C_alpha=c, generate_plot_stats=False)



