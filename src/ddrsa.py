import inference
import dataset
import train
import pdb
import pickle

if __name__ == "__main__":
    model_path = "lightning_logs/version_0/checkpoints/epoch=4999-val_loss=0.0000-other_metric=0.00.ckpt"
    
    print("#### start process the data ####")
    train_dataset, valid_dataset, test_dataset = dataset.process_data(
        "../AMLWorkshop/Data/features_15h.csv"
    )
    print("#### finish process the data ####")
    with open("testdataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    print("#### start loading the model ####")
    model = train.DDRSA.load_from_checkpoint(model_path, feature_size=16, learning_rate=0.01)
    print("#### finish loading the model ####")
    
    pdb.set_trace()
    for c in range(8, 258, 2):
        print(f"=== inference for C_alpha = {c}")
        inference.inference_test_data(10, test_dataset, model, C_alpha=c)



