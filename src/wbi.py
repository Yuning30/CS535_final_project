import inference
import dataset
import train
import pdb
import pickle
import torch

if __name__ == "__main__":
    model_path = "lightning_logs/version_8/checkpoints/epoch=73-val_loss=0.2405.ckpt"
    
    print("#### start process the data ####")
    train_dataset, valid_dataset, test_dataset = dataset.process_data(
        "../AMLWorkshop/Data/features_15h.csv", model="wbi"
    )
    print("#### finish process the data ####")

    print("#### start loading the model ####")
    model = train.WBI.load_from_checkpoint(model_path, feature_size=16, learning_rate=0.001)
    print("#### finish loading the model ####")
    
    # pdb.set_trace()
    model.eval()
    with torch.no_grad():
        for c in range(8, 258, 20):
            print(f"=== inference for C_alpha = {c}")
            inference.select_threshold_wbi(test_dataset, model, C_alpha=c)



