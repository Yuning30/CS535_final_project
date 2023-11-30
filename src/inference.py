import torch
import numpy as np
import pdb
DEBUG = False

def compute_T_j(L_max, h_j):
    one_minus_h_j = 1.0 - h_j
    T_j = 0.0
    
    for k in range(0, L_max):
        prod = np.prod(one_minus_h_j[0: k+1])
        T_j += prod
    
    return T_j

def compute_V_j(L_max, j, h_j, C_alpha):
    one_minus_h_j = 1.0 - h_j
    arr = []
    for l in range(0, L_max - j):
        first_term = 0.0
        for k in range(0, L_max - l):
            prod = np.prod(one_minus_h_j[0:k+l+1])
            first_term = first_term + prod

        second_term = 0.0
        for m in range(0, l + 1):
            prod = np.prod(one_minus_h_j[0:m])
            prod = prod * h_j[m]
            second_term += prod
        
        if DEBUG:
            print(f"first term {first_term}, second term {second_term}")
        arr.append(first_term + C_alpha * second_term)

    return np.min(arr)

def inference_one_series(L_max, series_with_label, model, C_alpha):
    series = torch.tensor(series_with_label).float()[:, :-1]
    label = torch.tensor(series_with_label).float()[:, -1]
    # print(label)
    success = None
    time_to_event = None
    # pdb.set_trace()
    for j in range(0, L_max):
        features = series[0: j+1]
        h_j = model.compute_h_j(L_max, j, features)
        # print(h_j)
        T_j = compute_T_j(L_max, h_j)
        V_j = compute_V_j(L_max, j, h_j, C_alpha)
        if DEBUG:
            print(f"T_j: {T_j}, V_j: {V_j}")
        # pdb.set_trace()
        intervene = (T_j <= V_j)
        if label[j] == 0:
            if intervene:
                success = True
                
                for t in range(0, len(label)):
                    if label[t] == 1.0:
                        event_time = t
                        break
                time_to_event = event_time - j
                assert time_to_event > 0
                print("success interveved")
                break
        elif label[j] == 1:
            success = False
            time_to_event = 0
            print("missed the event")
            break
        else:
            assert False
    assert success is not None
    assert time_to_event is not None
    return success, time_to_event

def inference_test_data(L_max, test_dataset, model, C_alpha):
    success_count = 0
    sum_tte = 0
    for i in range(0, len(test_dataset)):
        series_with_label = test_dataset[i]
        success, tte = inference_one_series(L_max, series_with_label, model, C_alpha)
        if success:
            success_count += 1
            sum_tte += tte
    
    success_rate = success_count / len(test_dataset)
    avg_tte = sum_tte / success_count
    print(f"success rate is {success_rate}")
    print(f"ave tte is {avg_tte}")

