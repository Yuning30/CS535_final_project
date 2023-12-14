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
            # pass
            print(f"first term {first_term}, second term {second_term}")
        arr.append(first_term + C_alpha * second_term)

    return np.min(arr)

def inference_one_series(L_max, series_with_label, model, C_alpha, generate_plot_stats=False):
    series, label = series_with_label
    # print(label)
    success = None
    time_to_event = None
    pdb.set_trace()
    for j in range(0, L_max):
        features = series[0: j+1]
        h_j = model.compute_h_j(L_max, j, features)

        ############ 2a stats ###########
        h_j_list = [x for x in h_j]
        print(f"j = {j}, h_j = {h_j_list}")
        ############ 2a stats ###########
        T_j = compute_T_j(L_max, h_j)
        V_j = compute_V_j(L_max, j, h_j, C_alpha)
        if DEBUG or generate_plot_stats:
            print(f"j: {j} T_j: {T_j}, V_j: {V_j}")
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

def inference_test_data(L_max, test_dataset, model, C_alpha, generate_plot_stats=False):
    success_count = 0
    sum_tte = 0
    length = len(test_dataset) if not generate_plot_stats else 1
    # pdb.set_trace()
    for i in range(0, 1):
        series_with_label = test_dataset[6]
        series, label = series_with_label
        # pdb.set_trace()
        success, tte = inference_one_series(L_max, series_with_label, model, C_alpha, generate_plot_stats=generate_plot_stats)
        if success:
            success_count += 1
            sum_tte += tte
    
    success_rate = success_count / len(test_dataset)
    if success_count > 0:
        avg_tte = sum_tte / success_count
    else:
        avg_tte = "n/a"
    print(f"success rate is {success_rate}")
    print(f"ave tte is {avg_tte}")

def inference_test_data_WBI(test_dataset, model, C_alpha, threshold):
    success_count = 0
    sum_tte = 0
    for i in range(0, len(test_dataset)):
        series_with_label = test_dataset[i]
        success, tte = inference_one_series_WBI(series_with_label, model, C_alpha, threshold)
        if success:
            success_count += 1
            sum_tte += tte
    
    success_rate = success_count / len(test_dataset)
    if sum_tte > 0:
        avg_tte = sum_tte / success_count
    else:
        avg_tte = "infinity"
    print(f"success rate is {success_rate}")
    print(f"ave tte is {avg_tte}")
    return success_rate, sum_tte


def inference_one_series_WBI(series_with_label, model, C_alpha, threshold):
    series, label = series_with_label
    # print(label)
    success = None
    time_to_event = None
    # pdb.set_trace()
    for j in range(0, len(series)):
        lower_bound = max(0, j - 128 + 1)
        features = series[lower_bound: j+1]
        _, (hidden_state, _) = model.encoder.encoder(features)
        logits = model.dense(hidden_state)
        logits = torch.squeeze(logits)
        # print(h_j)
        # pdb.set_trace()
        prob = torch.sigmoid(logits)
        if DEBUG:
            print(f"logits: {logits}, prob: {prob}")
        # pdb.set_trace()
        intervene = (prob > threshold) # because stop label is 1
        if label[j] == 0:
            if intervene:
                success = True
                
                for t in range(0, len(label)):
                    if label[t] == 1.0:
                        event_time = t
                        break
                time_to_event = event_time - j
                assert time_to_event > 0
                # print("success interveved")
                break
        elif label[j] == 1:
            success = False
            time_to_event = 0
            # print("missed the event")
            break
        else:
            assert False
    assert success is not None
    assert time_to_event is not None
    return success, time_to_event

def select_threshold_wbi(test_dataset, model, C_alpha):
    emprically_risk = None
    best_thres = None
    best_success_rate = None
    best_avg_tte = None
    for thres in np.linspace(0, 1, 101):
        success_rate, sum_tte = inference_test_data_WBI(test_dataset, model, C_alpha, thres)

        current_risk = sum_tte / len(test_dataset) + C_alpha * (1 - success_rate)
        print(f"N: {len(test_dataset)} sum {sum_tte}, first part {sum_tte / len(test_dataset)}. second_part {C_alpha * (1 - success_rate)}")
        if emprically_risk is None:
            emprically_risk = current_risk
            best_thres = thres
            best_success_rate = success_rate
            best_avg_tte = sum_tte / (len(test_dataset) * success_rate) if success_rate > 0 else "infinity"
        elif current_risk < emprically_risk:
            emprically_risk = current_risk
            best_thres = thres
            best_success_rate = success_rate
            best_avg_tte = sum_tte / (len(test_dataset) * success_rate) if success_rate > 0 else "infinity"

        print(f"threshold {thres}, risk {current_risk}")
        print("-------------------")
    print(f"for C alpha = {C_alpha}, best threshold is {best_thres}")
    print(f"bset success rate {best_success_rate}, bset avg tte {best_avg_tte}")

