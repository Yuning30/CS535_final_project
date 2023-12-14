import matplotlib.pyplot as plt
import pdb
import json

import parse

def plot_2b(log_file):
    C_alpha = []
    Tj_curves = []
    Vj_curves = []
    Tj_x, Tj_y, Vj_x, Vj_y = [], [], [], []

    format_str = "j: {:d} T_j: {:f}, V_j: {:f}\n"
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if "C_alpha" in line:
                # pdb.set_trace()
                idx = line.rindex("=")
                C_alpha.append(int(line[idx+1:]))
            elif line.startswith("j"):
                # parse the line
                j, T_j, V_j = parse.parse(format_str, line)
                if j == 0:
                    if len(Tj_x) > 0:
                        Tj_curves.append((Tj_x, Tj_y))
                        Vj_curves.append((Vj_x, Vj_y))
                    Tj_x, Tj_y, Vj_x, Vj_y = [], [], [], []
                Tj_x.append(j)
                Tj_y.append(T_j)
                Vj_x.append(j)
                Vj_y.append(V_j)
        Tj_curves.append((Tj_x, Tj_y))
        Vj_curves.append((Vj_x, Vj_y))
    
    Tj_x, Tj_y = Tj_curves[0]
    plt.plot(Tj_x, Tj_y, label=f"T_j")
    
    for c, (Vj_x, Vj_y) in zip(C_alpha, Vj_curves):
        plt.plot(Vj_x, Vj_y, label=f"V_j (C_alpha = {c})")

    plt.xlabel("time steps")
    plt.ylabel("risks")    
    plt.legend()
    plt.savefig("plot_2b.png")

def plot_2a(log_file):
    j_list = []
    hj_list = []
    format_str = "j = {:d}, h_j = {}\n"
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("j"):
                out = parse.parse(format_str, line)
                j_list.append(out[0])
                hj_list.append(json.loads(out[1]))
    
    L_max = len(hj_list[0])
    pdb.set_trace()
    for j in range(0, 13, 3):
        hj_y = hj_list[j]
        hj_x = [i for i in range(0, len(hj_y))]
        plt.plot(hj_x, hj_y, label=f"j = {j}")
    
    plt.xlabel("time steps from j")
    plt.ylabel("hazard rate")
    plt.legend()
    plt.savefig("plot_2a.png")

def plot_3(wbi_log, ddrsa_log):
    wbi_miss_rate = []
    wbi_avg_tte = []
    wbi_log_fmt = "bset success rate {:f}, bset avg tte {:f}\n"
    with open(wbi_log) as wbi_file:
        lines = wbi_file.readlines()
        for line in lines:
            if "bset success rate" in line:
                out = parse.parse(wbi_log_fmt, line)
                wbi_miss_rate.append(1 - out[0])
                wbi_avg_tte.append(out[1])
    plt.plot(wbi_miss_rate, wbi_avg_tte, "o-", label="wbi")
    
    ddrsa_miss_rate = []
    ddrsa_avg_tte = []

    with open(ddrsa_log) as ddrsa_file:
        lines = ddrsa_file.readlines()
        for line in lines:
            if "success rate" in line:
                out = parse.parse("success rate is {:f}\n", line)
                ddrsa_miss_rate.append(1 - out[0])
            elif "ave tte" in line:
                out = parse.parse("ave tte is {:f}\n", line)
                ddrsa_avg_tte.append(out[0])
    
    plt.plot(ddrsa_miss_rate, ddrsa_avg_tte, "o-", label="ddrsa")

    plt.xlabel("event miss rate")
    plt.ylabel("average time to event")
    plt.legend()
    plt.savefig("plot_3.png")







if __name__ == "__main__":
    plot_2a("2a_test_from_pkl_file.txt")
    # plot_2b("2b_test_from_pkl_file.txt")
    # plot_3("a.txt", "x_test_from_pkl_file.txt")