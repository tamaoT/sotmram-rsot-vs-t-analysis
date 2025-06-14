#%%
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ファイルパスを取得
file_paths = glob.glob(
    r"\\oslo\share\133-SPINTEC\133.2-Equipes\133.2.1-MRAM\133.2.1.2-People\Tamao\RsotvsT\*\Rsot_IVDC_bias*uA\*.dat"
)

# 各デバイスとバイアスで整理するリスト
data_list = []

for file_path in file_paths:
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    df = df[df['Amplitude_SOT'] <= 900]
    num_loops = len(df)//200
    # ファイル名からデバイス名を取得（例: "BTMA60B_3_T300K.dat" → "BTMA60B_3"）
    filename = os.path.basename(file_path)
    device_match = re.match(r'([A-Za-z0-9_]+)', filename)
    device = device_match.group(1) if device_match else "unknown_device"

    # 親フォルダ名から bias を取得（例: "Rsot_IVDC_bias100uA" → 100.0）
    parent_dir = os.path.basename(os.path.dirname(file_path))
    bias_match = re.search(r'bias(\d+(?:\.\d+)?)uA', parent_dir)
    bias = float(bias_match.group(1)) if bias_match else None

    # 情報を辞書として保存
    data_list.append({
        "device": device,
        "bias": bias,
        "df": df
    })

#%%
print(data_list[0]["bias"])
# %%
#plot all loops

# プロットの準備
plt.figure(figsize=(10, 6))

Rsot_bias_mean_array = []
# 各ループごとに処理
for i in range(num_loops):
    loop_data = df.iloc[i*200:(i+1)*200]  # ループごとの200行
    Rsot_bias_mean = loop_data["Rsot_bias"].mean()# ループごとの平均を計算
    Rsot_bias_mean_array.append(Rsot_bias_mean)
    #print(f"Loop {i+1}: Rsot_bias mean = {Rsot_bias_mean:.2f} Ohm")
    
    # Rsotのプロット（必要であれば）
    plt.plot(loop_data['Amplitude_SOT'], loop_data['Rsot'], alpha=0.2, marker='o', linewidth=0.8)
Rsot_bias_mean_of_T30 = np.mean(Rsot_bias_mean_array)
print("Rsot bias mean of each loop:", Rsot_bias_mean_array)
print("mean of the each loop(mean of T30):", Rsot_bias_mean_of_T30)
# プロット表示
plt.xlabel('Amplitude SOT (μA)')
plt.ylabel('Rsot (Ohm)')
plt.title('Rsot vs Amplitude SOT')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#one average loop of pulse

# Amplitude_SOTごとにグループ化し、Rsotの平均を取る
mean_df = df.groupby('Amplitude_SOT', as_index=False)['Rsot'].mean()

# プロット
plt.figure(figsize=(10, 6))
plt.plot(mean_df['Amplitude_SOT'], mean_df['Rsot'], color='red', linewidth=2, label='Average Rsot')
plt.xlabel('Amplitude SOT (μA)')
plt.ylabel('Rsot (Ohm)')
plt.title('Average of Rsot vs Amplitude SOT')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
#Rsot bias vs T


# 各ファイルの温度とバイアス平均を格納するリスト
temperatures = []
bias_means = []

# ファイルごとに処理
for path in file_paths:
    # ファイル名から温度を抽出（例：T30.1 → 30.1）
    base = os.path.basename(path)
    temp_str = base.split("T")[-1].replace(".dat", "")
    temperature = float(temp_str)

    df = pd.read_csv(path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    

    num_loops = len(df) // 200
    Rsot_bias_mean_array = []

    for i in range(num_loops):
        loop_data = df.iloc[i*200:(i+1)*200]
        Rsot_bias_mean = loop_data["Rsot_bias"].mean()
        Rsot_bias_mean_array.append(Rsot_bias_mean)

    Rsot_bias_mean_of_file = np.mean(Rsot_bias_mean_array)

    
    temperatures.append(temperature)
    bias_means.append(Rsot_bias_mean_of_file)
    


# 最後にプロット（任意）
plt.plot(temperatures, bias_means, "o")

plt.xlabel("Temperature (K)")
plt.ylabel("Average Rsot_bias (Ohm)")
plt.title("Temperature vs Rsot_bias")
plt.grid(True)
plt.show()



print("Temperatures:", temperatures)
print("Bias means:", bias_means)
#%%
# Rsot pulse(50 and 100) vs T, and compare to bias
rsot_at_50 = []
rsot_at_100 = []

for file_path in file_paths:
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    df_50 = df[df["Amplitude_SOT"] == 50]
    df_100 = df[df["Amplitude_SOT"] == 100]

    if not df_50.empty:
        rsot_at_50.append(df_50["Rsot"].mean())
    # else:
    #     rsot_at_50.append(float('nan')) 

    if not df_100.empty:
        rsot_at_100.append(df_100["Rsot"].mean())
    # else:
    #     rsot_at_100.append(float('nan'))

plt.plot(temperatures, bias_means, "x", label="Rsot bias") 
plt.plot(temperatures, rsot_at_50, 'o', label='Rsot pulse at Amplitude_SOT = 50')
plt.plot(temperatures, rsot_at_100, 's', label='Rsot pulse at Amplitude_SOT = 100')

plt.xlabel("Temperature (K)")
plt.ylabel("Rsot")
plt.title("Rsot_pulse at Amplitude_SOT = 50 & 100 vs Temperature, and Rsot_bias vs Temperature")
plt.grid(True)
plt.legend()
plt.show()

#%%
#fit to Rsot pulse
from scipy.optimize import curve_fit

# フィット関数の定義（非線形）
def nonlinear_func(T, R0, a, n, b):
    return R0 + a * T**n - b * np.log(T)

# numpy 配列に変換
T_array = np.array(temperatures)
R_array = np.array(rsot_at_50)
T_sorted_index = np.argsort(T_array)
T_sorted = T_array[T_sorted_index]
R_sorted = R_array[T_sorted_index]
# -------------------------
# 非線形フィット（初期値が必要）
initial_guess = [500, 1, 1, 1]  # [R0, a, n, b]
popt_nonlin, _ = curve_fit(nonlinear_func, T_sorted, R_sorted, p0=initial_guess)
fit_nonlin = nonlinear_func(T_sorted, *popt_nonlin)

# -------------------------
# プロット
plt.plot(T_sorted, R_sorted, "o", label="Data")
plt.plot(T_sorted, fit_nonlin, "-", label="Nonlinear Fit")

plt.xlabel("Temperature (K)")
plt.ylabel("Average Rsot_pulse (Ohm)")
plt.title("Temperature vs Rsot_pulse (with fits)")
plt.legend()
plt.grid(True)
plt.show()

# 結果の出力
print("Nonlinear fit parameters: R0 = {:.4f}, a = {:.4f}, n = {:.4f}, b = {:.4f}".format(*popt_nonlin))

#%%
#fit by non linear for Rsot bias
from scipy.optimize import curve_fit

# フィット関数の定義（非線形）
def nonlinear_func(T, R0, a, n, b):
    return R0 + a * T**n - b * np.log(T)

# 非線形フィット（初期値が必要）
initial_guess = [500, 1, 1, 1]  # [R0, a, n, b]
popt_nonlin, _ = curve_fit(nonlinear_func, T_sorted, R_sorted, p0=initial_guess)
fit_nonlin = nonlinear_func(T_sorted, *popt_nonlin)

# プロット
plt.plot(T_sorted, R_sorted, "o", label="Data")
plt.plot(T_sorted, fit_nonlin, "-", label="Nonlinear Fit")
plt.xlabel("Temperature (K)")
plt.ylabel("Average Rsot_bias (Ohm)")
plt.title("Temperature vs Rsot_bias (with fits)")
plt.legend()
plt.grid(True)
plt.show()

# 結果の出力
print("Nonlinear fit parameters: R0 = {:.4f}, a = {:.4f}, n = {:.4f}, b = {:.4f}".format(*popt_nonlin))

#%%

#Just one file loop 
# # ファイルパス（1温度）
# file_paths = glob.glob(r"\\oslo\share\133-SPINTEC\133.2-Equipes\133.2.1-MRAM\133.2.1.2-People\Tamao\BTMA T50_\BTMA60B_3_B0.0623_Bias0.00005_T90.0.dat")

# plt.figure(figsize=(10, 6))

# for file_path in file_paths:
#     df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
#     df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    
#     # 元データをコピーしておく
#     #df_copy = df.copy()

#     # 対称平均のための処理
#     df_pos = df[df['Amplitude_SOT'] > 0].copy()
#     df_neg = df[df['Amplitude_SOT'] < 0].copy()
#     df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()  # マイナス側を右に移動
    
#     # Amplitudeごとに平均Rsotを計算
#     pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
#     neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()
    
#     # プラス・マイナスのRsotを揃えて平均
#     common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))
#     avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]
    
#     # ファイル名から温度取得
#     filename = os.path.basename(file_path)
#     temp_match = re.search(r'T(\d+(?:\.\d+)?)', filename)
#     label = f"T{temp_match.group(1)}" if temp_match else filename
    
#     # プロット
#     plt.plot(common_amp, avg_rsot, label=label)

# # グラフ設定
# plt.xlabel('Isot (μA)')
# plt.ylabel('Symmetric averaged Rsot (Ohm)')
# plt.title('Symmetric Averaged Rsot vs Isot')
# plt.legend()
# plt.grid(True)
# plt.show()
#%%
#absolute and average(changed)

for file_path in file_paths:
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    
    # --- 追加：Amplitude_SOT の絶対値が900以下のみ ---
    df = df[df['Amplitude_SOT'].abs() <= 900]
    
    # プラス・マイナスで分けて、それぞれ絶対値で揃える
    df_pos = df[df['Amplitude_SOT'] > 0]
    df_neg = df[df['Amplitude_SOT'] < 0]
    df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()  # -x → x に

    # Amplitude_SOT ごとに Rsot を平均
    pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
    neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()

    # 共通の Amplitude_SOT のみに絞る
    common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))
    avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]

    # --- ラベルに温度情報を入れる ---
    filename = os.path.basename(file_path)
    temp_match = re.search(r'T(\d+(?:\.\d+)?)', filename)
    label = f"T={temp_match.group(1)}K" if temp_match else filename
    
    # --- プロット ---
    plt.plot(common_amp, avg_rsot, label=label)

# グラフの仕上げ
plt.xlabel('Isot (μA)')
plt.ylabel('Averaged Rsot (Ohm)')
#plt.title("Averaged Rsot vs Isot")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
#%%
# ##convert y axis into T(all temp)
# # Rsot → Temperature に変換
# import os
# from scipy.optimize import fsolve

# # パラメータ
# R0 = 504.4676
# a = 0.2119
# n = 0.8976
# b = 5.3713

# # データファイル
# #file_paths = glob.glob(r"\\oslo\share\133-SPINTEC\133.2-Equipes\133.2.1-MRAM\133.2.1.2-People\Tamao\BTMA T50_\*.dat")
# file_paths = glob.glob(r"\\oslo\share\133-SPINTEC\133.2-Equipes\133.2.1-MRAM\133.2.1.2-People\Tamao\RsotvsT\BTMA60B_3\Rsot_IVDC_bias50uA\*.dat")

# for file_path in file_paths:
#     # データ読み込み
#     df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
#     df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
#     df = df[df['Amplitude_SOT'].abs() <= 900]

#     df_pos = df[df['Amplitude_SOT'] > 0]
#     df_neg = df[df['Amplitude_SOT'] < 0].copy()
#     df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()

#     pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
#     neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()

#     common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))
#     avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]
    

#     # 温度推定
#     def RvsT_Fit(T, R_obs):
#         return R0 + a * T**n - b * np.log(T) - R_obs
            
#     # xとyの数を合わせるため、avg_rsotに対してループ
#     results = []
#     for R_obs in avg_rsot:
#         try:
#             T_sol = fsolve(RvsT_Fit, args=(R_obs))[0]
#             if 0 < T_sol < 400:
#                 results.append(T_sol)
#             else:
#                 results.append(np.nan)
#         except:
#             results.append(np.nan)


#     # ラベル
#     filename = os.path.basename(file_path)
#     temp_match = re.search(r'T(\d+(?:\.\d+)?)', filename)
#     label = f"T{temp_match.group(1)}" if temp_match else filename

#     # グラフ生成（ファイルごとに1枚）
#     plt.figure(figsize=(8, 5))
#     plt.plot(common_amp, results, marker='o')
#     plt.xlabel("Isot (Amplitude_SOT)")
#     plt.ylabel("Estimated Temperature (K)")
#     plt.title(f"Estimated Temperature vs Isot\n{label}")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#%%
from scipy.optimize import fsolve
#all estimated temperatures

params_1 = {'R0': 503.3878, 'a': 0.2260, 'n': 0.8910, 'b': 5.5296}
params_2 = {'R0': 504.4676, 'a': 0.2119, 'n': 0.8976, 'b': 5.3713} 

def estimate_temperature(R_obs, guess, params):
    def RvsT_Fit(T):
        R0 = params['R0']
        a = params['a']
        n = params['n']
        b = params['b']
        return R0 + a * T**n - b * np.log(T) - R_obs
    try:
        sol = fsolve(RvsT_Fit, guess)[0]
        return sol if 0 < sol < 400 else np.nan
    except:
        return np.nan

# 1枚のグラフにまとめてプロット

for file_path in file_paths:
    # ファイル名から温度を抽出
    filename = os.path.basename(file_path)
    temp_match = re.search(r'T(\d+(?:\.\d+)?)', filename)
    if not temp_match:
        continue
    nominal_temp = temp_match.group(1)

    #dont remove it
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']
    df = df[df['Amplitude_SOT'].abs() <= 900]

    # 正負で分けて平均化
    df_pos = df[df['Amplitude_SOT'] > 0]
    df_neg = df[df['Amplitude_SOT'] < 0].copy()
    df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()

    pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
    neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()

    common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))
    avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]

    # 推定温度リスト
    #results = []
    #guess = float(nominal_temp)

    # for R_obs in avg_rsot:
    #     try:
    #         T_sol = fsolve(estimate_temperature(R_obs, guess, params_1), guess, args=(R_obs))[0]
    #         if 0 < T_sol < 400:
    #             results.append(T_sol)
    #         else:
    #             results.append(np.nan)
    #     except:
    #         results.append(np.nan)
    
    results1 = [estimate_temperature(R, file_temp, params_1) for R in avg_rsot]
    results2 = [estimate_temperature(R, file_temp, params_2) for R in avg_rsot]
    
        # プロット追加
    plt.plot(common_amp, results1, marker='o', label=f"T={nominal_temp}K")

# グラフ設定
plt.xlabel("Isot (Amplitude_SOT)")
plt.ylabel("Estimated Temperature (K)")
#plt.title("Estimated Temperature vs Isot for each Temperature")
plt.grid(True)
plt.legend(title="Nominal Temp")
plt.tight_layout()
plt.show()

#%%
#view the difference

# 対象温度を指定
selected_temperature = 300

for file_path in file_paths:
    # ファイル名から温度を取得
    filename = os.path.basename(file_path)
    temp_match = re.search(r'T(\d+(?:\.\d+)?)', filename)
    if not temp_match:
        continue
    file_temp = float(temp_match.group(1))
    
    # 選択した温度以外はスキップ
    if abs(file_temp - selected_temperature) > 1e-3:
        continue

    df_pos = df[df['Amplitude_SOT'] > 0]
    df_neg = df[df['Amplitude_SOT'] < 0].copy()
    df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()

    pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
    neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()

    common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))
    avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]


    # ======= プロット =======
    plt.figure(figsize=(8, 5))
    plt.plot(common_amp, results1, marker='o', label='Rsot bias')
    plt.plot(common_amp, results2, marker='x', label='Rsot pulse')
    plt.xlabel("Isot (Amplitude_SOT)")
    plt.ylabel("Estimated Temperature (K)")
    plt.title(f"Comparison of Two Fits: T{file_temp}K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#%%
#for one file
device = "BTMA60B_3"
SOTbias = 50
temperature = str(re.search(r"T(\d+(?:\.\d+)?)", os.path.basename(file_path)).group(1))
# === 温度推定の係数 ===
slope = 12.9610
intercept = -6276.05

# === データ読み込み ===
for file_path in file_paths:
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None)
    df.columns = ['Amplitude_SOT', 'V_pulse', 'Rsot', 'Rsot_bias']

    # === 条件：|Isot| <= 900 に制限 ===
    df = df[df['Amplitude_SOT'].abs() <= 900]

    # === 正負分離（絶対値合わせ）===
    df_pos = df[df['Amplitude_SOT'] > 0]
    df_neg = df[df['Amplitude_SOT'] < 0].copy()
    df_neg['Amplitude_SOT'] = df_neg['Amplitude_SOT'].abs()

    # === 共通のIsotについて Rsot平均 ===
    pos_mean = df_pos.groupby('Amplitude_SOT')['Rsot'].mean()
    neg_mean = df_neg.groupby('Amplitude_SOT')['Rsot'].mean()
    common_amp = sorted(set(pos_mean.index) & set(neg_mean.index))

    # === Rsot平均 → 温度変換 ===
    avg_rsot = [(pos_mean[amp] + neg_mean[amp]) / 2 for amp in common_amp]
    avg_temp = [slope * r + intercept for r in avg_rsot]

    # === プロット ===
    plt.figure(figsize=(8, 6))
    plt.plot(common_amp, avg_temp, 'o-', label="Estimated Temperature")

    plt.xlabel("Isot (Amplitude_SOT)")
    plt.ylabel("Estimated Temperature (K)")
    #plt.title("Isot vs Estimated Temperature ()")
    plt.title(device+', SOTbias= '+str(SOTbias)+"uA, T="+temperature)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
