import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read excel files. t reads the time in column J of the excel file. N_T reads the normalized transmittance in column L of the excel file. 
t1 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\1.xlsx",
              usecols="J").to_numpy().flatten()

N_T1 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\1.xlsx",
              usecols="L").to_numpy().flatten()


t2 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\2.xlsx",
              usecols="J").to_numpy().flatten()

N_T2 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\2.xlsx",
              usecols="L").to_numpy().flatten()


t3 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\3.xlsx",
              usecols="J").to_numpy().flatten()

N_T3 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\3.xlsx",
              usecols="L").to_numpy().flatten()


t4 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\4.xlsx",
              usecols="J").to_numpy().flatten()

N_T4 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\4.xlsx",
              usecols="L").to_numpy().flatten()


t5 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\5.xlsx",
               usecols="J").to_numpy().flatten()

N_T5 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\5.xlsx",
               usecols="L").to_numpy().flatten()


t6 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\6.xlsx",
               usecols="J").to_numpy().flatten()

N_T6 = pd.read_excel(r"C:\Users\Lise\Desktop\LST\LST year 3\Bachelor research project\Data\Tartrazine experiments\Diffusion\20260116\cb\6.xlsx",
               usecols="L").to_numpy().flatten()




# function for the exponential curve fitted to the normalized transmittance
def expo(t,a,b):
    return a*(1-np.exp(-b*t))

# function to calculate R^2 of the fit
def r_squared(y,yf):
    y_data = np.array(y)
    y_fit = np.array(yf)
    ss_res = np.sum((y_data - y_fit) ** 2)              
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)   
    return 1 - (ss_res / ss_tot)

# curve fit parameters
po1, pc1 = curve_fit(expo,t1,N_T1)
po2, pc2 = curve_fit(expo,t2,N_T2)
po3, pc3 = curve_fit(expo,t3,N_T3)
po4, pc4 = curve_fit(expo,t4,N_T4)
po5, pc5 = curve_fit(expo,t5,N_T5,maxfev=9999999)
po6, pc6 = curve_fit(expo,t6,N_T6)

# Time constant of the fit and the corresponding error. 
tau1 = 1 / po1[1]
tau_err1 = np.abs(po1[1]**-2) * np.sqrt(np.diag(pc1))[1]

tau2 = 1 / po2[1]
tau_err2 = np.abs(po2[1]**-2) * np.sqrt(np.diag(pc2))[1]

tau3 = 1 / po3[1]
tau_err3 = np.abs(po3[1]**-2) * np.sqrt(np.diag(pc3))[1]

tau4 = 1 / po4[1]
tau_err4 = np.abs(po4[1]**-2) * np.sqrt(np.diag(pc4))[1]

tau5 = 1 / po5[1]
tau_err5 = np.abs(po5[1]**-2) * np.sqrt(np.diag(pc5))[1]
 
tau6 = 1 / po6[1]
tau_err6 = np.abs(po6[1]**-2) * np.sqrt(np.diag(pc6))[1]



# a is defined as 1/tau, which is equal to the curve fit parameter po1. Based on the function for the diffusion constant, D (diffusion) and the corresponding error is calculated
a1 = po1[1]
a1_err = np.sqrt(np.diag(pc1))[1]
D1 = ((4.42e-2) * (0.2**2) * a1)/60  
D1_err = ((4.42e-2) * (0.2**2) * a1_err)/60

a2 = po2[1]
a2_err = np.sqrt(np.diag(pc2))[1]
D2 = ((4.42e-2) * (0.2**2) * a2)/60
D2_err = ((4.42e-2) * (0.2**2) * a2_err)/60

a3 = po3[1]
a3_err = np.sqrt(np.diag(pc3))[1]
D3 = ((4.42e-2) * (0.2**2) * a3)/60
D3_err = ((4.42e-2) * (0.2**2) * a3_err)/60

a4 = po4[1]
a4_err = np.sqrt(np.diag(pc4))[1]
D4 = ((4.42e-2) * (0.2**2) * a4)/60
D4_err = ((4.42e-2) * (0.2**2) * a4_err)/60

a5 = po5[1]
a5_err = np.sqrt(np.diag(pc5))[1]
D5 = ((4.42e-2) * (0.2**2) * a5)/60
D5_err = ((4.42e-2) * (0.2**2) * a5_err)/60

a6 = po6[1]
a6_err = np.sqrt(np.diag(pc6))[1]
D6 = ((4.42e-2) * (0.2**2) * a6)/60
D6_err = ((4.42e-2) * (0.2**2) * a6_err)/60






# Plot all the samples individually
fig, ax = plt.subplots(figsize=(16,20), nrows=3, ncols=2)
fit_x = np.linspace(0,999,9999)
ax[0,0].plot(t1, N_T1, color="red", label="1")
ax[0,0].plot(
    fit_x,
    expo(fit_x, *po1),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T1), expo(np.array(t1), *po1)), 3)}\n"
        f"T = ({np.round(po1[0], 2)} ± {np.round(np.sqrt(np.diag(pc1))[0], 2)})"
        f"(1 - exp(-({np.round(po1[1], 2)} ± {np.round(np.sqrt(np.diag(pc1))[1], 2)})·t))\n"
        f"τ = {np.round(tau1, 2)} ± {np.round(tau_err1, 2)} (min)\n"
        f"$D=({np.round(D1*1e6, 4)}±{np.round(D1_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="red",
    linestyle="dashed",
    zorder=-1)
ax[0,0].set_xlim([0,50])
ax[0,0].set_ylim([0,1.1])
ax[0,0].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[0,0].set_ylabel(r"Transmittance $Tr$",fontsize=12)
# ax[0,0].set_title("1")
ax[0,0].tick_params(axis='both', which='major', labelsize=12)
ax[0,0].legend(fontsize=11)
ax[0,1].plot(t2, N_T2, color="orange", label="2")
ax[0,1].plot(
    fit_x,
    expo(fit_x, *po2),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T2), expo(np.array(t2), *po2)), 3)}\n"
        f"T = ({np.round(po2[0], 2)} ± {np.round(np.sqrt(np.diag(pc2))[0], 2)})"
        f"(1 - exp(-({np.round(po2[1], 2)} ± {np.round(np.sqrt(np.diag(pc2))[1], 2)})·t))\n"
        f"τ = {np.round(tau2, 2)} ± {np.round(tau_err2, 2)} (min)\n"
        f"$D=({np.round(D2*1e6, 4)}±{np.round(D2_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="orange",
    linestyle="dashed",
    zorder=-1)
ax[0,1].set_xlim([0,50])
ax[0,1].set_ylim([0,1.1])
ax[0,1].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[0,1].set_ylabel(r"Transmittance $Tr$",fontsize=12)
# ax[0,1].set_title("2")
ax[0,1].tick_params(axis='both', which='major', labelsize=12)
ax[0,1].legend(fontsize=11)
ax[1,0].plot(t3, N_T3, color="yellow", label="3")
ax[1,0].plot(
    fit_x,
    expo(fit_x, *po3),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T3), expo(np.array(t3), *po3)), 3)}\n"
        f"T = ({np.round(po3[0], 2)} ± {np.round(np.sqrt(np.diag(pc3))[0], 2)})"
        f"(1 - exp(-({np.round(po3[1], 2)} ± {np.round(np.sqrt(np.diag(pc3))[1], 2)})·t))\n"
        f"τ = {np.round(tau3, 2)} ± {np.round(tau_err3, 2)} (min)\n"
        f"$D=({np.round(D3*1e6, 4)}±{np.round(D3_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="yellow",
    linestyle="dashed",
    zorder=-1)
ax[1,0].set_xlim([0,50])
ax[1,0].set_ylim([0,1.1])
ax[1,0].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[1,0].set_ylabel(r"Transmittance $Tr$",fontsize=12)
# ax[1,0].set_title("3")
ax[1,0].tick_params(axis='both', which='major', labelsize=12)
ax[1,0].legend(fontsize=11)
ax[1,1].plot(t4, N_T4, color="green", label="4")
ax[1,1].plot(
    fit_x,
    expo(fit_x, *po4),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T4), expo(np.array(t4), *po4)), 3)}\n"
        f"T = ({np.round(po4[0], 2)} ± {np.round(np.sqrt(np.diag(pc4))[0], 2)})"
        f"(1 - exp(-({np.round(po4[1], 2)} ± {np.round(np.sqrt(np.diag(pc4))[1], 2)})·t))\n"
        f"τ = {np.round(tau4, 2)} ± {np.round(tau_err4, 2)} (min)\n"
        f"$D=({np.round(D4*1e6, 4)}±{np.round(D4_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="green",
    linestyle="dashed",
    zorder=-1)
ax[1,1].set_xlim([0,50])
ax[1,1].set_ylim([0,1.1])
ax[1,1].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[1,1].set_ylabel(r"Transmittance $Tr$",fontsize=12)
ax[1,1].set_title("4")
ax[1,1].tick_params(axis='both', which='major', labelsize=12)
ax[1,1].legend(fontsize=11)
ax[2,0].tick_params(axis='both', which='major', labelsize=12)
ax[2,0].legend(fontsize=11)
ax[2,0].plot(t5, N_T5, color="blue", label="5")
ax[2,0].plot(
    fit_x,
    expo(fit_x, *po5),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T5), expo(np.array(t5), *po5)), 3)}\n"
        f"T = ({np.round(po5[0], 2)} ± {np.round(np.sqrt(np.diag(pc5))[0], 2)})"
        f"(1 - exp(-({np.round(po5[1], 2)} ± {np.round(np.sqrt(np.diag(pc5))[1], 2)})·t))\n"
        f"τ = {np.round(tau5, 2)} ± {np.round(tau_err5, 2)} (min)\n"
        f"$D=({np.round(D5*1e6, 4)}±{np.round(D5_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="blue",
    linestyle="dashed",
    zorder=-1)
ax[2,0].set_xlim([0,50])
ax[2,0].set_ylim([0,1.1])
ax[2,0].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[2,0].set_ylabel(r"Transmittance $Tr$",fontsize=12)
ax[2,0].set_title("5")
ax[2,0].tick_params(axis='both', which='major', labelsize=12)
ax[2,0].legend(fontsize=11)
# ax[2,0].set_title("5")
ax[2,1].tick_params(axis='both', which='major', labelsize=12)
ax[2,1].legend(fontsize=11)
ax[2,1].plot(t6, N_T6, color="purple", label="6")
ax[2,1].plot(
    fit_x,
    expo(fit_x, *po6),
    label=(
        f"Exponential fit: R² = {np.round(r_squared(np.array(N_T6), expo(np.array(t6), *po6)), 3)}\n"
        f"T = ({np.round(po6[0], 2)} ± {np.round(np.sqrt(np.diag(pc6))[0], 2)})"
        f"(1 - exp(-({np.round(po6[1], 2)} ± {np.round(np.sqrt(np.diag(pc6))[1], 2)})·t))\n"
        f"τ = {np.round(tau6, 2)} ± {np.round(tau_err6, 2)} (min)\n"
        f"$D=({np.round(D6*1e6, 4)}±{np.round(D6_err*1e6, 4)})\\times 10^{{-6}}$ (cm$^2$/s)"
    ),
    color="purple",
    linestyle="dashed",
    zorder=-1)
ax[2,1].set_xlim([0,50])
ax[2,1].set_ylim([0,1.1])
ax[2,1].set_xlabel(r"Process time $t$ (minutes)",fontsize=12)
ax[2,1].set_ylabel(r"Transmittance $Tr$",fontsize=12)
ax[2,1].set_title("6")
ax[2,1].tick_params(axis='both', which='major', labelsize=12)
ax[2,1].legend(fontsize=11)

plt.show()


fig, ax = plt.subplots(figsize=(6, 4))
fit_x = np.linspace(0, 120, 9999)

# List of all datasets and their parameters
datasets = [
    (t1, N_T1, po1, pc1, tau1, tau_err1, D1, D1_err, "1", "red"),
    (t2, N_T2, po2, pc2, tau2, tau_err2, D2, D2_err, "2", "orange"),
    (t3, N_T3, po3, pc3, tau3, tau_err3, D3, D3_err, "3", "yellow"),
    (t4, N_T4, po4, pc4, tau4, tau_err4, D4, D4_err, "4", "green"),
    (t5, N_T5, po5, pc5, tau5, tau_err5, D5, D5_err, "5", "blue"),
    (t6, N_T6, po6, pc6, tau6, tau_err6, D6, D6_err, "6", "purple"),

]
# Fit the curve and plot collective graph
for t, N_T, popt, pcov, tau, tau_err, D, D_err, label_num, color in datasets:
    R2 = r_squared(np.array(N_T), expo(np.array(t), *popt))
    T0, k = popt
    T0_err, k_err = np.sqrt(np.diag(pcov))

    ax.plot(t, N_T, color=color)
    ax.plot(
        fit_x, expo(fit_x, *popt),
        linestyle='--',
        color=color,
        label=(
            f"Measurement {label_num}: R² = {R2:.3f}"
            # f"τ = {tau:.2f} ± {tau_err:.2f} min\n"
            # f"D = ({D*1e6:.4f} ± {D_err*1e6:.4f})×10^-6 cm²/s"
        )
    )

ax.set_xlim([0, 50])
ax.set_ylim([0, 1.1])
ax.set_xlabel(r"Process time $t$ (minutes)", fontsize=12)
ax.set_ylabel(r"Normalized Transmittance $Tr$", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=9) 
plt.tight_layout()
plt.show()

D_avg = np.mean([D1, D2, D3, D4, D5, D6])
D_error = np.std([D1, D2, D3, D4, D5, D6])
tau_avg = np.mean([tau1, tau2, tau3, tau4, tau5, tau6])
tau_error = np.std([tau_err1, tau_err2, tau_err3, tau_err4, tau_err5, tau_err6])

print("diffusion constant:", D_avg, "diffusion constant error:", D_error, "tau average:", tau_avg, "tau error", tau_error)
