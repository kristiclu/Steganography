import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from code2 import (
    load_grayscale_image, partition_host_image,
    permute_watermark, depermute_watermark,
    embed_watermark_in_ftl, embed_watermark_in_fbr_U,
    combine_subimages, extract_watermark_from_ftlw, extract_bits_from_fbrw_U,
    psnr, ber, normalized_correlation,
    attack_jpeg, attack_gaussian_noise, attack_blur, attack_resize
)

# ---------------------------------------------------------
# Lista napada koje koristimo za prosjek robusnosti
# ---------------------------------------------------------

ATTACKS = [
    ("JPEG_Q70",     lambda img: attack_jpeg(img, quality=70)),
    ("Gauss_sigma10", lambda img: attack_gaussian_noise(img, sigma=10)),
    ("Blur_r1",      lambda img: attack_blur(img, radius=1.0)),
    ("Resize_0.5",   lambda img: attack_resize(img, scale=0.5)),
]

# ---------------------------------------------------------
# Embedding + (višestruki) napadi + extraction + metrike
# ---------------------------------------------------------

def embed_and_evaluate_multi(host_path, wm_path, secret_key,
                             block_size, T, alpha,
                             attacks):
    """
    1) Učita host i watermark
    2) Embedda watermark u D (ftl) i U (fbr)
    3) Izračuna PSNR(host, watermarked) bez napada
    4) Za SVAKI attack_fn u 'attacks' primijeni napad na watermarked,
       extrakta watermark, računa BER i NC
    5) Vraća PSNR + PROSJEČNE BER/NC vrijednosti preko svih napada
    """
    # ---- LOAD IMAGES ----
    host = load_grayscale_image(host_path)
    wm_original = load_grayscale_image(wm_path)

    # ---- PARTITION HOST ----
    ftl, ftr, fbl, fbr = partition_host_image(host)

    # ---- RESIZE WATERMARK TO MATCH BLOCK CAPACITY ----
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size

    wm_resized_img = Image.fromarray(wm_original.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.NEAREST
    )
    wm_resized = np.array(wm_resized_img)

    # ground-truth bitovi watermarka (0/1)
    wm_true_bits = (wm_resized // 255).astype(np.uint8)

    # ---- PERMUTE WATERMARK ----
    wm_perm = permute_watermark(wm_resized, secret_key)

    # ---- EMBEDDING ----
    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size, alpha=alpha)

    watermarked = combine_subimages(ftl_w, ftr, fbl, fbr_w)

    # PSNR bez napada (kao u paperu)
    PSNR_no_attack = psnr(host, watermarked)

    # ---- LOOP PREKO NAPADA ----
    ber_D_list, ber_U_list = [], []
    nc_D_list, nc_U_list = [], []

    for name, attack_fn in attacks:
        attacked = attack_fn(watermarked)

        # EXTRACTION
        ftlw_ex, _, _, fbrw_ex = partition_host_image(attacked)

        wm_bits_D, _ = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)
        wm_bits_U, _ = extract_bits_from_fbrw_U(fbrw_ex, block_size)

        # de-permute extracted bits
        wm_bits_D_dep = depermute_watermark(wm_bits_D, secret_key)
        wm_bits_U_dep = depermute_watermark(wm_bits_U, secret_key)

        # METRIKE za ovaj napad
        ber_D_list.append(ber(wm_true_bits, wm_bits_D_dep))
        ber_U_list.append(ber(wm_true_bits, wm_bits_U_dep))
        nc_D_list.append(normalized_correlation(wm_true_bits, wm_bits_D_dep))
        nc_U_list.append(normalized_correlation(wm_true_bits, wm_bits_U_dep))

    # prosjek preko svih napada
    BER_D = float(np.mean(ber_D_list))
    BER_U = float(np.mean(ber_U_list))
    NC_D  = float(np.mean(nc_D_list))
    NC_U  = float(np.mean(nc_U_list))

    return PSNR_no_attack, BER_D, BER_U, NC_D, NC_U

# ---------------------------------------------------------
# Sweep po T (D grana)
# ---------------------------------------------------------

def sweep_T(host_path, wm_path, secret_key, block_size, alpha_fixed, T_values):
    rows = []

    for T in T_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
            host_path, wm_path, secret_key,
            block_size, T, alpha_fixed,
            attacks=ATTACKS
        )

        rows.append({
            "T": T,
            "alpha": alpha_fixed,
            "PSNR": PSNR_no_attack,
            "BER_D": BER_D,
            "BER_U": BER_U,
            "NC_D": NC_D,
            "NC_U": NC_U,
        })

    df_T = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df_T[numeric_cols] = df_T[numeric_cols].astype(float).round(3)

    df_T.to_csv(
        "T_sweep_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # --- GRAF 1: T vs PSNR (no attack) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["PSNR"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("PSNR (dB)")
    plt.title("Step size T vs PSNR (no attack)")
    plt.grid(True)
    plt.savefig("T_vs_PSNR.png", dpi=300, bbox_inches="tight")

    # --- GRAF 2: T vs NC_D (avg preko 4 napada) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["NC_D"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("NC_D (avg over attacks)")
    plt.title("Step size T vs NC_D (avg over JPEG, noise, blur, resize)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("T_vs_NC_D_avg_attacks.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO T ZAVRSEN =====")
    print(df_T.to_string(index=False))

    return df_T

# ---------------------------------------------------------
# Sweep po alpha (U grana)
# ---------------------------------------------------------

def sweep_alpha(host_path, wm_path, secret_key, block_size, T_fixed, alpha_values):
    rows = []

    for alpha in alpha_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
            host_path, wm_path, secret_key,
            block_size, T_fixed, alpha,
            attacks=ATTACKS
        )

        rows.append({
            "alpha": alpha,
            "T": T_fixed,
            "PSNR": PSNR_no_attack,
            "BER_D": BER_D,
            "BER_U": BER_U,
            "NC_D": NC_D,
            "NC_U": NC_U,
        })

    df_A = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df_A[numeric_cols] = df_A[numeric_cols].astype(float).round(3)

    df_A.to_csv(
        "alpha_sweep_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # --- GRAF 3: alpha vs PSNR ---
    plt.figure()
    plt.plot(df_A["alpha"], df_A["PSNR"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("PSNR (dB)")
    plt.title("alpha vs PSNR (no attack)")
    plt.grid(True)
    plt.savefig("alpha_vs_PSNR.png", dpi=300, bbox_inches="tight")

    # --- GRAF 4: alpha vs NC_U (avg preko napada) ---
    plt.figure()
    plt.plot(df_A["alpha"], df_A["NC_U"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("NC_U (avg over attacks)")
    plt.title("alpha vs NC_U (avg over JPEG, noise, blur, resize)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("alpha_vs_NC_U_avg_attacks.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO ALPHA ZAVRSEN =====")
    print(df_A.to_string(index=False))

    return df_A

# ---------------------------------------------------------
# Zajednički sweep po T i alpha (2D grid)
# ---------------------------------------------------------

def sweep_T_alpha_joint(host_path, wm_path, secret_key,
                        T_values, alpha_values,
                        block_size, PSNR_min):
    rows = []

    for T in T_values:
        for alpha in alpha_values:
            PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
                host_path, wm_path, secret_key,
                block_size, T, alpha,
                attacks=ATTACKS
            )

            rows.append({
                "T": T,
                "alpha": alpha,
                "PSNR": PSNR_no_attack,
                "BER_D": BER_D,
                "BER_U": BER_U,
                "NC_D": NC_D,
                "NC_U": NC_U,
            })

    df = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df[numeric_cols] = df[numeric_cols].astype(float).round(3)

    df["NC_sum"] = (df["NC_D"] + df["NC_U"]).round(3)

    df.to_csv(
        "T_alpha_grid_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    df_ok = df[df["PSNR"] >= PSNR_min].copy()
    if df_ok.empty:
        print(f"\nUPOZORENJE: niti jedan (T, alpha) nema PSNR >= {PSNR_min} dB.")
        print("Biramo najbolji par prema NC_sum bez PSNR praga.\n")
        df_ok = df.copy()

    best_row = df_ok.sort_values("NC_sum", ascending=False).iloc[0]
    best_T = float(best_row["T"])
    best_alpha = float(best_row["alpha"])

    print("\n===== ZAJEDNICKI SWEEP PO T I ALPHA =====")
    print(df.to_string(index=False))
    print("\nFiltrirano (PSNR >= {:.1f} dB):".format(PSNR_min))
    print(df_ok.to_string(index=False))
    print("\n>> Preporuceni parametri (na temelju PSNR praga i NC_sum, avg over 4 attacks):")
    print(f"   T*     = {best_T}")
    print(f"   alpha* = {best_alpha}")

    return df, best_T, best_alpha


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    host_path = "picture.png"
    wm_path   = "ivan.png"
    secret_key = "my_secret_key_123"
    block_size = 8

    # 1) Sweep po T (uz fiksni alpha)
    df_T = sweep_T(
        host_path, wm_path, secret_key,
        block_size=block_size,
        alpha_fixed=0.05,               # možeš promijeniti
        T_values=list(range(10, 101, 10))
    )

    # 2) Sweep po alpha (uz fiksni T)
    df_A = sweep_alpha(
        host_path, wm_path, secret_key,
        block_size=block_size,
        T_fixed=40,                     # ili kasnije T_star
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )

    # 3) Zajednička optimizacija T, alpha (2D grid, prosjek preko 4 napada)
    df_grid, T_star, alpha_star = sweep_T_alpha_joint(
        host_path, wm_path, secret_key,
        T_values=list(range(10, 101, 10)),
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        block_size=block_size,
        PSNR_min=40.0
    )

"""
“Za odabir parametara T i alpha proveli smo 2D grid pretragu. Robusnost smo definirali kao prosječnu 
vrijednost NC_D + NC_U preko četiri tipična napada (JPEG-kompresija, Gaussov šum, zamućenje i
promjena dimenzija), uz uvjet da PSNR ne padne ispod 40 dB.

Maksimalnu robusnost (NC_D + NC_U ≈ 1.549) postiže par (T=80, alpha=0.05), 
no uz relativno niži PSNR (≈40.6).Par (T=40, alpha=0.05) postiže malo 
manji zbroj NC (≈1.50), ali uz značajno bolju kvalitetu slike (PSNR ≈ 43.6 dB).
Zbog toga smo ga odabrali kao konačan kompromisni izbor parametara u nastavku eksperimenata.””
"""

"""
==== SWEEP PO T ZAVRSEN =====
  T  alpha   PSNR  BER_D  BER_U  NC_D  NC_U
 10   0.05 45.728  0.534  0.374 0.630 0.760
 20   0.05 45.153  0.505  0.369 0.649 0.763
 30   0.05 44.427  0.474  0.370 0.662 0.763
 40   0.05 43.672  0.444  0.367 0.696 0.764
 50   0.05 42.866  0.435  0.371 0.697 0.761
 60   0.05 41.897  0.489  0.365 0.651 0.765
 70   0.05 41.235  0.434  0.361 0.705 0.768
 80   0.05 40.658  0.371  0.363 0.752 0.767
 90   0.05 39.944  0.409  0.372 0.716 0.761
100   0.05 39.068  0.274  0.367 0.823 0.765

===== SWEEP PO ALPHA ZAVRSEN =====
 alpha  T   PSNR  BER_D  BER_U  NC_D  NC_U
  0.00 50 45.089  0.435  0.651 0.697 0.509
  0.05 50 42.866  0.421  0.365 0.713 0.766
  0.10 50 39.203  0.441  0.308 0.690 0.809
  0.15 50 36.307  0.443  0.277 0.686 0.832
  0.20 50 34.195  0.436  0.252 0.694 0.849
  0.25 50 32.589  0.437  0.233 0.695 0.864
  0.30 50 31.318  0.439  0.222 0.692 0.872

[Done] exited with code=1 in 21.462 seconds

[Running] python -u "c:\Users\lukasmich\Documents\LUKAS\Skola\Faks\diplomski\5_godina\mitmap\projekt_1\experiments_avg.py"

===== SWEEP PO T ZAVRSEN =====
  T  alpha   PSNR  BER_D  BER_U  NC_D  NC_U
 10   0.05 45.618  0.505  0.389 0.656 0.748
 20   0.05 45.075  0.423  0.383 0.720 0.753
 30   0.05 44.347  0.453  0.384 0.689 0.752
 40   0.05 43.575  0.385  0.380 0.741 0.755
 50   0.05 42.833  0.417  0.377 0.708 0.757
 60   0.05 41.830  0.488  0.379 0.647 0.755
 70   0.05 41.199  0.387  0.382 0.740 0.754
 80   0.05 40.602  0.335  0.384 0.770 0.752
 90   0.05 39.853  0.246  0.382 0.845 0.753
100   0.05 39.052  0.257  0.386 0.838 0.751

===== SWEEP PO ALPHA ZAVRSEN =====
 alpha  T   PSNR  BER_D  BER_U  NC_D  NC_U
  0.00 40 46.474  0.404  0.656 0.724 0.504
  0.05 40 43.575  0.391  0.378 0.736 0.756
  0.10 40 39.468  0.399  0.309 0.729 0.808
  0.15 40 36.430  0.381  0.276 0.744 0.832
  0.20 40 34.275  0.394  0.262 0.734 0.844
  0.25 40 32.642  0.402  0.238 0.725 0.861
  0.30 40 31.355  0.384  0.229 0.743 0.867

===== ZAJEDNICKI SWEEP PO T I ALPHA =====
  T  alpha   PSNR  BER_D  BER_U  NC_D  NC_U  NC_sum
 10   0.00 52.189  0.504  0.665 0.657 0.496   1.153
 10   0.05 45.618  0.514  0.381 0.649 0.754   1.403
 10   0.10 40.153  0.508  0.312 0.654 0.806   1.460
 10   0.15 36.756  0.504  0.278 0.656 0.831   1.487
 10   0.20 34.471  0.506  0.258 0.655 0.846   1.501
 10   0.25 32.776  0.511  0.236 0.651 0.862   1.513
 10   0.30 31.454  0.512  0.229 0.650 0.867   1.517
 20   0.00 50.133  0.426  0.660 0.718 0.502   1.220
 20   0.05 45.075  0.421  0.388 0.722 0.749   1.471
 20   0.10 39.992  0.426  0.309 0.717 0.808   1.525
 20   0.15 36.682  0.435  0.275 0.710 0.833   1.543
 20   0.20 34.426  0.421  0.262 0.722 0.843   1.565
 20   0.25 32.746  0.416  0.238 0.726 0.861   1.587
 20   0.30 31.432  0.423  0.226 0.719 0.869   1.588
 30   0.00 48.133  0.438  0.661 0.702 0.499   1.201
 30   0.05 44.347  0.453  0.382 0.690 0.753   1.443
 30   0.10 39.752  0.462  0.312 0.682 0.806   1.488
 30   0.15 36.568  0.464  0.278 0.681 0.831   1.512
 30   0.20 34.359  0.430  0.259 0.709 0.846   1.555
 30   0.25 32.700  0.442  0.237 0.699 0.861   1.560
 30   0.30 31.398  0.439  0.225 0.702 0.870   1.572
 40   0.00 46.474  0.408  0.660 0.720 0.501   1.221
 40   0.05 43.575  0.377  0.387 0.749 0.750   1.499
 40   0.10 39.468  0.401  0.308 0.727 0.809   1.536
 40   0.15 36.430  0.391  0.276 0.737 0.832   1.569
 40   0.20 34.275  0.397  0.263 0.731 0.842   1.573
 40   0.25 32.642  0.405  0.236 0.723 0.862   1.585
 40   0.30 31.355  0.393  0.228 0.734 0.868   1.602
 50   0.00 45.129  0.416  0.656 0.709 0.505   1.214
 50   0.05 42.833  0.411  0.386 0.714 0.750   1.464
 50   0.10 39.165  0.419  0.307 0.705 0.809   1.514
 50   0.15 36.276  0.424  0.278 0.699 0.831   1.530
 50   0.20 34.181  0.403  0.254 0.723 0.848   1.571
 50   0.25 32.577  0.416  0.236 0.709 0.862   1.571
 50   0.30 31.307  0.414  0.227 0.711 0.868   1.579
 60   0.00 43.543  0.455  0.656 0.689 0.505   1.194
 60   0.05 41.830  0.416  0.380 0.726 0.755   1.481
 60   0.10 38.705  0.467  0.311 0.674 0.807   1.481
 60   0.15 36.034  0.366  0.279 0.765 0.830   1.595
 60   0.20 34.030  0.448  0.258 0.696 0.846   1.542
 60   0.25 32.472  0.477  0.237 0.663 0.862   1.525
 60   0.30 31.228  0.425  0.229 0.717 0.867   1.584
 70   0.00 42.636  0.393  0.656 0.733 0.505   1.238
 70   0.05 41.199  0.411  0.388 0.713 0.749   1.462
 70   0.10 38.387  0.421  0.309 0.702 0.808   1.510
 70   0.15 35.859  0.414  0.286 0.712 0.826   1.538
 70   0.20 33.918  0.412  0.263 0.713 0.843   1.556
 70   0.25 32.394  0.385  0.238 0.742 0.861   1.603
 70   0.30 31.170  0.396  0.226 0.731 0.869   1.600
 80   0.00 41.826  0.367  0.657 0.733 0.504   1.237
 80   0.05 40.602  0.309  0.383 0.796 0.753   1.549
 80   0.10 38.064  0.317  0.306 0.787 0.810   1.597
 80   0.15 35.676  0.349  0.279 0.754 0.830   1.584
 80   0.20 33.800  0.356  0.261 0.746 0.844   1.590
 80   0.25 32.311  0.309  0.235 0.795 0.862   1.657
 80   0.30 31.107  0.297  0.226 0.806 0.869   1.675
 90   0.00 40.858  0.318  0.653 0.785 0.508   1.293
 90   0.05 39.853  0.322  0.383 0.781 0.753   1.534
 90   0.10 37.630  0.300  0.307 0.802 0.809   1.611
 90   0.15 35.420  0.354  0.282 0.744 0.829   1.573
 90   0.20 33.633  0.330  0.259 0.772 0.845   1.617
 90   0.25 32.191  0.338  0.233 0.763 0.864   1.627
 90   0.30 31.016  0.317  0.227 0.785 0.869   1.654
100   0.00 39.871  0.280  0.660 0.819 0.501   1.320
100   0.05 39.052  0.282  0.381 0.818 0.754   1.572
100   0.10 37.133  0.280  0.319 0.819 0.802   1.621
100   0.15 35.114  0.268  0.281 0.829 0.829   1.658
100   0.20 33.428  0.243  0.259 0.848 0.846   1.694
100   0.25 32.043  0.295  0.238 0.807 0.861   1.668
100   0.30 30.902  0.266  0.226 0.831 0.869   1.700

Filtrirano (PSNR >= 40.0 dB):
 T  alpha   PSNR  BER_D  BER_U  NC_D  NC_U  NC_sum
10   0.00 52.189  0.504  0.665 0.657 0.496   1.153
10   0.05 45.618  0.514  0.381 0.649 0.754   1.403
10   0.10 40.153  0.508  0.312 0.654 0.806   1.460
20   0.00 50.133  0.426  0.660 0.718 0.502   1.220
20   0.05 45.075  0.421  0.388 0.722 0.749   1.471
30   0.00 48.133  0.438  0.661 0.702 0.499   1.201
30   0.05 44.347  0.453  0.382 0.690 0.753   1.443
40   0.00 46.474  0.408  0.660 0.720 0.501   1.221
40   0.05 43.575  0.377  0.387 0.749 0.750   1.499
50   0.00 45.129  0.416  0.656 0.709 0.505   1.214
50   0.05 42.833  0.411  0.386 0.714 0.750   1.464
60   0.00 43.543  0.455  0.656 0.689 0.505   1.194
60   0.05 41.830  0.416  0.380 0.726 0.755   1.481
70   0.00 42.636  0.393  0.656 0.733 0.505   1.238
70   0.05 41.199  0.411  0.388 0.713 0.749   1.462
80   0.00 41.826  0.367  0.657 0.733 0.504   1.237
80   0.05 40.602  0.309  0.383 0.796 0.753   1.549
90   0.00 40.858  0.318  0.653 0.785 0.508   1.293

>> Preporuceni parametri (na temelju PSNR praga i NC_sum, avg over 4 attacks):
   T*     = 80.0
   alpha* = 0.05
"""

"""
GRAFOVI
T vs PSNR
Oblik je skoro linearan: svako povećanje T za 10 sruši PSNR za otprilike 0.6-0.8 dB.
Ovo je i paper tvrdio - veći T jače kvantizira najveće singularne vrijednosti, 
pa je watermark više “utisnut” ali host slika izgleda lošije.
T=40 očito leži još u “visokokvalitetnom” području: ~43.5 dB je jako dobar PSNR.
Između T=40 i T=70 samo ~2.5 dB pada, pa ako želimo dodatnu robusnost možemo uzeti veci T

T vs NC_D (avg over JPEG, noise, blur, resize)
Ima malo “zupčast” oblik (npr. mali pad oko T=30, T=60), što je normalno jer uzimamo prosijek
preko različitih napada, ali globalno trend je jasan - raste s T.
D-grana je sve robusnija kako povećavamo T  što je očekivano, za T = 80-100 PSNR pada ispod 41db

alpha vs PSNR
Krivulja je dosta strmo opadajuća. Ovdje je trade-off veci nego za T: alpha kontrolira U-granu,
ali cijena u PSNR-u je vrlo visoka. alpha između 0.05 i 0.10 je nekakav “razuman” raspon; iznad toga PSNR pada 
u 30-ak dB i slika će već biti vidljivo lošija.

alpha vs NC_U (avg over attacks)
veliki skok robusnosti između alpha=0 i alpha=0.05,
nakon toga dobili smo “diminishing returns”: svaki daljnji korak u alpha donosi malo poboljšanje NC_U,
ali PSNR vrijednost u isto vrijeme dosta pada.
"""