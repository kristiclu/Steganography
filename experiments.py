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
    attack_jpeg      # koristimo JPEG q=30 kao reprezentativan napad
)

# ---------------------------------------------------------
# Pomoćna funkcija: embed + (opcionalno) napad + extraction + metrike
# ---------------------------------------------------------

def embed_and_evaluate(host_path, wm_path, secret_key, block_size, T, alpha, attack_fn=None):
    """
    1) učita host i watermark
    2) embedda watermark u D (ftl) i U (fbr)
    3) izračuna PSNR(host, watermarked) bez napada
    4) (opcionalno) primijeni attack_fn na watermarked
    5) extrakta watermark iz D i U, depermutira
    6) vrati PSNR + BER + NC za obje grane
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

    # ---- ATTACK (ako je zadan) ----
    if attack_fn is not None:
        attacked = attack_fn(watermarked)
    else:
        attacked = watermarked

    # ---- EXTRACTION ----
    ftlw_ex, _, _, fbrw_ex = partition_host_image(attacked)

    wm_bits_D, _ = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)
    wm_bits_U, _ = extract_bits_from_fbrw_U(fbrw_ex, block_size)

    # de-permute extracted bits
    wm_bits_D_dep = depermute_watermark(wm_bits_D, secret_key)
    wm_bits_U_dep = depermute_watermark(wm_bits_U, secret_key)

    # ---- METRIKE ----
    BER_D = ber(wm_true_bits, wm_bits_D_dep)
    BER_U = ber(wm_true_bits, wm_bits_U_dep)

    NC_D = normalized_correlation(wm_true_bits, wm_bits_D_dep)
    NC_U = normalized_correlation(wm_true_bits, wm_bits_U_dep)

    return PSNR_no_attack, BER_D, BER_U, NC_D, NC_U

# ---------------------------------------------------------
# Sweep po T (D grana) – reproducira Fig.6 + graf robustnosti
# ---------------------------------------------------------

def sweep_T(host_path, wm_path, secret_key, block_size, alpha_fixed, T_values):
    rows = []

    for T in T_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate(
            host_path, wm_path, secret_key,
            block_size, T, alpha_fixed,
            attack_fn=lambda img: attack_jpeg(img, quality=30)  # JPEG q=30
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

    # spremi CSV (za Excel)
    df_T.to_csv(
        "T_sweep_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # --- GRAF 1: T vs PSNR (bez napada) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["PSNR"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("PSNR (dB)")
    plt.title("Step size T vs PSNR (no attack)")
    plt.grid(True)
    plt.savefig("T_vs_PSNR.png", dpi=300, bbox_inches="tight")

    # --- GRAF 2: T vs NC_D (robustnost D grane pod JPEG q=30) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["NC_D"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("NC (D branch, JPEG q=30)")
    plt.title("Step size T vs NC_D (JPEG attack)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("T_vs_NC_D_JPEG30.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO T ZAVRSEN =====")
    print(df_T.to_string(index=False))

    return df_T

# ---------------------------------------------------------
# Sweep po alpha (U grana) – PSNR + robustnost NC_U
# ---------------------------------------------------------

def sweep_alpha(host_path, wm_path, secret_key, block_size, T_fixed, alpha_values):
    rows = []

    for alpha in alpha_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate(
            host_path, wm_path, secret_key,
            block_size, T_fixed, alpha,
            attack_fn=lambda img: attack_jpeg(img, quality=30)
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

    # --- GRAF 4: alpha vs NC_U (robustnost U grane) ---
    plt.figure()
    plt.plot(df_A["alpha"], df_A["NC_U"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("NC (U branch, JPEG q=30)")
    plt.title("alpha vs NC_U (JPEG attack)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("alpha_vs_NC_U_JPEG30.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO ALPHA ZAVRSEN =====")
    print(df_A.to_string(index=False))

    return df_A

def sweep_T_alpha_joint(host_path, wm_path, secret_key, T_values, alpha_values, block_size, PSNR_min, jpeg_quality):
    """
    Radi 2D pretragu po (T, alpha). Za svaki par:
      - embed + extraction
      - PSNR (bez napada)
      - NC_D i NC_U nakon JPEG napada (quality=jpeg_quality)
    Zatim unutar svih parova s PSNR >= PSNR_min
    bira onaj s najvećim (NC_D + NC_U) kao "najbolji kompromis".
    """

    rows = []

    for T in T_values:
        for alpha in alpha_values:
            PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate(
                host_path, wm_path, secret_key,
                block_size, T, alpha,
                attack_fn=lambda img: attack_jpeg(img, quality=jpeg_quality)
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

    # dodatna metrika: zbroj NC_D + NC_U
    df["NC_sum"] = (df["NC_D"] + df["NC_U"]).round(3)

    # spremi CSV za Excel
    df.to_csv(
        "T_alpha_grid_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # filtriraj parove koji zadovoljavaju minimalni PSNR
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
    print("\n>> Preporuceni parametri (na temelju PSNR praga i NC_sum):")
    print(f"   T*     = {best_T}")
    print(f"   alpha* = {best_alpha}")

    return df, best_T, best_alpha

# ---------------------------------------------------------
# MAIN – pokretanje oba sweepa
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
        alpha_fixed=0.05,
        T_values=list(range(10, 101, 10))
    )

    # 2) Sweep po alpha (uz fiksni T_opt)
    df_A = sweep_alpha(
        host_path, wm_path, secret_key,
        block_size=block_size,
        T_fixed=60,
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    
    df_grid, T_star, alpha_star = sweep_T_alpha_joint(
        host_path, wm_path, secret_key,
        T_values=list(range(10, 101, 10)),
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        block_size=block_size,
        PSNR_min=40.0,
        jpeg_quality=30
    )

"""
Za odabir parametara T i alpha koristili smo JPEG-kompresiju (QF=30) kao reprezentativan realističan 
napad, jer je JPEG najčešći format distribucije slika i standardni napad u watermarking literaturi.
T i ačpha su odabrani tako da postignu kompromis između PSNR-a i robusnosti na ovaj napad (NC, BER).”
"""