import numpy as np

energies1 = [
    -65.83039,
            -65.81811,
            -65.80582,
            -65.79353,
            -65.78124,
            -65.76895,
            -65.75666,
            -65.74437,
            -65.73209,
            -65.7198,
            -65.70751,
            -65.69522,
            -65.68293,
            -65.67064,
            -65.65835,
            -65.64606,
            -65.63378,
            -65.62149,
            -65.6092,
            -65.59691,
            -65.58462,
            -65.57233,
            -65.56004,
            -65.54776,
            -65.53547,
            -65.52318,
            -65.51089,
]

energies2 = [
    -24.61397,
            -24.6084,
            -24.60284,
            -24.59728,
            -24.59172,
            -24.58615,
            -24.58059,
            -24.57503,
            -24.56947,
            -24.5639,
            -24.55834,
            -24.55278,
            -24.54722,
            -24.54165,
            -24.53609,
            -24.53053,
            -24.52497,
            -24.5194,
            -24.51384,
            -24.50828,
            -24.50272,
            -24.49715,
            -24.49159,
            -24.48603,
            -24.48047,
            -24.4749,
            -24.46934,
            -24.46378,
            -24.45822,
            -24.45265,
]

materials = [energies1, energies2]

for i in range(len(materials)):
    energies = materials[i]

    diffs = []
    for j in range(1, len(energies)):
        diff = energies[j] - energies[j - 1]
        diffs.append(diff)

    diffs_array = np.array(diffs)

    mean_spacing = diffs_array.mean()
    std_spacing = diffs_array.std()

    print("Mean spacing:", mean_spacing)
    print("All spacings:", diffs_array.tolist())
