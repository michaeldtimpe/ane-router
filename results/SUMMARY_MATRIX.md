# Benchmark matrix

## [block] mixtral-8x7b  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx ⭐ | 3104.75 | 8494.22 | 3.034 | GPU |
| hybrid_router_ane | 3696.31 | 8373.95 | 10.038 | MIXED |
| hybrid_expert_ane | 7146.06 | 7339.75 | 10.547 | MIXED |
| hybrid_all_coreml | 6651.23 | 6836.57 | 6.680 | ANE |

## [block] mixtral-8x7b  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx | 7838.79 | 11277.53 | 9.944 | GPU |
| hybrid_router_ane ⭐ | 7818.04 | 11049.64 | 6.695 | MIXED |
| hybrid_expert_ane | 9342.79 | 12528.28 | 12.460 | MIXED |
| hybrid_all_coreml | 8612.52 | 8881.22 | 8.723 | ANE |

## [block] mixtral-8x7b  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx ⭐ | 7849.12 | 8740.82 | 7.218 | GPU |
| hybrid_router_ane | 8593.15 | 17265.85 | 10.513 | MIXED |
| hybrid_expert_ane | 10611.85 | 11285.11 | 12.941 | MIXED |
| hybrid_all_coreml | 10273.92 | 10884.76 | 13.381 | ANE |

## [block] olmoe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx | 689.56 | 1625.66 | 1.040 | GPU |
| hybrid_router_ane | 506.08 | 822.82 | 0.966 | MIXED |
| hybrid_expert_ane | 989.50 | 1331.26 | 1.958 | MIXED |
| hybrid_all_coreml ⭐ | 456.46 | 550.55 | 1.349 | ANE |

## [block] olmoe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx ⭐ | 543.69 | 961.36 | 0.684 | GPU |
| hybrid_router_ane | 1246.92 | 2790.52 | 3.236 | MIXED |
| hybrid_expert_ane | 1018.65 | 1382.36 | 2.037 | MIXED |
| hybrid_all_coreml | 701.29 | 752.03 | 1.869 | ANE |

## [block] olmoe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| hybrid_all_mlx ⭐ | 555.73 | 6015.33 | 0.701 | GPU |
| hybrid_router_ane | 1243.15 | 1456.23 | 2.363 | MIXED |
| hybrid_expert_ane | 2807.35 | 9774.36 | 8.415 | MIXED |
| hybrid_all_coreml | 1937.73 | 2208.29 | 4.070 | ANE |

## [expert] deepseek-moe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 781.08 | 805.21 | 0.792 | CPU |
| torch_cpu_fp16 ⭐ | 257.44 | 310.85 | 0.280 | CPU |
| mlx | 334.83 | 494.36 | 0.525 | CPU |
| coreml/ALL | 564.85 | 664.75 | 1.571 | ANE |
| coreml/CPU_AND_GPU | 695.79 | 864.32 | 0.827 | CPU |
| coreml/CPU_AND_NE | 573.65 | 694.11 | 1.375 | ANE |
| coreml/CPU_ONLY | 776.58 | 874.37 | 0.914 | CPU |

## [expert] deepseek-moe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2103.17 | 2402.26 | 2.006 | CPU |
| torch_cpu_fp16 ⭐ | 329.31 | 359.91 | 0.336 | CPU |
| mlx | 513.96 | 1094.68 | 0.844 | GPU |
| coreml/ALL | 592.27 | 721.88 | 1.641 | ANE |
| coreml/CPU_AND_GPU | 726.17 | 875.53 | 1.088 | GPU |
| coreml/CPU_AND_NE | 641.35 | 940.26 | 1.746 | ANE |
| coreml/CPU_ONLY | 904.79 | 1153.51 | 1.029 | CPU |

## [expert] deepseek-moe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2077.52 | 2201.43 | 2.133 | CPU |
| torch_cpu_fp16 ⭐ | 459.23 | 521.30 | 0.491 | CPU |
| mlx | 513.98 | 547.81 | 1.188 | GPU |
| coreml/ALL | 651.83 | 770.46 | 1.743 | ANE |
| coreml/CPU_AND_GPU | 1023.04 | 7555.42 | 1.561 | GPU |
| coreml/CPU_AND_NE | 646.06 | 760.97 | 1.551 | ANE |
| coreml/CPU_ONLY | 675.90 | 752.35 | 1.032 | CPU |

## [expert] deepseek-moe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1688.75 | 1858.28 | 1.663 | CPU |
| torch_cpu_fp16 ⭐ | 597.31 | 725.82 | 0.561 | CPU |
| mlx | 614.04 | 747.76 | 1.294 | GPU |
| coreml/ALL | 743.50 | 855.30 | 1.806 | ANE |
| coreml/CPU_AND_GPU | 1071.33 | 1329.75 | 1.448 | GPU |
| coreml/CPU_AND_NE | 741.71 | 878.67 | 1.733 | ANE |
| coreml/CPU_ONLY | 770.31 | 836.36 | 0.936 | CPU |

## [expert] deepseek-moe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1130.00 | 1217.82 | 1.174 | CPU |
| torch_cpu_fp16 | 1710.04 | 2725.34 | 1.700 | CPU |
| mlx ⭐ | 564.65 | 676.95 | 0.914 | CPU |
| coreml/ALL | 1963.00 | 2271.09 | 3.147 | ANE |
| coreml/CPU_AND_GPU | 1619.98 | 1849.69 | 1.641 | CPU |
| coreml/CPU_AND_NE | 1922.62 | 2202.61 | 2.953 | ANE |
| coreml/CPU_ONLY | 1608.48 | 1845.34 | 1.721 | CPU |

## [expert] deepseek-moe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2654.40 | 2766.83 | 2.658 | CPU |
| torch_cpu_fp16 | 6442.00 | 8871.01 | 7.118 | CPU |
| mlx ⭐ | 891.48 | 925.01 | 2.901 | GPU |
| coreml/ALL | 9963.65 | 12492.11 | 10.975 | ANE |
| coreml/CPU_AND_GPU | 18353.08 | 31177.89 | 15.279 | GPU |
| coreml/CPU_AND_NE | 9991.42 | 12301.21 | 10.090 | ANE |
| coreml/CPU_ONLY | 5787.12 | 6485.29 | 5.866 | CPU |

## [expert] deepseek-moe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 5456.44 | 5847.49 | 5.538 | CPU |
| torch_cpu_fp16 | 26171.77 | 33471.91 | 23.222 | CPU |
| mlx ⭐ | 2334.75 | 2457.38 | 3.816 | GPU |
| coreml/ALL | 38559.54 | 43583.80 | 44.161 | ANE |
| coreml/CPU_AND_GPU | 101902.92 | 258592.95 | 88.447 | GPU |
| coreml/CPU_AND_NE | 38469.38 | 43387.22 | 45.558 | ANE |
| coreml/CPU_ONLY | 26075.29 | 27457.88 | 28.257 | CPU |

## [expert] mixtral-8x7b  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 22946.50 | 23434.73 | 23.508 | CPU |
| torch_cpu_fp16 | 3260.60 | 3563.82 | 3.162 | CPU |
| mlx | 2631.81 | 2736.26 | 3.271 | GPU |
| coreml/ALL | 6588.27 | 6889.82 | 6.593 | ANE |
| coreml/CPU_AND_GPU ⭐ | 2542.15 | 2819.51 | 4.971 | GPU |
| coreml/CPU_AND_NE | 6584.98 | 6714.25 | 7.000 | ANE |
| coreml/CPU_ONLY | 7314.33 | 7553.43 | 7.441 | CPU |

## [expert] mixtral-8x7b  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 27498.85 | 30649.18 | 27.428 | CPU |
| torch_cpu_fp16 | 3575.88 | 4623.27 | 3.684 | CPU |
| mlx | 5279.56 | 5378.75 | 5.816 | GPU |
| coreml/ALL | 6657.88 | 6789.47 | 6.810 | ANE |
| coreml/CPU_AND_GPU ⭐ | 3312.25 | 3625.23 | 4.166 | GPU |
| coreml/CPU_AND_NE | 6648.94 | 6778.77 | 6.693 | ANE |
| coreml/CPU_ONLY | 9921.77 | 10358.23 | 9.985 | CPU |

## [expert] mixtral-8x7b  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 27024.33 | 30660.47 | 27.291 | CPU |
| torch_cpu_fp16 | 4972.04 | 6578.19 | 5.012 | CPU |
| mlx | 5452.88 | 5872.14 | 6.468 | GPU |
| coreml/ALL | 6979.71 | 7427.23 | 6.985 | ANE |
| coreml/CPU_AND_GPU | 4364.69 | 4981.68 | 5.662 | GPU |
| coreml/CPU_AND_NE | 6816.85 | 7126.14 | 6.936 | ANE |
| coreml/CPU_ONLY ⭐ | 3931.90 | 4468.64 | 3.999 | CPU |

## [expert] mixtral-8x7b  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 26983.65 | 30293.90 | 29.012 | CPU |
| torch_cpu_fp16 | 10783.17 | 14816.40 | 9.733 | CPU |
| mlx | 5296.87 | 5409.59 | 7.561 | GPU |
| coreml/ALL | 8478.04 | 8939.10 | 8.567 | ANE |
| coreml/CPU_AND_GPU | 7440.88 | 7795.34 | 7.565 | GPU |
| coreml/CPU_AND_NE | 8466.87 | 8797.50 | 8.596 | ANE |
| coreml/CPU_ONLY ⭐ | 4603.10 | 5307.69 | 4.560 | CPU |

## [expert] mixtral-8x7b  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 27245.98 | 30842.45 | 27.065 | CPU |
| torch_cpu_fp16 | 31661.38 | 40527.38 | 32.133 | CPU |
| mlx ⭐ | 5543.50 | 5731.89 | 7.117 | GPU |
| coreml/ALL | 9892.29 | 10890.19 | 11.392 | ANE |
| coreml/CPU_AND_GPU | 66794.42 | 68152.55 | 67.611 | GPU |
| coreml/CPU_AND_NE | 9605.77 | 10862.17 | 9.935 | ANE |
| coreml/CPU_ONLY | 6253.50 | 6898.65 | 6.068 | CPU |

## [expert] mixtral-8x7b  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 46112.62 | 50331.16 | 47.894 | CPU |
| torch_cpu_fp16 | 124088.77 | 147494.99 | 134.746 | CPU |
| mlx | 9590.94 | 9781.55 | 9.673 | GPU |
| coreml/ALL ⭐ | 0.00 | 0.00 | 0.000 | ANE |
| coreml/CPU_AND_GPU | 304535.54 | 327082.52 | 307.147 | GPU |
| coreml/CPU_AND_NE | 0.00 | 0.00 | 0.000 | ANE |
| coreml/CPU_ONLY | 20304.44 | 21387.86 | 20.738 | CPU |

## [expert] mixtral-8x7b  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 117853.46 | 119300.09 | 119.305 | CPU |
| torch_cpu_fp16 | 571284.69 | 595409.31 | 552.945 | CPU |
| mlx ⭐ | 35173.54 | 35878.47 | 35.102 | GPU |
| coreml/ALL | 269260.44 | 283683.50 | 271.255 | ANE |
| coreml/CPU_AND_GPU | 1312034.12 | 1373756.53 | 1224.381 | GPU |
| coreml/CPU_AND_NE | 284818.31 | 292400.21 | 270.846 | ANE |
| coreml/CPU_ONLY | 83144.08 | 85603.58 | 82.261 | CPU |

## [expert] olmoe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 559.25 | 589.74 | 0.597 | CPU |
| torch_cpu_fp16 ⭐ | 257.92 | 280.83 | 0.224 | CPU |
| mlx | 354.46 | 458.05 | 0.513 | CPU |
| coreml/ALL | 468.33 | 501.92 | 1.130 | ANE |
| coreml/CPU_AND_GPU | 462.88 | 508.53 | 0.573 | CPU |
| coreml/CPU_AND_NE | 470.96 | 504.52 | 1.172 | ANE |
| coreml/CPU_ONLY | 459.96 | 525.51 | 0.556 | CPU |

## [expert] olmoe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1460.40 | 1513.97 | 1.422 | CPU |
| torch_cpu_fp16 ⭐ | 257.62 | 279.95 | 0.285 | CPU |
| mlx | 425.27 | 462.23 | 0.915 | GPU |
| coreml/ALL | 500.58 | 562.78 | 1.197 | ANE |
| coreml/CPU_AND_GPU | 680.69 | 6380.33 | 0.931 | GPU |
| coreml/CPU_AND_NE | 498.94 | 533.70 | 1.253 | ANE |
| coreml/CPU_ONLY | 505.17 | 604.48 | 0.709 | CPU |

## [expert] olmoe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1496.42 | 1542.44 | 1.551 | CPU |
| torch_cpu_fp16 ⭐ | 354.54 | 404.37 | 0.362 | CPU |
| mlx | 731.71 | 867.34 | 0.972 | GPU |
| coreml/ALL | 524.92 | 558.84 | 1.274 | ANE |
| coreml/CPU_AND_GPU | 965.04 | 1122.59 | 1.288 | GPU |
| coreml/CPU_AND_NE | 547.62 | 597.17 | 0.941 | ANE |
| coreml/CPU_ONLY | 575.02 | 653.07 | 0.599 | CPU |

## [expert] olmoe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1252.06 | 1277.85 | 1.250 | CPU |
| torch_cpu_fp16 | 480.06 | 525.75 | 0.487 | CPU |
| mlx ⭐ | 453.87 | 565.62 | 0.794 | GPU |
| coreml/ALL | 594.42 | 611.89 | 1.351 | ANE |
| coreml/CPU_AND_GPU | 1487.50 | 2009.27 | 1.464 | GPU |
| coreml/CPU_AND_NE | 622.40 | 723.20 | 1.489 | ANE |
| coreml/CPU_ONLY | 608.02 | 716.20 | 0.696 | CPU |

## [expert] olmoe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1380.81 | 1463.69 | 1.353 | CPU |
| torch_cpu_fp16 | 1276.48 | 1836.55 | 1.297 | CPU |
| mlx ⭐ | 462.52 | 516.13 | 0.999 | CPU |
| coreml/ALL | 1731.48 | 2123.61 | 2.372 | ANE |
| coreml/CPU_AND_GPU | 1308.06 | 1411.65 | 1.590 | CPU |
| coreml/CPU_AND_NE | 1505.12 | 1992.50 | 1.836 | ANE |
| coreml/CPU_ONLY | 1316.77 | 1439.17 | 1.401 | CPU |

## [expert] olmoe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2053.40 | 2158.11 | 2.123 | CPU |
| torch_cpu_fp16 | 4711.58 | 6891.50 | 4.550 | CPU |
| mlx ⭐ | 702.38 | 763.40 | 1.033 | GPU |
| coreml/ALL | 7919.27 | 10722.38 | 9.706 | ANE |
| coreml/CPU_AND_GPU | 13464.46 | 20167.73 | 15.214 | GPU |
| coreml/CPU_AND_NE | 8388.50 | 11037.65 | 10.212 | ANE |
| coreml/CPU_ONLY | 5786.79 | 6390.87 | 5.828 | CPU |

## [expert] olmoe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 4618.17 | 4869.21 | 4.972 | CPU |
| torch_cpu_fp16 | 19075.90 | 27706.57 | 17.634 | CPU |
| mlx ⭐ | 1744.88 | 1836.03 | 3.276 | GPU |
| coreml/ALL | 37231.29 | 43103.46 | 41.255 | ANE |
| coreml/CPU_AND_GPU | 81313.58 | 112684.71 | 88.981 | GPU |
| coreml/CPU_AND_NE | 36619.75 | 40462.66 | 44.630 | ANE |
| coreml/CPU_ONLY | 25811.25 | 27482.95 | 24.338 | CPU |

## [expert] qwen2-moe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 813.38 | 918.92 | 0.798 | CPU |
| torch_cpu_fp16 ⭐ | 278.79 | 327.25 | 0.366 | CPU |
| mlx | 375.13 | 506.42 | 0.554 | CPU |
| coreml/ALL | 555.69 | 584.93 | 1.315 | ANE |
| coreml/CPU_AND_GPU | 683.44 | 911.56 | 0.680 | CPU |
| coreml/CPU_AND_NE | 554.29 | 611.34 | 1.371 | ANE |
| coreml/CPU_ONLY | 618.77 | 753.93 | 0.654 | CPU |

## [expert] qwen2-moe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2095.67 | 2429.28 | 2.988 | CPU |
| torch_cpu_fp16 ⭐ | 329.81 | 364.50 | 0.334 | CPU |
| mlx | 530.94 | 1099.26 | 1.230 | GPU |
| coreml/ALL | 579.87 | 603.34 | 1.766 | ANE |
| coreml/CPU_AND_GPU | 699.19 | 6910.08 | 0.898 | GPU |
| coreml/CPU_AND_NE | 585.33 | 615.14 | 1.481 | ANE |
| coreml/CPU_ONLY | 935.87 | 1009.09 | 0.995 | CPU |

## [expert] qwen2-moe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2094.98 | 2371.81 | 2.001 | CPU |
| torch_cpu_fp16 ⭐ | 408.92 | 457.53 | 0.416 | CPU |
| mlx | 520.92 | 678.30 | 1.189 | GPU |
| coreml/ALL | 635.04 | 665.54 | 1.676 | ANE |
| coreml/CPU_AND_GPU | 946.81 | 1070.24 | 1.221 | GPU |
| coreml/CPU_AND_NE | 634.29 | 664.31 | 1.340 | ANE |
| coreml/CPU_ONLY | 715.19 | 767.95 | 0.790 | CPU |

## [expert] qwen2-moe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1696.90 | 1886.94 | 1.685 | CPU |
| torch_cpu_fp16 | 597.08 | 746.68 | 0.606 | CPU |
| mlx ⭐ | 531.65 | 587.92 | 1.275 | GPU |
| coreml/ALL | 738.21 | 829.46 | 1.831 | ANE |
| coreml/CPU_AND_GPU | 967.27 | 7448.82 | 1.425 | GPU |
| coreml/CPU_AND_NE | 730.54 | 792.60 | 1.731 | ANE |
| coreml/CPU_ONLY | 826.94 | 907.26 | 0.822 | CPU |

## [expert] qwen2-moe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1153.85 | 1312.05 | 1.147 | CPU |
| torch_cpu_fp16 | 1965.67 | 3513.93 | 2.297 | CPU |
| mlx ⭐ | 561.88 | 621.21 | 0.910 | CPU |
| coreml/ALL | 1891.38 | 2214.60 | 2.708 | ANE |
| coreml/CPU_AND_GPU | 1529.40 | 1625.92 | 1.448 | CPU |
| coreml/CPU_AND_NE | 1942.94 | 2299.79 | 3.054 | ANE |
| coreml/CPU_ONLY | 1589.33 | 1728.28 | 1.722 | CPU |

## [expert] qwen2-moe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2744.92 | 2992.74 | 2.818 | CPU |
| torch_cpu_fp16 | 6564.42 | 8869.36 | 6.426 | CPU |
| mlx ⭐ | 1133.33 | 1358.44 | 2.488 | GPU |
| coreml/ALL | 10091.54 | 13015.80 | 10.588 | ANE |
| coreml/CPU_AND_GPU | 15680.92 | 16918.39 | 17.133 | GPU |
| coreml/CPU_AND_NE | 9949.44 | 12562.59 | 11.313 | ANE |
| coreml/CPU_ONLY | 6239.12 | 7112.20 | 6.153 | CPU |

## [expert] qwen2-moe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 5502.37 | 5723.49 | 5.691 | CPU |
| torch_cpu_fp16 | 27262.44 | 35067.11 | 25.223 | CPU |
| mlx ⭐ | 2304.50 | 2347.28 | 5.086 | GPU |
| coreml/ALL | 39597.27 | 43025.22 | 45.436 | ANE |
| coreml/CPU_AND_GPU | 79405.54 | 99828.96 | 102.919 | GPU |
| coreml/CPU_AND_NE | 40026.96 | 43127.97 | 45.783 | ANE |
| coreml/CPU_ONLY | 26304.67 | 27122.27 | 26.746 | CPU |

## [router_only] deepseek-moe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 11.08 | 11.54 | 0.016 | CPU |
| torch_cpu_fp16 | 45.10 | 58.79 | 0.080 | CPU |
| mlx | 261.15 | 301.30 | 0.276 | CPU |
| coreml/ALL | 65.12 | 68.38 | 0.094 | CPU |
| coreml/CPU_AND_GPU | 65.17 | 68.14 | 0.101 | CPU |
| coreml/CPU_AND_NE | 64.67 | 67.30 | 0.108 | CPU |
| coreml/CPU_ONLY | 67.33 | 81.31 | 0.111 | CPU |

## [router_only] deepseek-moe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 64.52 | 85.97 | 0.085 | CPU |
| torch_cpu_fp16 | 86.04 | 112.38 | 0.110 | CPU |
| mlx | 257.73 | 312.50 | 0.296 | CPU |
| coreml/ALL | 81.62 | 88.08 | 0.130 | CPU |
| coreml/CPU_AND_GPU | 81.62 | 88.84 | 0.130 | CPU |
| coreml/CPU_AND_NE | 84.88 | 103.76 | 0.129 | CPU |
| coreml/CPU_ONLY | 81.25 | 87.04 | 0.122 | CPU |

## [router_only] deepseek-moe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 64.08 | 84.04 | 0.158 | CPU |
| torch_cpu_fp16 | 88.00 | 121.54 | 0.130 | CPU |
| mlx | 279.71 | 330.19 | 0.410 | CPU |
| coreml/ALL | 110.90 | 120.62 | 0.155 | CPU |
| coreml/CPU_AND_GPU | 104.08 | 106.29 | 0.148 | CPU |
| coreml/CPU_AND_NE | 104.17 | 108.17 | 0.145 | CPU |
| coreml/CPU_ONLY | 104.50 | 112.39 | 0.159 | CPU |

## [router_only] deepseek-moe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 66.62 | 88.35 | 0.084 | CPU |
| torch_cpu_fp16 | 97.04 | 126.09 | 0.127 | CPU |
| mlx | 233.52 | 277.34 | 0.299 | CPU |
| coreml/ALL | 156.77 | 173.47 | 0.197 | CPU |
| coreml/CPU_AND_GPU | 155.75 | 158.58 | 0.192 | CPU |
| coreml/CPU_AND_NE | 156.00 | 162.59 | 0.201 | CPU |
| coreml/CPU_ONLY | 155.83 | 159.79 | 0.200 | CPU |

## [router_only] deepseek-moe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 71.33 | 89.92 | 0.093 | CPU |
| torch_cpu_fp16 | 131.85 | 156.40 | 0.149 | CPU |
| mlx | 237.81 | 266.17 | 0.303 | CPU |
| coreml/ALL | 542.71 | 614.00 | 0.606 | CPU |
| coreml/CPU_AND_GPU | 542.04 | 625.10 | 0.632 | CPU |
| coreml/CPU_AND_NE | 540.85 | 633.06 | 0.605 | CPU |
| coreml/CPU_ONLY | 539.42 | 592.71 | 0.593 | CPU |

## [router_only] deepseek-moe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 111.58 | 132.55 | 0.138 | CPU |
| torch_cpu_fp16 | 234.23 | 273.67 | 0.245 | CPU |
| mlx | 257.48 | 304.88 | 0.365 | CPU |
| coreml/ALL | 3070.71 | 3446.18 | 3.222 | CPU |
| coreml/CPU_AND_GPU | 3100.96 | 3431.58 | 2.915 | CPU |
| coreml/CPU_AND_NE | 3283.29 | 3488.59 | 2.545 | CPU |
| coreml/CPU_ONLY | 3055.56 | 3470.48 | 3.074 | CPU |

## [router_only] deepseek-moe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 357.75 | 425.06 | 0.556 | CPU |
| torch_cpu_fp16 | 519.40 | 617.85 | 0.659 | CPU |
| mlx ⭐ | 317.50 | 432.96 | 0.483 | CPU |
| coreml/ALL | 11358.00 | 11920.74 | 11.792 | ANE |
| coreml/CPU_AND_GPU | 9568.21 | 10156.08 | 8.831 | CPU |
| coreml/CPU_AND_NE | 11391.71 | 11885.52 | 13.228 | ANE |
| coreml/CPU_ONLY | 9432.88 | 10179.50 | 9.271 | CPU |

## [router_only] mixtral-8x22b  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 12.21 | 12.38 | 0.018 | CPU |
| torch_cpu_fp16 | 45.29 | 60.38 | 0.066 | CPU |
| mlx | 330.75 | 432.60 | 0.420 | CPU |
| coreml/ALL | 99.40 | 117.67 | 0.128 | CPU |
| coreml/CPU_AND_GPU | 103.83 | 114.51 | 0.129 | CPU |
| coreml/CPU_AND_NE | 98.73 | 111.00 | 0.134 | CPU |
| coreml/CPU_ONLY | 102.10 | 110.17 | 0.131 | CPU |

## [router_only] mixtral-8x22b  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 61.62 | 82.34 | 0.073 | CPU |
| torch_cpu_fp16 | 85.08 | 107.63 | 0.105 | CPU |
| mlx | 250.58 | 275.88 | 0.276 | CPU |
| coreml/ALL | 129.54 | 146.81 | 0.159 | CPU |
| coreml/CPU_AND_GPU | 129.88 | 144.96 | 0.166 | CPU |
| coreml/CPU_AND_NE | 129.10 | 147.04 | 0.166 | CPU |
| coreml/CPU_ONLY | 129.56 | 146.06 | 0.171 | CPU |

## [router_only] mixtral-8x22b  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 65.31 | 84.29 | 0.102 | CPU |
| torch_cpu_fp16 | 81.29 | 104.07 | 0.102 | CPU |
| mlx | 242.42 | 280.01 | 0.279 | CPU |
| coreml/ALL | 259.12 | 280.10 | 0.302 | CPU |
| coreml/CPU_AND_GPU | 261.21 | 290.57 | 0.306 | CPU |
| coreml/CPU_AND_NE | 262.33 | 295.64 | 0.308 | CPU |
| coreml/CPU_ONLY | 261.17 | 288.76 | 0.301 | CPU |

## [router_only] mixtral-8x22b  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 63.02 | 82.83 | 0.069 | CPU |
| torch_cpu_fp16 | 83.58 | 109.13 | 0.122 | CPU |
| mlx | 267.69 | 319.14 | 0.294 | CPU |
| coreml/ALL | 326.98 | 374.67 | 0.402 | CPU |
| coreml/CPU_AND_GPU | 319.50 | 347.26 | 0.513 | CPU |
| coreml/CPU_AND_NE | 318.29 | 345.90 | 0.333 | CPU |
| coreml/CPU_ONLY | 317.44 | 347.12 | 0.333 | CPU |

## [router_only] mixtral-8x22b  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 75.96 | 93.92 | 0.079 | CPU |
| torch_cpu_fp16 | 97.33 | 126.51 | 0.114 | CPU |
| mlx | 253.50 | 291.36 | 0.269 | CPU |
| coreml/ALL | 1595.12 | 1691.93 | 1.614 | CPU |
| coreml/CPU_AND_GPU | 1579.46 | 1684.59 | 1.646 | CPU |
| coreml/CPU_AND_NE | 1588.98 | 1692.58 | 1.715 | CPU |
| coreml/CPU_ONLY | 1592.96 | 1694.78 | 1.600 | CPU |

## [router_only] mixtral-8x22b  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 147.67 | 177.60 | 0.175 | CPU |
| torch_cpu_fp16 | 150.75 | 175.93 | 0.158 | CPU |
| mlx | 255.38 | 281.51 | 0.324 | CPU |
| coreml/ALL | 7962.17 | 9846.66 | 10.328 | ANE |
| coreml/CPU_AND_GPU | 7959.42 | 8428.35 | 8.223 | CPU |
| coreml/CPU_AND_NE | 7967.31 | 9888.41 | 10.649 | ANE |
| coreml/CPU_ONLY | 7920.00 | 8407.02 | 7.993 | CPU |

## [router_only] mixtral-8x22b  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 666.65 | 727.43 | 0.881 | CPU |
| torch_cpu_fp16 ⭐ | 277.69 | 367.06 | 0.280 | CPU |
| mlx | 326.21 | 397.69 | 0.401 | GPU |
| coreml/ALL | 54582.48 | 55630.38 | 55.950 | ANE |
| coreml/CPU_AND_GPU | 27250.50 | 27754.42 | 29.893 | GPU |
| coreml/CPU_AND_NE | 54625.98 | 55794.72 | 57.638 | ANE |
| coreml/CPU_ONLY | 29389.50 | 30255.57 | 29.953 | CPU |

## [router_only] mixtral-8x7b  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 10.54 | 10.67 | 0.019 | CPU |
| torch_cpu_fp16 | 44.73 | 58.14 | 0.117 | CPU |
| mlx | 336.90 | 385.04 | 0.574 | CPU |
| coreml/ALL | 88.17 | 97.54 | 0.120 | CPU |
| coreml/CPU_AND_GPU | 89.88 | 94.92 | 0.120 | CPU |
| coreml/CPU_AND_NE | 88.56 | 97.09 | 0.111 | CPU |
| coreml/CPU_ONLY | 91.19 | 105.46 | 0.195 | CPU |

## [router_only] mixtral-8x7b  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 54.92 | 74.71 | 0.072 | CPU |
| torch_cpu_fp16 | 82.75 | 109.55 | 0.086 | CPU |
| mlx | 237.08 | 276.62 | 0.330 | CPU |
| coreml/ALL | 109.29 | 122.81 | 0.140 | CPU |
| coreml/CPU_AND_GPU | 113.12 | 122.97 | 0.141 | CPU |
| coreml/CPU_AND_NE | 111.46 | 121.33 | 0.145 | CPU |
| coreml/CPU_ONLY | 111.50 | 125.09 | 0.143 | CPU |

## [router_only] mixtral-8x7b  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 53.38 | 75.05 | 0.071 | CPU |
| torch_cpu_fp16 | 79.56 | 100.85 | 0.084 | CPU |
| mlx | 244.56 | 274.29 | 0.249 | CPU |
| coreml/ALL | 192.33 | 208.05 | 0.233 | CPU |
| coreml/CPU_AND_GPU | 193.98 | 225.57 | 0.239 | CPU |
| coreml/CPU_AND_NE | 192.58 | 213.54 | 0.240 | CPU |
| coreml/CPU_ONLY | 192.67 | 215.97 | 0.243 | CPU |

## [router_only] mixtral-8x7b  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 55.48 | 74.63 | 0.063 | CPU |
| torch_cpu_fp16 | 81.71 | 102.77 | 0.100 | CPU |
| mlx | 269.73 | 312.77 | 0.503 | CPU |
| coreml/ALL | 250.00 | 282.84 | 0.269 | CPU |
| coreml/CPU_AND_GPU | 247.90 | 267.89 | 0.277 | CPU |
| coreml/CPU_AND_NE | 248.25 | 275.02 | 0.265 | CPU |
| coreml/CPU_ONLY | 249.58 | 286.96 | 0.273 | CPU |

## [router_only] mixtral-8x7b  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 63.88 | 82.00 | 0.082 | CPU |
| torch_cpu_fp16 | 92.44 | 113.63 | 0.102 | CPU |
| mlx | 250.15 | 281.23 | 0.350 | CPU |
| coreml/ALL | 1079.10 | 1155.85 | 1.108 | CPU |
| coreml/CPU_AND_GPU | 1058.29 | 1151.54 | 1.044 | CPU |
| coreml/CPU_AND_NE | 1067.71 | 1147.26 | 1.082 | CPU |
| coreml/CPU_ONLY | 1081.08 | 1161.03 | 1.049 | CPU |

## [router_only] mixtral-8x7b  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 115.81 | 132.51 | 0.130 | CPU |
| torch_cpu_fp16 | 128.50 | 152.88 | 0.150 | CPU |
| mlx | 257.42 | 278.98 | 0.361 | CPU |
| coreml/ALL | 5273.38 | 6259.96 | 5.714 | ANE |
| coreml/CPU_AND_GPU | 5416.29 | 6057.39 | 5.073 | CPU |
| coreml/CPU_AND_NE | 5308.62 | 6234.32 | 6.529 | ANE |
| coreml/CPU_ONLY | 5236.08 | 5467.18 | 4.809 | CPU |

## [router_only] mixtral-8x7b  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 459.02 | 517.21 | 0.549 | CPU |
| torch_cpu_fp16 ⭐ | 226.48 | 266.63 | 0.239 | CPU |
| mlx | 262.29 | 325.04 | 0.423 | GPU |
| coreml/ALL | 37261.08 | 38033.08 | 39.499 | ANE |
| coreml/CPU_AND_GPU | 19209.81 | 19847.23 | 22.065 | GPU |
| coreml/CPU_AND_NE | 37475.19 | 38242.70 | 38.781 | ANE |
| coreml/CPU_ONLY | 20626.42 | 20942.95 | 20.887 | CPU |

## [router_only] olmoe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 10.79 | 11.08 | 0.015 | CPU |
| torch_cpu_fp16 | 46.46 | 62.63 | 0.092 | CPU |
| mlx | 255.10 | 282.05 | 0.254 | CPU |
| coreml/ALL | 65.08 | 72.98 | 0.096 | CPU |
| coreml/CPU_AND_GPU | 65.25 | 67.88 | 0.094 | CPU |
| coreml/CPU_AND_NE | 65.12 | 70.64 | 0.102 | CPU |
| coreml/CPU_ONLY | 66.12 | 96.92 | 0.110 | CPU |

## [router_only] olmoe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 57.60 | 77.55 | 0.063 | CPU |
| torch_cpu_fp16 | 81.33 | 105.48 | 0.087 | CPU |
| mlx | 252.58 | 284.09 | 0.365 | CPU |
| coreml/ALL | 81.50 | 87.67 | 0.124 | CPU |
| coreml/CPU_AND_GPU | 81.38 | 87.96 | 0.120 | CPU |
| coreml/CPU_AND_NE | 81.50 | 89.00 | 0.122 | CPU |
| coreml/CPU_ONLY | 81.77 | 92.59 | 0.122 | CPU |

## [router_only] olmoe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 61.62 | 81.51 | 0.091 | CPU |
| torch_cpu_fp16 | 85.44 | 106.35 | 0.103 | CPU |
| mlx | 273.73 | 316.76 | 0.312 | CPU |
| coreml/ALL | 106.88 | 112.39 | 0.130 | CPU |
| coreml/CPU_AND_GPU | 106.67 | 114.72 | 0.145 | CPU |
| coreml/CPU_AND_NE | 106.50 | 111.00 | 0.149 | CPU |
| coreml/CPU_ONLY | 106.88 | 112.21 | 0.153 | CPU |

## [router_only] olmoe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 61.04 | 80.02 | 0.075 | CPU |
| torch_cpu_fp16 | 94.50 | 116.00 | 0.124 | CPU |
| mlx | 244.85 | 280.80 | 0.282 | CPU |
| coreml/ALL | 152.67 | 157.29 | 0.190 | CPU |
| coreml/CPU_AND_GPU | 152.58 | 158.29 | 0.185 | CPU |
| coreml/CPU_AND_NE | 147.62 | 158.44 | 0.195 | CPU |
| coreml/CPU_ONLY | 152.79 | 161.12 | 0.184 | CPU |

## [router_only] olmoe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 67.69 | 93.38 | 0.082 | CPU |
| torch_cpu_fp16 | 128.98 | 156.59 | 0.163 | CPU |
| mlx | 256.33 | 281.54 | 0.307 | CPU |
| coreml/ALL | 546.56 | 625.65 | 0.601 | CPU |
| coreml/CPU_AND_GPU | 541.19 | 597.12 | 0.598 | CPU |
| coreml/CPU_AND_NE | 539.23 | 616.51 | 0.609 | CPU |
| coreml/CPU_ONLY | 549.21 | 611.56 | 0.595 | CPU |

## [router_only] olmoe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 96.19 | 119.84 | 0.241 | CPU |
| torch_cpu_fp16 | 230.35 | 273.78 | 0.251 | CPU |
| mlx | 278.81 | 326.59 | 0.369 | CPU |
| coreml/ALL | 2975.63 | 3427.45 | 2.913 | CPU |
| coreml/CPU_AND_GPU | 2990.33 | 3402.22 | 2.806 | CPU |
| coreml/CPU_AND_NE | 2913.13 | 3399.18 | 2.944 | CPU |
| coreml/CPU_ONLY | 2941.52 | 3401.80 | 2.772 | CPU |

## [router_only] olmoe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 289.10 | 331.56 | 0.461 | CPU |
| torch_cpu_fp16 | 482.21 | 699.69 | 0.518 | CPU |
| mlx ⭐ | 278.77 | 340.75 | 0.382 | CPU |
| coreml/ALL | 11459.37 | 12000.21 | 12.213 | ANE |
| coreml/CPU_AND_GPU | 9236.67 | 10073.03 | 8.694 | CPU |
| coreml/CPU_AND_NE | 11409.19 | 11943.99 | 11.492 | ANE |
| coreml/CPU_ONLY | 9207.88 | 10069.85 | 9.239 | CPU |

## [router_only] qwen2-moe  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 10.71 | 11.96 | 0.016 | CPU |
| torch_cpu_fp16 | 46.58 | 61.59 | 0.073 | CPU |
| mlx | 265.48 | 308.43 | 0.290 | CPU |
| coreml/ALL | 62.17 | 69.67 | 0.109 | CPU |
| coreml/CPU_AND_GPU | 63.04 | 67.47 | 0.109 | CPU |
| coreml/CPU_AND_NE | 61.75 | 75.96 | 0.108 | CPU |
| coreml/CPU_ONLY | 63.04 | 82.95 | 0.100 | CPU |

## [router_only] qwen2-moe  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 58.88 | 79.67 | 0.076 | CPU |
| torch_cpu_fp16 | 81.98 | 105.38 | 0.110 | CPU |
| mlx | 245.35 | 280.18 | 0.292 | CPU |
| coreml/ALL | 76.21 | 84.75 | 0.116 | CPU |
| coreml/CPU_AND_GPU | 76.38 | 81.79 | 0.116 | CPU |
| coreml/CPU_AND_NE | 76.79 | 90.47 | 0.123 | CPU |
| coreml/CPU_ONLY | 75.96 | 85.55 | 0.116 | CPU |

## [router_only] qwen2-moe  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 59.65 | 80.38 | 0.106 | CPU |
| torch_cpu_fp16 | 85.35 | 109.02 | 0.099 | CPU |
| mlx | 262.67 | 310.13 | 0.284 | CPU |
| coreml/ALL | 106.62 | 111.88 | 0.148 | CPU |
| coreml/CPU_AND_GPU | 106.50 | 111.64 | 0.148 | CPU |
| coreml/CPU_AND_NE | 106.50 | 108.83 | 0.144 | CPU |
| coreml/CPU_ONLY | 106.58 | 114.84 | 0.155 | CPU |

## [router_only] qwen2-moe  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 61.40 | 80.59 | 0.068 | CPU |
| torch_cpu_fp16 | 95.83 | 122.24 | 0.120 | CPU |
| mlx | 259.04 | 305.04 | 0.344 | CPU |
| coreml/ALL | 152.62 | 164.72 | 0.195 | CPU |
| coreml/CPU_AND_GPU | 152.50 | 160.63 | 0.195 | CPU |
| coreml/CPU_AND_NE | 152.33 | 159.34 | 0.196 | CPU |
| coreml/CPU_ONLY | 152.67 | 167.47 | 0.196 | CPU |

## [router_only] qwen2-moe  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 64.38 | 84.51 | 0.156 | CPU |
| torch_cpu_fp16 | 128.96 | 151.92 | 0.142 | CPU |
| mlx | 254.04 | 283.09 | 0.396 | CPU |
| coreml/ALL | 540.96 | 616.27 | 0.583 | CPU |
| coreml/CPU_AND_GPU | 538.23 | 618.23 | 0.578 | CPU |
| coreml/CPU_AND_NE | 546.15 | 591.14 | 0.705 | CPU |
| coreml/CPU_ONLY | 541.79 | 640.49 | 0.585 | CPU |

## [router_only] qwen2-moe  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 99.65 | 120.93 | 0.113 | CPU |
| torch_cpu_fp16 | 218.23 | 261.03 | 0.239 | CPU |
| mlx | 264.65 | 296.21 | 0.370 | CPU |
| coreml/ALL | 3206.96 | 3427.19 | 2.709 | CPU |
| coreml/CPU_AND_GPU | 3205.15 | 3441.99 | 3.162 | CPU |
| coreml/CPU_AND_NE | 3192.71 | 3427.89 | 2.896 | CPU |
| coreml/CPU_ONLY | 3070.21 | 3428.53 | 3.284 | CPU |

## [router_only] qwen2-moe  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 294.46 | 338.59 | 0.519 | CPU |
| torch_cpu_fp16 | 485.50 | 751.44 | 0.504 | CPU |
| mlx | 300.98 | 382.40 | 0.449 | CPU |
| coreml/ALL | 11187.92 | 11868.90 | 11.391 | ANE |
| coreml/CPU_AND_GPU | 9060.79 | 9852.62 | 9.040 | CPU |
| coreml/CPU_AND_NE | 11236.08 | 11845.95 | 10.655 | ANE |
| coreml/CPU_ONLY | 9288.85 | 9971.02 | 9.969 | CPU |

## [router_only] router_stress_dense  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1871.23 | 2001.97 | 1.850 | CPU |
| torch_cpu_fp16 | 440.12 | 479.46 | 0.478 | CPU |
| mlx ⭐ | 406.46 | 573.53 | 0.622 | GPU |
| coreml/ALL | 875.19 | 1026.13 | 2.246 | ANE |
| coreml/CPU_AND_GPU | 578.67 | 750.94 | 1.476 | GPU |
| coreml/CPU_AND_NE | 882.42 | 1606.85 | 2.207 | ANE |
| coreml/CPU_ONLY | 857.27 | 886.30 | 0.909 | CPU |

## [router_only] router_stress_dense  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 4216.27 | 4836.52 | 4.086 | CPU |
| torch_cpu_fp16 | 603.67 | 685.01 | 0.589 | CPU |
| mlx ⭐ | 392.94 | 679.63 | 0.726 | GPU |
| coreml/ALL | 966.71 | 1690.00 | 1.912 | ANE |
| coreml/CPU_AND_GPU | 675.52 | 1278.19 | 1.767 | GPU |
| coreml/CPU_AND_NE | 954.42 | 1867.48 | 3.420 | ANE |
| coreml/CPU_ONLY | 1167.54 | 1253.36 | 1.268 | CPU |

## [router_only] router_stress_dense  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 4130.08 | 5067.80 | 4.016 | CPU |
| torch_cpu_fp16 | 869.29 | 1009.27 | 0.878 | CPU |
| mlx ⭐ | 391.94 | 434.19 | 0.751 | GPU |
| coreml/ALL | 1739.92 | 3347.73 | 3.861 | ANE |
| coreml/CPU_AND_GPU | 1936.19 | 2144.46 | 2.442 | GPU |
| coreml/CPU_AND_NE | 1382.44 | 2311.55 | 1.941 | ANE |
| coreml/CPU_ONLY | 840.79 | 949.85 | 0.863 | CPU |

## [router_only] router_stress_dense  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2127.48 | 2825.92 | 2.623 | CPU |
| torch_cpu_fp16 | 856.73 | 1478.84 | 1.491 | CPU |
| mlx ⭐ | 393.67 | 429.02 | 0.746 | GPU |
| coreml/ALL | 1345.48 | 2169.82 | 2.771 | ANE |
| coreml/CPU_AND_GPU | 1436.48 | 3155.17 | 2.748 | GPU |
| coreml/CPU_AND_NE | 1350.29 | 2276.06 | 2.876 | ANE |
| coreml/CPU_ONLY | 1086.94 | 1183.52 | 1.236 | CPU |

## [router_only] router_stress_dense  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 2125.77 | 2568.44 | 3.260 | CPU |
| torch_cpu_fp16 | 2970.54 | 5606.74 | 2.982 | CPU |
| mlx ⭐ | 495.27 | 1009.83 | 1.070 | GPU |
| coreml/ALL | 4911.04 | 6849.94 | 5.559 | ANE |
| coreml/CPU_AND_GPU | 6356.88 | 11332.58 | 8.984 | GPU |
| coreml/CPU_AND_NE | 5043.40 | 8701.82 | 7.496 | ANE |
| coreml/CPU_ONLY | 2945.06 | 3258.46 | 3.448 | CPU |

## [router_only] router_stress_dense  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 4812.77 | 5380.31 | 4.929 | CPU |
| torch_cpu_fp16 | 12136.92 | 20978.00 | 20.830 | CPU |
| mlx ⭐ | 988.02 | 1260.80 | 2.780 | GPU |
| coreml/ALL | 22083.69 | 25259.58 | 24.656 | ANE |
| coreml/CPU_AND_GPU | 37457.71 | 39159.98 | 38.111 | GPU |
| coreml/CPU_AND_NE | 22103.96 | 26008.00 | 24.742 | ANE |
| coreml/CPU_ONLY | 11199.04 | 13128.06 | 11.091 | CPU |

## [router_only] router_stress_dense  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 10273.10 | 10999.20 | 10.179 | CPU |
| torch_cpu_fp16 | 50796.75 | 57741.81 | 46.922 | CPU |
| mlx ⭐ | 2828.25 | 2923.72 | 4.235 | GPU |
| coreml/ALL | 134663.54 | 138116.66 | 146.859 | ANE |
| coreml/CPU_AND_GPU | 149807.15 | 152309.94 | 165.840 | GPU |
| coreml/CPU_AND_NE | 134565.00 | 141748.06 | 148.656 | ANE |
| coreml/CPU_ONLY | 52082.96 | 56954.54 | 50.294 | CPU |

## [router_only] stress  B=1
_null calibration: boundary=1.88µs, dispatch=302.31µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 ⭐ | 59.46 | 110.72 | 0.083 | CPU |
| torch_cpu_fp16 | 71.65 | 83.97 | 0.119 | CPU |
| mlx | 313.33 | 347.43 | 0.476 | CPU |
| coreml/ALL | 132.75 | 155.05 | 0.180 | CPU |
| coreml/CPU_AND_GPU | 134.58 | 162.55 | 0.196 | CPU |
| coreml/CPU_AND_NE | 131.88 | 142.39 | 0.159 | CPU |
| coreml/CPU_ONLY | 138.56 | 155.67 | 0.163 | CPU |

## [router_only] stress  B=2
_null calibration: boundary=2.04µs, dispatch=383.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 407.92 | 519.96 | 0.395 | CPU |
| torch_cpu_fp16 ⭐ | 130.04 | 155.35 | 0.155 | CPU |
| mlx | 314.81 | 353.53 | 0.375 | CPU |
| coreml/ALL | 363.98 | 408.12 | 0.676 | ANE |
| coreml/CPU_AND_GPU | 198.79 | 236.38 | 0.248 | CPU |
| coreml/CPU_AND_NE | 370.88 | 426.61 | 0.854 | ANE |
| coreml/CPU_ONLY | 196.94 | 237.37 | 0.245 | CPU |

## [router_only] stress  B=4
_null calibration: boundary=2.25µs, dispatch=473.33µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 418.65 | 521.21 | 0.426 | CPU |
| torch_cpu_fp16 ⭐ | 165.06 | 192.62 | 0.171 | CPU |
| mlx | 268.85 | 293.84 | 0.401 | CPU |
| coreml/ALL | 494.21 | 552.21 | 0.796 | ANE |
| coreml/CPU_AND_GPU | 316.79 | 376.94 | 0.439 | CPU |
| coreml/CPU_AND_NE | 494.75 | 530.45 | 0.745 | ANE |
| coreml/CPU_ONLY | 329.13 | 404.71 | 0.364 | CPU |

## [router_only] stress  B=8
_null calibration: boundary=2.83µs, dispatch=631.15µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 415.17 | 484.30 | 0.428 | CPU |
| torch_cpu_fp16 ⭐ | 235.67 | 281.88 | 0.254 | CPU |
| mlx | 263.19 | 307.04 | 0.402 | GPU |
| coreml/ALL | 689.21 | 761.13 | 1.018 | ANE |
| coreml/CPU_AND_GPU | 1000.35 | 2030.48 | 2.087 | GPU |
| coreml/CPU_AND_NE | 686.42 | 778.08 | 1.008 | ANE |
| coreml/CPU_ONLY | 494.83 | 587.88 | 0.623 | CPU |

## [router_only] stress  B=32
_null calibration: boundary=7.04µs, dispatch=2456.23µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 486.83 | 611.75 | 0.486 | CPU |
| torch_cpu_fp16 | 481.10 | 745.02 | 0.500 | CPU |
| mlx ⭐ | 314.79 | 404.18 | 0.497 | CPU |
| coreml/ALL | 2232.48 | 2390.08 | 2.553 | CPU |
| coreml/CPU_AND_GPU | 2208.27 | 2352.16 | 2.268 | CPU |
| coreml/CPU_AND_NE | 2263.44 | 2424.68 | 2.403 | CPU |
| coreml/CPU_ONLY | 2201.33 | 2303.97 | 2.265 | CPU |

## [router_only] stress  B=128
_null calibration: boundary=20.33µs, dispatch=11065.96µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 698.12 | 816.23 | 0.905 | CPU |
| torch_cpu_fp16 | 1500.92 | 2391.66 | 1.542 | CPU |
| mlx | 342.83 | 374.09 | 0.712 | CPU |
| coreml/ALL ⭐ | 0.00 | 0.00 | 0.000 | ANE |
| coreml/CPU_AND_GPU | 9379.08 | 9911.66 | 9.596 | CPU |
| coreml/CPU_AND_NE | 0.00 | 0.00 | 0.000 | ANE |
| coreml/CPU_ONLY | 9521.75 | 10320.74 | 9.694 | CPU |

## [router_only] stress  B=512
_null calibration: boundary=118.19µs, dispatch=42935.06µs (symmetric h=4096)_

| backend/unit | median µs | p95 µs | first ms | actual |
|---|---:|---:|---:|---|
| torch_cpu_fp32 | 1700.00 | 1890.12 | 1.750 | CPU |
| torch_cpu_fp16 | 5776.79 | 9070.01 | 5.762 | CPU |
| mlx ⭐ | 627.65 | 1063.38 | 1.423 | GPU |
| coreml/ALL | 137615.21 | 140970.09 | 142.723 | ANE |
| coreml/CPU_AND_GPU | 52317.33 | 57110.49 | 50.836 | GPU |
| coreml/CPU_AND_NE | 137509.67 | 141750.70 | 147.561 | ANE |
| coreml/CPU_ONLY | 42372.73 | 44729.63 | 42.659 | CPU |
