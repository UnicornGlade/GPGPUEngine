# Задача

## Сообщения пользователя

- «теперь добавь новый vulkan unit-test в нем мы сравним насколько быстро возможно заматчить 40000 128 x float SIFT дескрипторов одной и другой картинки, данные создай случайные, сверяй результат с brute force поиском на CPU многопоточным, а на видеокарте меня интересуют две реализации - Brute Force матчинг как в OpenCV (с использованием локальной памяти), и GEMM-like вариант (чтобы в т.ч. получить ускорение благодаря Tensor cores, WMMA вероятно должно получиться использовать через NVIDIA extension или через cooperative matrices) - обязательно вычисли и логгируй не только время но и сколько достигнуто GFLOPS»
- «можно ли ускорить еще дальше? если да - то добавь еще отдельный кернел (чтобы легко было его сравнить с уже имеющимся GEMM-like) и в нем попробуй оптимизации разные, если есть сомнения - спроси»

## Лог действий

- Добавлен новый Vulkan test `vulkan.siftMatchBenchmark`.
- Добавлены два compute shader-а: `sift_match_bruteforce_local.comp` и `sift_match_gemm_like.comp`.
- Подключены новые kernels в `libs/gpu/CMakeLists.txt` и `kernels.{h,cpp}`.
- В тесте реализованы:
  - генерация случайных нормализованных `128 x float` descriptor-ов;
  - многопоточный CPU brute-force reference;
  - GPU brute-force matcher с local/shared memory;
  - GPU GEMM-like blocked matcher;
  - сверка GPU top-1 match против CPU reference;
  - логирование времени и достигнутых GFLOPS.
- Добавлены env override-параметры:
  - `GPGPU_VULKAN_SIFT_MATCH_COUNT`
  - `GPGPU_VULKAN_SIFT_MATCH_ITERS`
  - `GPGPU_VULKAN_SIFT_MATCH_CPU_THREADS`
- Проект пересобран, затем тест проверен:
  - на уменьшенном размере `1024 x 1024`;
  - на полном размере `40000 x 40000` на RTX 4090.
- Добавлен ещё один отдельный kernel `sift_match_gemm_like_vec4.comp`:
  - `vec4`-загрузка descriptor-ов;
  - query descriptor держится в регистрах;
  - train tile кладётся в shared memory;
  - сравнение добавлено в тот же benchmark рядом с brute-force и исходным GEMM-like.
- Новый kernel проверен:
  - на размере `2048 x 2048`;
  - на полном размере `40000 x 40000` на RTX 4090.

## Результат

- Новый тест собирается и проходит.
- На RTX 4090 для `40000 x 40000` и `128`-мерных descriptor-ов получены ориентировочно:
  - CPU brute-force: `~5.11 s`, `~80.1 GFLOPS`
  - Vulkan brute-force local: `~98.2 ms`, `~4169 GFLOPS`
  - Vulkan GEMM-like: `~58.2 ms`, `~7038 GFLOPS`
- После добавления `vec4/register-tiled` kernel на RTX 4090 для `40000 x 40000` получены ориентировочно:
  - CPU brute-force: `~4.93 s`, `~83.1 GFLOPS`
  - Vulkan brute-force local: `~99.6 ms`, `~4112 GFLOPS`
  - Vulkan GEMM-like: `~59.3 ms`, `~6908 GFLOPS`
  - Vulkan GEMM-like vec4: `~29.9 ms`, `~13688 GFLOPS`
- В текущем driver/device config тест логгирует `cooperative_matrix_fp32=no`, то есть GEMM-like путь сейчас реализован как blocked FMA kernel, без активированного fp32 cooperative-matrix path.
