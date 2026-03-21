# Задача

## Сообщения пользователя

- «теперь добавь новый vulkan unit-test в нем мы сравним насколько быстро возможно заматчить 40000 128 x float SIFT дескрипторов одной и другой картинки, данные создай случайные, сверяй результат с brute force поиском на CPU многопоточным, а на видеокарте меня интересуют две реализации - Brute Force матчинг как в OpenCV (с использованием локальной памяти), и GEMM-like вариант (чтобы в т.ч. получить ускорение благодаря Tensor cores, WMMA вероятно должно получиться использовать через NVIDIA extension или через cooperative matrices) - обязательно вычисли и логгируй не только время но и сколько достигнуто GFLOPS»
- «можно ли ускорить еще дальше? если да - то добавь еще отдельный кернел (чтобы легко было его сравнить с уже имеющимся GEMM-like) и в нем попробуй оптимизации разные, если есть сомнения - спроси»
- «коммить, пуш, после этого действительно добавь еще один кернел - который работает с fp16»
- «да, результат на fp16 может немного отличаться, но тем не менее добавь логгирование - сколько результатов отличается от результата в fp32»
- «и добавь RootSIFT нормализацию»
- «да, хочу на tensor cores»
- «не уберай этот пусть, что тебе нужно установить чтобы все было ок?»
- «посмотри где в репозитории скрипт который устанавливает glslc?»
- «обнови его чтобы он устанавливал более свежую версию и скажи какой его кусок выполнить мне в терминале чтобы установить эту новую версию»
- «установил, обновил, действуй дальше»

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
- Добавлен ещё один отдельный kernel `sift_match_gemm_like_fp16_packed.comp`:
  - descriptors упаковываются на host в `fp16` по две компоненты в один `uint32_t`;
  - shader использует `unpackHalf2x16(...)`;
  - корректность проверяется против CPU brute-force на тех же уже-квантованных `fp16` данных;
  - дополнительно логгируется, сколько top-1 результатов отличается от исходного `fp32` reference.
- Synthetic descriptors переведены на RootSIFT normalization:
  - случайные неотрицательные компоненты;
  - `L1 normalize -> sqrt`.
- Новый kernel проверен:
  - на размере `2048 x 2048`;
  - на полном размере `40000 x 40000` на RTX 4090.
- Найдено, что старый system `glslc` не поддерживает cooperative matrices; обновлён `scripts/linux/install_vulkan_sdk.sh`, чтобы он ставил свежие `shaderc/glslc` и `glslang`.
- После обновления toolchain добавлены tensor-core kernels:
  - `sift_match_tensor_cores.comp` для `VK_KHR_cooperative_matrix`;
  - `sift_match_tensor_cores_nv.comp` для `VK_NV_cooperative_matrix`.
- В Vulkan host code добавлено включение нужных feature bits:
  - `shaderFloat16`;
  - `storageBuffer16BitAccess`;
  - `vulkanMemoryModel`;
  - `VK_KHR_cooperative_matrix` или `VK_NV_cooperative_matrix` feature chain, если поддерживается устройством.
- В benchmark добавлен runtime detection tensor-core backend:
  - `KHR`, если доступен `VK_KHR_cooperative_matrix`;
  - `NV`, если доступен только `VK_NV_cooperative_matrix`.
- На RTX 4090 с драйвером `535.171.4` оказалось, что рабочий tensor-core path идёт через `VK_NV_cooperative_matrix`, а не через `VK_KHR_cooperative_matrix`.
- После этого benchmark перепроверен:
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
- После добавления `fp16 packed` kernel и RootSIFT normalization на RTX 4090 для `40000 x 40000` получены ориентировочно:
  - CPU brute-force RootSIFT fp32: `~5.15 s`, `~79.5 GFLOPS`
  - CPU brute-force RootSIFT fp16-quantized: `~5.69 s`, `~72.0 GFLOPS`
  - CPU fp16 drift vs fp32: `192 / 40000` (`~0.48%`) top-1 differences
  - Vulkan brute-force local: `~99.9 ms`, `~4099 GFLOPS`
  - Vulkan GEMM-like fp16 packed: `~39.2 ms`, `~10453 GFLOPS`, top-1 differences vs fp32: `192 / 40000` (`~0.48%`)
  - Vulkan GEMM-like fp32: `~60.3 ms`, `~6794 GFLOPS`
  - Vulkan GEMM-like vec4 fp32: `~32.0 ms`, `~12808 GFLOPS`
- В текущем driver/device config тест логгирует `cooperative_matrix_fp32=no`, то есть GEMM-like путь сейчас реализован как blocked FMA kernel, без активированного fp32 cooperative-matrix path.
- После добавления NV tensor-core path на RTX 4090 для `40000 x 40000` получены ориентировочно:
  - CPU brute-force RootSIFT fp32: `~5.12 s`, `~80.0 GFLOPS`
  - CPU brute-force RootSIFT fp16-quantized: `~5.66 s`, `~72.3 GFLOPS`
  - CPU fp16 drift vs fp32: `192 / 40000` (`~0.48%`) top-1 differences
  - Vulkan brute-force local: `~96.8 ms`, `~4229 GFLOPS`
  - Vulkan GEMM-like fp16 packed: `~39.7 ms`, `~10312 GFLOPS`, top-1 differences vs fp32: `192 / 40000` (`~0.48%`)
  - Vulkan tensor-core RootSIFT matcher (NV): `~18.5 ms`, `~22093 GFLOPS`, top-1 differences vs fp32: `193 / 40000` (`~0.4825%`)
  - Vulkan GEMM-like fp32: `~62.0 ms`, `~6611 GFLOPS`
  - Vulkan GEMM-like vec4 fp32: `~32.0 ms`, `~12786 GFLOPS`
- Итог: tensor-core backend через `VK_NV_cooperative_matrix` примерно в `1.73x` быстрее текущего `vec4 fp32` kernel на целевом размере.
