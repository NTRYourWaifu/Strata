# Strata：兩台電腦版

[English](README.md) · **繁體中文**

> Fork 自 [Niko1221/Strata](https://github.com/Niko1221/Strata)（MIT 授權）。Strata 原本的說明（安裝、模型、用法）：[README.upstream.zh-TW.md](README.upstream.zh-TW.md)。

這個 fork 只替 Strata 加了一個功能：**讓同一個模型跨區網上的兩台電腦執行**。第一台跑前 K 層，第二台跑剩下的層和輸出頭。
每台只把自己那幾層的專家放在自己的顯示卡和記憶體裡，所以兩台普通電腦加起來，能放下任何一台單獨放不下的模型。

它是為一個具體情況做的：Qwen3.8-Flash-Next **IQ3_XXS（記憶體＋視訊記憶體共需 47 GB）**，一台 32 GB 記憶體的桌機，
加一台 16 GB 記憶體的電競筆電。只用桌機約 **每秒 10 個 token**（放不下的專家得從 SSD 小塊隨機讀取）；
兩台一起跑，寫回答 **每秒 50～62 個 token**。

- [實測成績](#實測成績)
- [原理](#原理)
- [需要什麼](#需要什麼)
- [編譯](#編譯)
- [執行](#執行)
- [調校：真正有差的幾件事](#調校真正有差的幾件事)
- [限制與注意事項](#限制與注意事項)
- [為什麼不用現成的多機方案](#為什麼不用現成的多機方案)

## 實測成績

測試機器：

| | 第一台（桌機） | 第二台（筆電） |
| --- | --- | --- |
| 顯示卡 | RTX 3080 Ti，12 GB | RTX 4060 Laptop，8 GB |
| 記憶體 | 32 GB | 16 GB DDR5-5600，**只插一條（單通道）** |
| CPU | 支援 AVX-512 的桌機 CPU | i7-14650HX（不支援 AVX-512） |
| 網路 | 有線 1 Gbps，來回約 2 ms | 同左 |
| 系統 | Windows 10 | Windows 11 |

模型：Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS**，分界 **K = 32**（筆電跑第 32～47 層）。走伺服器、上下文 32K、
temperature 0.6、生成 800 token、英文題目。每一組都沒有從 SSD 讀專家（為什麼這很重要，見[調校](#調校真正有差的幾件事)）。

單位都是 token/s（每秒幾個 token；英文 1 token 約 0.75 個字）。

| 題目 | 暖機 | 正式 |
| --- | ---: | ---: |
| 聊天 | 50.0 | 44.3 |
| 寫程式 1（INI 解析器） | 64.9 | 61.5 |
| 寫程式 2（LRU 快取） | 66.9 | 60.5 |
| 寫程式 3（TS debounce） | 62.4 | 50.4 |
| 長題目 4,132 token：讀題 | 228（18.2 秒） | |
| 長題目：生成 | 48.1 | |
| 對話第二輪（4,268 token，沿用 4,237） | 總共 2.1 秒 | |

速度一路怎麼上來的（生成 token/s，三個短題目：寫程式／故事／解說）：

| 步驟 | 寫程式 | 故事 | 解說 |
| --- | ---: | ---: | ---: |
| 只用桌機、不開加速 | 10.5 | 11.1 | 10.4 |
| 兩台 K = 36、不開加速 | 17.1 | 16.0 | 28.6 |
| ＋MTP 猜字、調記憶體餘量、K = 32 | 27.7～29.6 | 19.0～19.6 | 32.3～35.1 |
| ＋筆電電源改「高效能」 | 56.5 | 38.1 | 58.1 |
| ＋兩台都開自適應專家快取 | 68.8～70.7 | 46.8～48.8 | 51.2～52.8 |

對照：同樣兩台電腦用 llama.cpp RPC 分片跑稠密 27B（加 MTP），寫程式約 52、聊天 28.8，讀題約 400 token/s。

**日常使用會慢一些。** 桌機開著瀏覽器、編輯器、聊天軟體時，可用記憶體只剩約 15 GiB，約 7 GiB 的專家要從 SSD 讀，
暖機後寫程式 40～43 token/s。

## 原理

```
 第一台（桌機）                                   第二台（筆電）
 ┌──────────────────────────────┐   TCP，區網   ┌──────────────────────────────┐
 │ 伺服器、tokenizer            │               │ strata --net-listen PORT     │
 │ 第 0 .. K-1 層               │ ── 中間結果 ─▶ │ 第 K .. 47 層＋輸出頭        │
 │ 自己的專家：顯卡＋記憶體(+SSD)│               │ 自己的專家：只放顯卡＋記憶體 │
 │ MTP 猜字層                   │ ◀── 選中的字 ─ │ 取樣                         │
 └──────────────────────────────┘ （＋最後一層） └──────────────────────────────┘
```

- Strata 原本就能在一台電腦裡，把模型按層切給兩張顯示卡（`--layer-split`）。兩段之間的交接只有一個地方：
  驗證器裡每個 token 一列、`hc × n_embd + n_embd + hc` 個 float。這個 fork 把這一列改成走 TCP，
  不再走共用的鎖頁記憶體。引擎本身的運算沒有改。
- Flash-Next 的隱藏維度只有 2560，每個 token 的交接資料約 **10 KB**。一個驗證視窗（4～5 個 token）只來回一次網路，
  實測每個視窗約 **3.5 ms**，網路不是瓶頸。
- 第二台自己取樣，只回傳選中的字。第一台要跑 MTP 猜字時，回覆會附上最後一層的殘差列（T × hc × n_embd 個 float，
  T = 4 時約 160 KB）。
- 下面這些加速功能是一項一項接回兩台模式的，每項都和單機輸出比對過：
  - **MTP 猜字**：留在第一台，用第二台傳回的殘差列。
  - **對話存檔點**：兩台各自存、讀自己那幾層的狀態，下一輪只讀新增的部分。
  - **批次讀題**：第二台處理第 c 段時，第一台同時讀第 c + 1 段。
  - **自適應專家快取**：兩台各自調整自己那幾層的快取；第二台在兩個視窗之間做。
  - **控制向量**（`--control-vector-scaled`）：每個請求的開關會轉給第二台。
- 協定（`include/strata/core/net_stage.hpp`，第 4 版）：`Hello`、`Run`、`Commit`、`Zero`、`Logits`、`Bye`、`Save`、
  `Restore`、`Prefill`、`Cvec` 十種訊息，格式都是 `[u32 magic][u32 op][u64 長度][內容]`。

改了哪些程式：`src/core/net_stage.cpp`、`include/strata/core/net_stage.hpp`（新檔）、`src/program/generate.cpp`
（新選項、每台只載自己那段、第二台的訊息迴圈）、`src/core/verify.cpp`／`.hpp` 和 `src/prefill/prefill.cpp`／`.hpp`
（下一段在另一台時改走網路）、`CMakeLists.txt`。

## 需要什麼

- 兩台有 NVIDIA 顯示卡、在同一個區網的電腦，**盡量用有線**。Wi-Fi 也能跑，但延遲忽高忽低，每個視窗都會被拖慢。
- 兩台的記憶體＋視訊記憶體合計放得下模型。IQ3_XXS 約 47 GB，不含留在 SSD 上的 27 GB 查表檔。
- 兩台用**同一份** `strata` 編譯結果（連線時會檢查協定版本）。
- 兩台都要有模型檔。第二台只會讀 GGUF 檔頭、非專家張量，以及第 K 層之後的專家，所以可以只放這些部分的稀疏檔。

## 編譯

用 CUDA 從原始碼編譯。有兩個旗標很重要：

- 只要有一台 CPU 不支援 AVX-512（大部分筆電），就要加 **`-DSTRATA_PORTABLE=ON`**。一般編譯會把 ggml-cpu 編成 AVX-512，
  拿到另一台一跑就當掉。**兩台都**用可攜版。
- Windows 上用 CUDA 12.x 要加 **`-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`**，不然連結會失敗（LNK1169）。
- `-DCMAKE_CUDA_ARCHITECTURES="86;89"`（換成你兩張卡的架構）讓同一個執行檔兩台都能用。

```bat
:: 在「x64 Native Tools」命令列（Visual Studio 2022 Build Tools）裡、儲存庫資料夾中執行
cmake -G Ninja -S . -B build-port -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_PORTABLE=ON -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=86;89" ^
  "-DSTRATA_GGML_DIR=<llama.cpp 的路徑，setup.py 裝好的那份>"
cmake --build build-port --target strata -j 8
```

把 `build-port\strata.exe`（和它需要的 CUDA 執行階段 DLL）複製到第二台。

## 執行

兩台的引擎選項都和單機一樣（pack、模型、`--expert-profile`、`--resident-experts`、`--spec` 等），再加上：

| 選項 | 哪一台 | 意思 |
| --- | --- | --- |
| `--net-listen PORT` | 第二台 | 在 PORT 等第一台連過來（所有網卡） |
| `--net-split K` | 兩台 | 第二台從第幾層開始（兩台要填同一個 K） |
| `--net-connect HOST:PORT` | 第一台 | 連到第二台 |

啟動時會檢查：一定要有 `--resident-experts` 和 `--expert-profile`；不能和 `--layer-split`（一台電腦兩張卡）一起用。

1. **先開第二台。** 執行 `strata.exe <選項> --net-listen 47700 --net-split 32`。它載入自己那幾層，準備好時印出 `NETREADY`。
2. **再開第一台。** 照平常啟動伺服器，在伺服器設定檔的引擎 `args` 裡加上 `--net-connect <第二台 IP>:47700 --net-split 32`
   （`python -m serve.server --engine strata --config <config.json>`）。
3. 第一台斷線後，第二台會繼續等下一次連線，可以一直開著；要把它的顯示卡讓出來時再關掉。

有用的環境變數：

| 變數 | 用途 |
| --- | --- |
| `STRATA_RESIDENT_HEADROOM_GIB` | 這台要保留多少記憶體給系統。我們桌機設 3、筆電設 1 |
| `STRATA_WATCHDOG_S` | 卡住偵測的秒數；第二台的 SSD 可能偶爾卡住時要調高（我們用 120～600） |
| `STRATA_NET_TRACE=1` | 記下每一則網路訊息 |
| `STRATA_NET_EXACT=1` | 測試模式：搭配固定的 `--expert-cache N`，每台只放「單機 N 格快取裡屬於自己那幾層」的專家，才能逐字比對輸出 |

我們最好的設定（K = 32）：

- 第一台：`--spec 4 --spec-min-p 0.5 --mtp <mtp 資料夾> --prefill 2048 --pcie-frac 0 --suffix-draft 0 --expert-cache auto --vram-reserve-mib 700`，`STRATA_RESIDENT_HEADROOM_GIB=3`。
- 第二台：`--spec 4 --vram-reserve-mib 300`，`STRATA_RESIDENT_HEADROOM_GIB=1`。

## 調校：真正有差的幾件事

依影響大小排列：

1. **筆電的 Windows 電源計畫。** 「平衡」會壓住 CPU，而放在記憶體裡的專家是 CPU 在算：筆電那段每個視窗要 78 ms。
   改成「高效能」後變 24 ms，**總速度翻倍**（27.7 → 56.5 token/s）。用 `powercfg /getactivescheme` 確認。
2. **專家絕對不要掉到 SSD。** 桌機只要有 1～4 GiB 的專家放不進記憶體，速度就掉 20～50%（56～59 → 29～44）。
   通常是電腦上開著別的程式造成的。比速度時，只拿可用記憶體相同的兩組來比。
3. **從哪裡切（K）。** 第二台分到的層數，剛好塞滿它的視訊記憶體＋記憶體、不讀 SSD 就好，其餘交給記憶體多的那台。
   K 從 36 改成 32，桌機每題從讀 SSD 約 36 GB 降到 0。
4. **沒有溢出時，MTP 才有用。** 一次驗 4～5 個猜的字，需要的專家也多 4～5 倍。專家還在 SSD 上時 MTP 幾乎沒幫助；
   全部放進記憶體後提升很大。
5. **讀題的分段大小。** `--prefill 2048` 讀 4.5K token 題目是 448 token/s；8192 只有 344（整題一段，兩台沒辦法同時做），
   1024 是 364。
6. **不要強制加 CPU 執行緒。** Strata 預設每個實體核心一條。用 `--pool-workers` 加滿，兩台都變慢：專家運算卡在記憶體頻寬。
7. **筆電瘦身。** 移除吃資源的常駐軟體、分頁檔改固定大小。我們的筆電可分配記憶體從 18.7 GB 變 31.7 GB，
   Strata 才敢把專家鎖進記憶體。

試過但不值得：`--spec 3` 或 `5`（4 最好）、把沒命中的專家分給 PCIe（沒差）、suffix 猜字（只對「重複前文的改程式」有用）。

## 限制與注意事項

- **沒有驗證。** `--net-listen` 會接受任何連到這個埠的連線。只在家用區網使用，防火牆規則要限定區網網段。
- **兩台、每台一張卡。** 最多兩段；不能和 `--layer-split` 或輔助顯示卡一起用。
- **兩台模式下關閉：** 對話停車（`conversation_cache_mib`）、同時處理多個請求（`parallel`）。不支援 `window_logprobs`。
- **不同世代的顯示卡，運算捨入不同。** 同型號顯示卡上，兩台的輸出和單機逐字相同（測到 4,461 token 的題目）。
  RTX 3080 Ti（Ampere）和 RTX 4060（Ada）之間，貪婪取樣在幾十個 token 後可能換一個用詞，文字仍然通順。
  這是硬體造成的，不是網路程式的問題。
- **第一次讀題比單機慢。** 我們的機器在 32K 上下文約 230～255 token/s。之後每輪靠存檔點，只讀新增的部分。
- 只在 Windows 上測過。socket 程式有 Linux 用的 BSD sockets 寫法，但沒測過。
- 基於 Strata commit `d5ea713`（0.1.40.3）。官方之後還有更新，這個分支沒有跟上。

## 為什麼不用現成的多機方案

2026-10-08 查各專案自己的文件：

| 專案 | 為什麼不行 |
| --- | --- |
| prima.cpp | 不支援 Windows；沒有這種架構的 MoE 模型 |
| exo | Windows＋CUDA 列在「長期規劃」 |
| distributed-llama | 只有 CPU／Vulkan；最大到 Qwen3 30B-A3B |
| Parallax | 權重要全放視訊記憶體（我們兩台合計 20 GB）；要 WSL2 |
| llama.cpp RPC | 跑得動，但同一顆模型、在單機全放記憶體的條件下，llama.cpp 只有 9～16 token/s，Strata 有 42～58；跨兩台只會更慢（我們實測 Coder 模型 6 token/s） |
| Strata 官方 | 作者表示不打算做跨機（issue #1256） |

我們也用 llama.cpp RPC 的 `-sm tensor` 實測過跨機張量並行（兩台同時算每一層），有線 Gigabit 下只有 0.6～0.7 token/s。
沒有 RDMA 網路的話，按層切是唯一可行的做法。
