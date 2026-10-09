<h1 align="center">Strata</h1>

[English](README.upstream.md) · [简体中文](README.zh-CN.md) · **繁體中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

> **這是 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的 fork，新增「兩台電腦跑同一個模型」。** 本 fork 的說明在 [首頁（繁體中文）](README.zh-TW.md)。以下是原版 Strata 的 README。

<p align="center"><b>在你自己的遊戲電腦上執行 1250 億參數的 AI 模型</b><br>
NVIDIA 或 AMD 顯示卡（12 GB 及以上）· Windows 或 Linux · 免費開源</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="Strata 的模型寫出的體素寶塔花園，在瀏覽器中執行"></a><br>
<sub>體素寶塔花園，一次提示生成，在 RTX 5070 上用 Strata 執行（IQ3_S，128K 上下文）·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">完整影片（49 秒）</a></sub></p>

Strata 能在普通電腦上執行 **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**。這是一個又大又聰明的
AI 模型，通常要用伺服器才能跑。它能聊天、寫程式碼、看圖片，還能配合你的應用和程式設計代理一起工作。所有資料都留在你的電腦上。

## 速度有多快？

我們在兩臺普通的遊戲電腦上做了測試。一個 token 大約相當於 ¾ 個英文單詞。

- **寫回答：** 短對話中回覆出現的速度。每秒 60 個 token 已經比你的閱讀速度還快。
- **讀提示：** 讀入你傳送內容的速度（這裡是一份 32K token 的文件、程式碼或聊天記錄）。

<table>
<tr><th>NVIDIA：RTX 5070（12 GB）、Ryzen 5 7600、64 GB 記憶體</th><th>AMD：RX 9070 XT（16 GB）、Ryzen 9 3900X、47 GB 記憶體</th></tr>
<tr><td>

| 規格 | 寫回答 | 讀提示 |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| 規格 | 寫回答 | 讀提示 |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA：Q2_0 用的是引擎 0.1.36，其他各行用的是 0.1.26（4K 回答，32K 提示）。完整表格見
[DETAILS.md](docs/DETAILS.md#speed-measured)。視訊記憶體越大的顯示卡越快：RTX 3090（24 GB）寫回答應該能達到每秒約
100-140 個 token。長對話和其他顯示卡的資料：[各模型的速度](docs/MODELS.md#how-fast-is-each-size)、
[社群測試結果](docs/COMMUNITY_BENCHMARKS.md)。

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="請我喝杯咖啡" height="50"></a><br>
<sub>Strata 是免費的。如果它在你的電腦上跑得不錯，請我喝杯咖啡，讓這個專案繼續下去。</sub></p>

## 你需要什麼

| | |
| --- | --- |
| **顯示卡** | **NVIDIA** GeForce RTX 20、30、40 或 50 系列，或 **AMD** Radeon RX 7900 XT / XTX、RX 7800 XT / 7700 XT、RX 9060 XT、RX 9070 / 9070 XT、Radeon AI PRO R9700 或 RX 6800 / 6900 系列。需要 **12 GB 或以上視訊記憶體**。 |
| **記憶體** | 32 GB 或以上。記憶體大小決定能裝下[哪個模型](#該選哪個模型)。64 GB 可以執行所有規格。 |
| **硬碟** | 約 80 GB 可用空間。儘量用 SSD：第一次啟動會快很多。 |
| **系統** | Windows 10 / 11 或 Linux，以及 NVIDIA 或 AMD 的最新顯示卡驅動。 |

其他的都由安裝程式搞定。兩到三張顯示卡可以分擔同一個模型（[多 GPU](docs/MULTI_GPU.md)）。

實驗性支援，由社群成員在自己的機器上編寫和測試：

- **較老的顯示卡**（Tesla P40 / V100、GTX 10、Radeon VII / MI50、RX 6700 XT、RX 5500 XT）：[較老的 GPU](docs/OLDER_GPUS.md)。
- **Intel Arc**，在 Linux 上從原始碼建置：[Intel Arc](docs/INTEL_ARC.md)。
- **AMD Ryzen AI Max（Strix Halo）**，在 Linux 上從原始碼建置：[Strix Halo](docs/STRIX_HALO.md)。
- **不支援 AVX2 的老處理器**：能用，但很慢。[較老的 CPU](docs/INSTALL.md#older-cpus-experimental)。

完整列表：[docs/INSTALL.md](docs/INSTALL.md#what-you-need)。

## 安裝

### 讓你的 AI 幫你裝

你在用 AI 程式設計助手嗎（Claude Code、Cursor、Codex、GitHub Copilot 等）？把下面這段貼上給它：

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

它會檢查你的顯示卡、記憶體和硬碟，選出合適的模型。然後安裝並啟動它，再告訴你怎麼連線你的應用。AI 工具也可以透過
Strata 的 [MCP 伺服器](docs/MCP_SERVER.md)來安裝、啟動和停止 Strata。

### 或者自己動手

[下載 Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) 並解壓（或者用 `git clone`）。
**Windows：** 雙擊 **`START-HERE.bat`**。**Linux：** 在 Strata 資料夾裡執行 **`./setup.sh`**。

NVIDIA 和 AMD 的步驟完全一樣。安裝程式會偵測你的顯示卡，並裝好對應的引擎。它會問你幾個問題：

- 用哪個模型、哪個規格，
- 上下文多大（模型能記住多少文字），
- 是否要辨識圖片。

每次直接按 Enter就是推薦選項。之後它會下載模型（約 70 GB）並啟動。如果下載中斷了，再執行一次即可：它會從中斷的地方繼續。
瀏覽器會開啟 Strata 應用，網址是 `http://127.0.0.1:8080`。

> **模型啟動時，你的電腦可能會變慢或卡住 1-3 分鐘**（第一次最久）。
> Strata 會把 35-55 GB 載入到記憶體，並為顯示卡鎖定其中一部分。這是正常的。請耐心等待，不要關閉視窗。
> 視窗裡會顯示 Strata 正在做什麼。

**下次使用時**，再執行一次 `START-HERE.bat`（或 `./setup.sh`）。它會馬上啟動，已下載的東西不會重複下載。關閉它的視窗就能停止模型。
`UPDATE.bat`（`./update.sh`）只更新 Strata，不啟動。更新、Docker、多張顯示卡、檔案存放位置以及所有選項：
[docs/INSTALL.md](docs/INSTALL.md)。

## 該選哪個模型

安裝程式會根據你的記憶體推薦一個。同一個模型有幾種規格，壓縮程度不同。規格越小越快，越大越聰明一些。

| 你的記憶體 | 選擇 | 原因 |
| --- | --- | --- |
| **32 GB** | **Coder** | 32 GB 裝得下，而且專為程式碼打造（如果顯示卡是 24 GB，Q2_0 和 IQ2_XS 也能跑） |
| **48 GB** | **IQ2_XS**（或 Q2_0，最快） | 更大的規格裝不下 |
| **64 GB** | **IQ2_XS**（推薦），或 IQ3_XXS / IQ3_S | 所有規格都裝得下；IQ3_S 最好，也最慢 |
| **96 GB 或以上** | **IQ3_S**，或 Unsloth 的 UD-IQ4_XS（約 4-bit） | 開著其他程式也能放下最大的規格 |

- **[Coder](docs/MODELS.md#coder)：** 程式設計版本，去掉了一半專家。它達到完整模型 SWE-bench Verified 分數的 91%（由其作者測得），
  32 GB 記憶體就能裝下。程式碼以外的能力較弱，包括中文和其他中日韓文本（#438）。這類用途請選 Q2_0、IQ2_XS 或 IQ3_S，
  它們保留了所有專家。
- **[Swift 1.5](docs/MODELS.md#swift-15)：** 一個微調版本，回答前思考的時間短得多。你能更早拿到答案，品質基本不變。
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)：** Unsloth 的約 4-bit 版本，品質介於 IQ3_S 和 UD-Q4_K_XL 之間。
  下載 94 GB。記憶體少於約 80 GB 時，Strata 回答時要從 SSD 讀取其中一部分，所以會更慢（NVMe SSD 有幫助）。
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)**（實驗性）：最接近完整模型。但 Strata
  回答時要從 SSD 讀取其中大部分內容，所以在 64 GB 的電腦上每秒只能寫 7-8.5 個 token。
- **[OrcaRouter 的 Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)：** 需要手動設定，不在安裝程式的選單裡。

規格、下載以及各配置能裝下什麼：[docs/MODELS.md](docs/MODELS.md)。以後想再新增模型，執行
`SETUP.bat`（Linux：`./setup.sh --setup`）。

## 使用方法

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="Strata 應用的 Monitor 標籤頁，旁邊是一個程式設計代理"><br>
<sub>Strata 應用的 <b>Monitor</b>（左），程式設計代理正在寫影片裡的寶塔花園（右）</sub></p>

- **在瀏覽器裡：** 開啟 `http://127.0.0.1:8080`。裡面有 **Chat**（聊天）、即時 **Monitor**（監控模型和你的
  GPU/CPU/記憶體），以及 **About**（設定和網址）。
- **你的應用和程式設計代理：** 新增一個「OpenAI 相容」的提供商，base URL 填 **`http://127.0.0.1:8080/v1`**。
  API key 和模型名隨便填都行。
  - 使用 Anthropic API 的應用：`http://127.0.0.1:8080/v1/messages`（Claude Code：
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）。
  - Codex CLI 和其他使用 OpenAI Responses API 的應用：`/v1/responses`
    （[設定方法](docs/DETAILS.md#the-responses-api-and-codex-cli)）。
- **思考：** 在聊天選單或應用的「reasoning effort」裡選擇 **off、low、medium 或 high**。off 最快，
  high 最適合難題。
- **圖片：** 在安裝時對「Images?」選是。然後在聊天裡點 **Picture**，或在你的應用裡附上圖片。
  AMD 顯示卡在 Linux 上透過處理器辨識圖片；在 Windows 上暫時還不行。
- **從手機或另一臺電腦連線：** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`。一定要設定 key。
- **一次處理一個請求：** 預設情況下 Strata 一次只回答一個請求，其他請求排隊等待。想同時回答多個，
  設定 `"parallel": 2`（[BATCHING.md](docs/BATCHING.md)）。在 12 GB 顯示卡上，這會讓每個回答變慢。
- **長提示：** Strata 會完整讀入對話的第一條訊息，大約每 30,000 個 token 需要 1 分鐘。之後的訊息幾秒內就開始回答。

更多：[聊天記錄存在哪裡](docs/INSTALL.md#where-things-are-stored)、[API](docs/DETAILS.md#using-it)。

## 出問題了？

- **Strata 第一次啟動時電腦卡死了。** 載入模型時這是正常的。請耐心等待，不要關閉視窗。
  10 分鐘後還卡著？重啟電腦，關掉其他程式再試一次，或者換一個更小的規格。
- **下載或安裝時中斷了。** 再執行一次 `START-HERE.bat`（或 `./setup.sh`）。它會從中斷的地方繼續。
- **非常慢，硬碟燈一直閃，或者提示「the engine stopped unexpectedly」。** 你的電腦可用記憶體不夠。
  關掉其他程式（瀏覽器很佔記憶體），或者換一個更小的規格（Q2_0 或 IQ2_XS）。
- **提示 8080 埠已被佔用。** Strata 已經在執行了。找一下它的視窗。

更多問題和解決辦法：[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。還是解決不了？提交一個
[issue](https://github.com/Niko1221/Strata/issues)，並附上 Strata 資料夾裡的 `strata-<model>.log`。發現了安全問題？
請私下報告：[SECURITY.md](SECURITY.md)。

## 它是怎麼工作的？

這類模型通常執行在擁有數百 GB 視訊記憶體的伺服器上。而你的顯示卡只有 12-24 GB。Strata 讓模型裝得下的辦法是
**把工作分攤到整臺電腦上**。可以想像一個廚房：常用的東西放在臺面上，其餘的放在儲藏室裡。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="模型的 24,576 個專家：最常用的在顯示卡上，全部在記憶體裡，一張查詢表在 SSD 上"></p>

- **這個模型是由 24,576 個小專家（「experts」）組成的團隊。** 每個詞只需要其中 10 個。
- **你的顯示卡** 存放最常用的幾千個專家。**你的記憶體** 存放全部專家，
  **你的處理器** 同時處理其餘的部分。**你的 SSD** 存放一張大查詢表。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="一個小助手猜測接下來的幾個詞；大模型一次性全部檢查，保留正確的"></p>

- **先猜，再檢查：** 一個小助手先猜接下來的幾個詞，大模型再一次性檢查它們。答案完全一樣，但快 1.6-1.8 倍。
- **長文本分大塊讀入**（每次最多 8,192 個 token），速度超過每秒 1,000 個 token。

更詳細的解釋：[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)。每個部分及其資料：
[詳細說明](docs/DETAILS.md#how-it-works)和[論文](docs/paper/Strata-Paper.pdf)。

## 致謝與授權

模型是 Qwen 團隊的 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。它由
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）
和 Unsloth 壓縮。Strata 使用了 [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 的部分程式碼。完整致謝：
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits)。Strata 以 [MIT 授權條款](LICENSE)開源。少數部分和每個模型
有各自的授權條款（[具體是哪些](docs/HOW_IT_WORKS.md#license)）。

## 支援 Strata

Strata 免費且開源。如果它對你有用，歡迎支援它的開發：

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="請我喝杯咖啡" height="50"></a></p>
