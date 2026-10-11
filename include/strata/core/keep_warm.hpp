// include/strata/core/keep_warm.hpp - keep a lightly loaded card at its compute clocks while windows arrive.
//
// The clock governor of some cards drops a card that is busy only a few ms of every window - a layer split's short
// stage - to a low state.  Measured (GTX 1660 SUPER, driver 616.92, 32 MB reads 6 ms of every 70 ms): P3, SM 360 MHz,
// memory 5000 MHz instead of P2, 1530-1935 / 6801 - 122 GB/s instead of 303, and the stage's small kernels 2-5x
// slower.  The same card with one sleeping one-thread kernel beside the work stays at P2, 1935 / 6801.
//
// keep_warm_poke(dev), called for each window the card runs, keeps such a kernel on the card (a low-priority stream
// of its own) until KEEP_MS after the last poke; then the card is left alone and its governor idles it as before.
// It changes no value: the kernel touches no memory.
//
//     STRATA_KEEP_WARM unset: GTX 16xx cards only (Turing without tensor cores, where it was measured)
//                      =0     off
//                      =1     every card that runs windows
#pragma once

namespace strata::core {

void keep_warm_poke(int device);

}  // namespace strata::core
