// BC1 -> ASTC 4x4 (bc1_astc.h). A BC1 block is two RGB565 endpoints and a 2-bit index per texel into four colours on
// the line between them (or three, plus transparent black, when c0 <= c1). ASTC 4x4 holds that line as is: one
// partition, the endpoints as 8-bit RGB (CEM 8, LDR RGB direct), a weight per texel - no search, a few bit moves a
// block:
// - 4 colours: weights of 4 levels (0, 21, 43, 64 of 64 for BC1's 0, 1/3, 2/3, 1);
// - 3 colours, none transparent: weights of 3 levels (trits: 0, 32, 64);
// - 3 colours with transparent texels: dual plane - RGBA endpoints (CEM 12, 7-bit), the RGB weights as above and a
//   second plane for alpha (0 or 255).
// Encoding per the Khronos ASTC spec; checked against runtime/nvn/astc.c (bit-exact with ARM's decoder).
#include "bc1_astc.h"

#include <stdbool.h>
#include <pthread.h>
#include <string.h>

typedef struct { uint64_t w[2]; } Bits;
// bits [pos, pos + n) of the block, LSB first (n <= 32, pos + n <= 128): a shift and an OR on the word (or two) they
// are in - this was a loop over the bits, and the block's 100 or so bits made it most of the time of a block
static inline void put(Bits *b, int pos, int n, uint32_t v) {
    uint64_t x = (uint64_t)v & ((1ull << n) - 1);
    int lo = pos & 63, wi = pos >> 6;
    b->w[wi] |= x << lo;
    if (lo + n > 64) b->w[wi + 1] |= x >> (64 - lo);
}

// x with its 64 bits in reverse order (a single rbit on arm64)
static inline uint64_t rev64(uint64_t x) {
#if defined(__clang__)
    return __builtin_bitreverse64(x);
#else
    x = ((x >> 1) & 0x5555555555555555ull) | ((x & 0x5555555555555555ull) << 1);
    x = ((x >> 2) & 0x3333333333333333ull) | ((x & 0x3333333333333333ull) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((x & 0x0F0F0F0F0F0F0F0Full) << 4);
    return __builtin_bswap64(x);
#endif
}

// The 8-bit code of five trits: the smallest T that the spec's decoding maps to them (C.2.12), so that a short last
// group's missing high bits are zero
static uint8_t g_trit_code[243];
static void trits_init(void) {
    uint8_t seen[243] = {0};
    for (int T = 0; T < 256; T++) {
        int t[5], C;
        if ((T >> 2 & 7) == 7) {
            C = (T >> 5 & 7) << 2 | (T & 3);
            t[4] = 2, t[3] = 2;
        } else {
            C = T & 31;
            if ((T >> 5 & 3) == 3) t[4] = 2, t[3] = T >> 7 & 1;
            else t[4] = T >> 7 & 1, t[3] = T >> 5 & 3;
        }
        if ((C & 3) == 3) t[2] = 2, t[1] = C >> 4 & 1, t[0] = (C >> 3 & 1) << 1 | (C >> 2 & 1 & ~(C >> 3) & 1);
        else if ((C >> 2 & 3) == 3) t[2] = 2, t[1] = 2, t[0] = C & 3;
        else t[2] = C >> 4 & 1, t[1] = C >> 2 & 3, t[0] = (C >> 1 & 1) << 1 | (C & 1 & ~(C >> 1) & 1);
        int i = t[0] + 3 * t[1] + 9 * t[2] + 27 * t[3] + 81 * t[4];
        if (!seen[i]) seen[i] = 1, g_trit_code[i] = (uint8_t)T;
    }
}
static pthread_once_t g_trits_once = PTHREAD_ONCE_INIT;

// The weights (n values of 2 bits, or trits when trits) as the spec stores them: the integer sequence, its bits
// reversed from bit 127 down. Returns its length in bits.
// (the weights are at most 52 bits: one word, which put_weights reverses into the block's upper word)
static int trit_stream(const uint8_t *w, int n, uint64_t *out) {
    uint64_t s = 0;
    int len = 0;
    for (int g = 0; g < n; g += 5) {
        int k = n - g < 5 ? n - g : 5, idx = 0;
        for (int j = k - 1; j >= 0; j--) idx = idx * 3 + w[g + j];
        int bits = (8 * k + 4) / 5;  // (8 bits for 5 trits; a short group only the bits it needs)
        s |= (uint64_t)g_trit_code[idx] << len;
        len += bits;
    }
    *out = s;
    return len;
}
// Bit i of the stream s goes to bit 127 - i of the block: the stream reversed, in the block's upper word
static void put_stream(Bits *b, uint64_t s) { b->w[1] |= rev64(s); }

static int unq7(int q) { return q << 1 | q >> 6; }  // (a 7-bit endpoint back to 8 bits: bit replication)
static int q7(int e) {  // (the nearest 7-bit endpoint)
    int q = e >> 1;
    return q < 127 && unq7(q + 1) - e < e - unq7(q) ? q + 1 : q;
}

static void expand565(uint32_t c, int e[3]) {  // (bcdec's expansion: the colours it decoded before)
    e[0] = (int)(((c >> 11 & 31) * 527 + 23) >> 6);
    e[1] = (int)(((c >> 5 & 63) * 259 + 33) >> 6);
    e[2] = (int)(((c & 31) * 527 + 23) >> 6);
}

static void one(const uint8_t *in, uint8_t *out) {
    uint32_t c0 = in[0] | in[1] << 8, c1 = in[2] | in[3] << 8;
    uint32_t idx = in[4] | in[5] << 8 | in[6] << 16 | (uint32_t)in[7] << 24;
    int e0[3], e1[3];
    expand565(c0, e0);
    expand565(c1, e1);
    bool four = c0 > c1;
    // (3 colours: a texel with index 3 is transparent - both bits of its two set)
    bool clear = !four && ((idx & (idx >> 1)) & 0x55555555u) != 0;
    Bits b = {{0, 0}};
    uint8_t w[32];
    if (!clear) {
        // Index -> weight: 4 colours 0, 3, 1, 2 (of 0..3); 3 colours 0, 2, 1 (of 0..2). The second endpoint's sum must
        // be at least the first's (else the decoder blue-contracts): swapped with the weights mirrored if not.
        bool swap = e1[0] + e1[1] + e1[2] < e0[0] + e0[1] + e0[2];
        put(&b, 0, 11, four ? 0x42 : 0x51);  // (4x4 weights, one plane; 4 levels / 3 levels)
        put(&b, 13, 4, 8);                   // (one partition: bits 11-12 zero; CEM 8)
        const int *a = swap ? e1 : e0, *z = swap ? e0 : e1;
        for (int c = 0; c < 3; c++) put(&b, 17 + 16 * c, 8, (uint32_t)a[c]), put(&b, 25 + 16 * c, 8, (uint32_t)z[c]);
        if (four) {
            // 0, 3, 1, 2 for the indices 0..3 is, per 2-bit field, (b0 << 1) | (b0 ^ b1); mirrored (3 - q) is the
            // field's bits inverted. The weights in the order the spec stores them are this word, 2 bits a texel.
            uint32_t lo = idx & 0x55555555u, hi = (idx >> 1) & 0x55555555u;
            uint32_t ww = lo << 1 | (lo ^ hi);
            put_stream(&b, (uint64_t)(swap ? ~ww : ww));
        } else {
            static const uint8_t map3[4] = {0, 2, 1, 0};
            for (int i = 0; i < 16; i++) {
                int q = map3[idx >> 2 * i & 3];
                w[i] = (uint8_t)(swap ? 2 - q : q);
            }
            uint64_t s;
            trit_stream(w, 16, &s);
            put_stream(&b, s);
        }
    } else {
        // Dual plane: RGB on plane 1 (a transparent texel on the darker endpoint), alpha on plane 2. 52 weight bits
        // leave 57 for 8 endpoint values: 7 bits each (bit replication back to 8).
        int q0[3], q1[3], u0 = 0, u1 = 0;
        for (int c = 0; c < 3; c++) {
            q0[c] = q7(e0[c]), q1[c] = q7(e1[c]);
            u0 += unq7(q0[c]), u1 += unq7(q1[c]);
        }
        bool swap = u1 < u0;
        static const uint8_t map3[3] = {0, 2, 1};
        for (int i = 0; i < 16; i++) {
            int k = idx >> 2 * i & 3;
            int q = k == 3 ? (u0 <= u1 ? 0 : 2) : map3[k];
            w[2 * i] = (uint8_t)(swap ? 2 - q : q);
            w[2 * i + 1] = k == 3 ? 0 : 2;
        }
        put(&b, 0, 11, 0x451);  // (4x4 weights, two planes, 3 levels)
        put(&b, 13, 4, 12);     // (CEM 12: RGBA direct)
        const int *a = swap ? q1 : q0, *z = swap ? q0 : q1;
        for (int c = 0; c < 3; c++) put(&b, 17 + 14 * c, 7, (uint32_t)a[c]), put(&b, 24 + 14 * c, 7, (uint32_t)z[c]);
        put(&b, 59, 7, 0), put(&b, 66, 7, 127);  // (alpha 0 .. 255)
        uint64_t s;
        int len = trit_stream(w, 32, &s);
        put_stream(&b, s);
        put(&b, 128 - len - 2, 2, 3);  // (the second plane's component: alpha)
    }
    memcpy(out, b.w, 16);
}

void bc1_to_astc(const uint8_t *bc1, uint8_t *astc, uint64_t n) {
    pthread_once(&g_trits_once, trits_init);
    for (uint64_t i = 0; i < n; i++) one(bc1 + 8 * i, astc + 16 * i);
}
