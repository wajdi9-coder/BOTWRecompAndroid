// Sampling profiler (Android; off unless asked). LARECOMP_PROF_SAMPLE=<from>[:<length>] samples the process's running
// threads for <length> seconds (default 15) once the app's clock (the seconds in the fps log's "t=") reaches <from>,
// then logs, for the busiest threads: how much of the time each was running, on which CPU cores, and where its samples
// fell - the game's own functions (f_<guest address>), the runtime's, the C library's, the GPU driver's. It tells what
// a thread that is too slow is doing, which the fps log's counters don't.
//
// A sampler thread (on core 0, the smallest) looks at /proc/self/task every ~5 ms; a thread in state R gets a
// SIGPROF whose handler keeps its program counter. Threads that are waiting are not signalled (a wait is not work),
// and count only in their "running" share. The report is made by the sampler thread after it stops.
#ifdef __ANDROID__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#include "gamelib.h"
#include "rt.h"

#if defined(__aarch64__)
#define UC_PC(uc) ((uint64_t)(uc)->uc_mcontext.pc)
#elif defined(__x86_64__)
#define UC_PC(uc) ((uint64_t)(uc)->uc_mcontext.gregs[REG_RIP])
#else
#define UC_PC(uc) ((uint64_t)0)
#endif

#define MAXS 300000
#define MAXT 160
#define NCPU 16
#define TOP_THREADS 4
#define TOP_ROWS 28

static uint64_t s_pc[MAXS];
static uint32_t s_tid[MAXS];
static _Atomic uint32_t s_n;
static volatile int s_on;
static double s_len = 15;

typedef struct {
    int tid;
    char comm[20];
    uint32_t rounds, running;
    uint32_t cpu[NCPU];
} Th;
static Th s_th[MAXT];
static int s_nth;

static void on_sig(int sig, siginfo_t *si, void *ucv) {
    (void)sig;
    (void)si;
    if (!s_on) return;
    uint32_t i = atomic_fetch_add_explicit(&s_n, 1, memory_order_relaxed);
    if (i >= MAXS) return;
    s_pc[i] = UC_PC((ucontext_t *)ucv);
    s_tid[i] = (uint32_t)syscall(SYS_gettid);
}

// /proc/self/task/<tid>/stat: the state letter, the core it last ran on (field 39) and its name
static int read_stat(int tid, char *state, int *cpu, char *comm, size_t cn) {
    char p[64], buf[600];
    snprintf(p, sizeof p, "/proc/self/task/%d/stat", tid);
    FILE *f = fopen(p, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    char *open = strchr(buf, '('), *close = strrchr(buf, ')');
    if (!open || !close || close < open) return 0;
    size_t k = (size_t)(close - open - 1);
    if (k >= cn) k = cn - 1;
    memcpy(comm, open + 1, k);
    comm[k] = 0;
    char *sp = NULL, *t = strtok_r(close + 1, " ", &sp);
    if (!t) return 0;
    *state = t[0];
    *cpu = -1;
    for (int field = 4; (t = strtok_r(NULL, " ", &sp)); field++)
        if (field == 39) {
            *cpu = atoi(t);
            break;
        }
    return 1;
}

static Th *find_th(int tid, const char *comm) {
    for (int i = 0; i < s_nth; i++)
        if (s_th[i].tid == tid) return &s_th[i];
    if (s_nth >= MAXT) return NULL;
    Th *t = &s_th[s_nth++];
    memset(t, 0, sizeof *t);
    t->tid = tid;
    snprintf(t->comm, sizeof t->comm, "%s", comm);
    return t;
}

// ------------------------------------------------------------------ naming the samples
typedef struct {
    uintptr_t host;
    uint64_t guest;
} FnPair;
static FnPair *s_fns;
static size_t s_nfns;

static int cmp_fn(const void *a, const void *b) {
    uintptr_t x = ((const FnPair *)a)->host, y = ((const FnPair *)b)->host;
    return x < y ? -1 : x > y;
}

static void build_fn_map(void) {
    const GameLibTables *g = gamelib_tables();
    if (!g || !g->main_count || !g->sdk_count) return;
    size_t nm = (size_t)*g->main_count, ns = (size_t)*g->sdk_count;
    s_fns = malloc((nm + ns) * sizeof *s_fns);
    if (!s_fns) return;
    for (size_t i = 0; i < nm; i++) s_fns[s_nfns++] = (FnPair){(uintptr_t)g->main_fns[i], g->main_addrs[i]};
    for (size_t i = 0; i < ns; i++) s_fns[s_nfns++] = (FnPair){(uintptr_t)g->sdk_fns[i], g->sdk_addrs[i]};
    qsort(s_fns, s_nfns, sizeof *s_fns, cmp_fn);
}

static const char *base_name(const char *path) {
    const char *s = path ? strrchr(path, '/') : NULL;
    return s ? s + 1 : (path ? path : "?");
}

// A sample's key (what is counted together) and, in name, what to print for it
static uint64_t key_of(uint64_t pc, char *name, size_t nn, char *mod, size_t mn) {
    Dl_info di;
    memset(&di, 0, sizeof di);
    int ok = dladdr((void *)(uintptr_t)pc, &di) && di.dli_fname;
    snprintf(mod, mn, "%s", ok ? base_name(di.dli_fname) : "?");
    if (ok && strstr(di.dli_fname, "botwgame") && s_nfns) {
        size_t lo = 0, hi = s_nfns;  // the last function that starts at or before pc
        while (lo + 1 < hi) {
            size_t m = (lo + hi) / 2;
            if (s_fns[m].host <= pc) lo = m;
            else hi = m;
        }
        if (s_fns[lo].host <= pc) {
            snprintf(name, nn, "game f_%llx", (unsigned long long)s_fns[lo].guest);
            return (uint64_t)s_fns[lo].host;
        }
    }
    if (ok && di.dli_sname && di.dli_saddr) {
        snprintf(name, nn, "%s:%s", mod, di.dli_sname);
        return (uint64_t)(uintptr_t)di.dli_saddr;
    }
    uint64_t off = ok ? pc - (uint64_t)(uintptr_t)di.dli_fbase : pc;
    snprintf(name, nn, "%s+0x%llx", mod, (unsigned long long)(off & ~0xffull));  // (a 256-byte bucket)
    return ok ? (uint64_t)(uintptr_t)di.dli_fbase + (off & ~0xffull) : pc;
}

typedef struct {
    uint64_t key, pc;
    uint32_t n;
} Row;
static int cmp_key(const void *a, const void *b) {
    uint64_t x = ((const Row *)a)->key, y = ((const Row *)b)->key;
    return x < y ? -1 : x > y;
}
static int cmp_n(const void *a, const void *b) {
    uint32_t x = ((const Row *)a)->n, y = ((const Row *)b)->n;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void report_thread(const Th *t, uint32_t total) {
    uint32_t cnt = 0;
    for (uint32_t i = 0; i < total; i++) cnt += s_tid[i] == (uint32_t)t->tid;
    if (!cnt) return;
    Row *rows = malloc(cnt * sizeof *rows);
    if (!rows) return;
    uint32_t nr = 0;
    char name[200], mod[64];
    typedef struct {
        char m[64];
        uint32_t n;
    } ModN;
    ModN mods[24];
    int nmod = 0;
    for (uint32_t i = 0; i < total; i++) {
        if (s_tid[i] != (uint32_t)t->tid) continue;
        rows[nr].key = key_of(s_pc[i], name, sizeof name, mod, sizeof mod);
        rows[nr].pc = s_pc[i];
        rows[nr].n = 1;
        nr++;
        int k = 0;
        while (k < nmod && strcmp(mods[k].m, mod)) k++;
        if (k == nmod && nmod < 24) {
            snprintf(mods[nmod].m, sizeof mods[nmod].m, "%s", mod);
            mods[nmod++].n = 0;
        }
        if (k < nmod) mods[k].n++;
    }
    qsort(rows, nr, sizeof *rows, cmp_key);
    uint32_t w = 0;
    for (uint32_t i = 0; i < nr; i++) {
        if (w && rows[w - 1].key == rows[i].key) rows[w - 1].n++;
        else rows[w++] = rows[i];
    }
    qsort(rows, w, sizeof *rows, cmp_n);
    LOG("[sampleprof] ---- thread %d \"%s\": %u samples\n", t->tid, t->comm, cnt);
    for (int a = 0; a < nmod; a++)  // (in the order found: a few, sorted by a simple pass)
        for (int b = a + 1; b < nmod; b++)
            if (mods[b].n > mods[a].n) {
                ModN x = mods[a];
                mods[a] = mods[b];
                mods[b] = x;
            }
    for (int a = 0; a < nmod && a < 8; a++)
        LOG("[sampleprof]   module %-28s %5.1f%%\n", mods[a].m, 100.0 * mods[a].n / cnt);
    for (uint32_t i = 0; i < w && i < TOP_ROWS; i++) {
        key_of(rows[i].pc, name, sizeof name, mod, sizeof mod);
        LOG("[sampleprof]   %5.1f%% %6u  %s\n", 100.0 * rows[i].n / cnt, rows[i].n, name);
    }
    free(rows);
}

static int cmp_th(const void *a, const void *b) {
    uint32_t x = ((const Th *)a)->running, y = ((const Th *)b)->running;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void report(void) {
    uint32_t total = atomic_load(&s_n);
    if (total > MAXS) total = MAXS;
    build_fn_map();
    LOG("[sampleprof] %u samples in %.0f s (a thread is sampled only while running; \"run\" = its share of the sampler's looks)\n",
        total, s_len);
    qsort(s_th, (size_t)s_nth, sizeof s_th[0], cmp_th);
    for (int i = 0; i < s_nth && i < 12; i++) {
        const Th *t = &s_th[i];
        if (!t->running) break;
        char cpus[160] = "";
        size_t n = 0;
        for (int c = 0; c < NCPU; c++)
            if (t->cpu[c] && n + 12 < sizeof cpus)
                n += (size_t)snprintf(cpus + n, sizeof cpus - n, " cpu%d:%u%%", c, 100 * t->cpu[c] / t->running);
        LOG("[sampleprof] thread %d \"%s\" run %.0f%%, on:%s\n", t->tid, t->comm,
            t->rounds ? 100.0 * t->running / t->rounds : 0.0, cpus);
    }
    for (int i = 0; i < s_nth && i < TOP_THREADS; i++)
        if (s_th[i].running) report_thread(&s_th[i], total);
    LOG("[sampleprof] done\n");
}

static void *sampler(void *arg) {
    (void)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    sched_setaffinity(0, sizeof set, &set);  // (the smallest core: the big ones are the game's)
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_sig;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);
    pid_t pid = getpid();
    int self = (int)syscall(SYS_gettid);
    struct timespec t0, now, nap = {0, 5 * 1000 * 1000};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    s_on = 1;
    LOG("[sampleprof] sampling for %.0f s\n", s_len);
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        if ((double)(now.tv_sec - t0.tv_sec) + (double)(now.tv_nsec - t0.tv_nsec) * 1e-9 >= s_len) break;
        DIR *d = opendir("/proc/self/task");
        if (d) {
            struct dirent *e;
            while ((e = readdir(d))) {
                int tid = atoi(e->d_name);
                char state, comm[20];
                int cpu;
                if (tid <= 0 || tid == self || !read_stat(tid, &state, &cpu, comm, sizeof comm)) continue;
                Th *t = find_th(tid, comm);
                if (!t) continue;
                t->rounds++;
                if (state != 'R') continue;
                t->running++;
                if (cpu >= 0 && cpu < NCPU) t->cpu[cpu]++;
                syscall(SYS_tgkill, pid, tid, SIGPROF);
            }
            closedir(d);
        }
        nanosleep(&nap, NULL);
    }
    s_on = 0;
    usleep(60 * 1000);  // (signals still on their way)
    report();
    return NULL;
}

// Called every few seconds by the fps log (runtime/nvn/vulkan/device.c); now: the same clock as its "t="
void sampleprof_tick(double now) {
    static int state;  // 0 not read yet, 1 waiting for its time, 2 over (or never asked)
    static double from;
    if (state == 2) return;
    if (state == 0) {
        const char *e = getenv("LARECOMP_PROF_SAMPLE");
        if (!e || !*e) {
            state = 2;
            return;
        }
        from = atof(e);
        const char *c = strchr(e, ':');
        s_len = c ? atof(c + 1) : 15;
        if (s_len < 2 || s_len > 60) s_len = 15;
        state = 1;
        LOG("[sampleprof] armed: from t=%.0f for %.0f s\n", from, s_len);
    }
    if (state == 1 && now >= from) {
        state = 2;
        pthread_t th;
        if (pthread_create(&th, NULL, sampler, NULL) == 0) pthread_detach(th);
    }
}
#endif
