/* qp_test.c - checks of the QP solver (the SOCP developer, 22 October 2026): (1) the interior-point and the active-set
 * method on random strictly convex and semidefinite QPs of every constraint kind agree on the objective; (2) the warm
 * start of the solver object: a depth-first branch and bound on random mixed-integer QPs (integer variables in boxes),
 * each node re-solved from its parent's active set (qp_set_var_bounds, qp_resolve), against cold solves of the same
 * nodes: the same optimal values, and the iterations of both counted.  make qptest */
#include "../qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static unsigned long long rs = 88172645463325252ULL;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-300, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

/* a random QP: H = M M' (+ diag when strict), m rows of density 0.5 with ranges, equalities, one-sided rows, bounds */
static void rand_qp(QpProb *P, int n, int m, int strict) {
    memset(P, 0, sizeof(*P)); P->n = n; P->m = m;
    double *M = malloc(sizeof(double) * n * n), *Hd = calloc((size_t)n * n, sizeof(double));
    const int r = strict ? n : n / 2;
    for (int i = 0; i < n * n; i++) M[i] = nrand();
    for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) { double s = 0; for (int k = 0; k < r; k++) s += M[i + k * n] * M[j + k * n]; Hd[i + j * n] = s + (i == j && strict ? 0.1 : 0); }
    P->Hp = malloc(sizeof(int) * (n + 1)); P->Hi = malloc(sizeof(int) * n * n); P->Hx = malloc(sizeof(double) * n * n);
    int w = 0; for (int j = 0; j < n; j++) { P->Hp[j] = w; for (int i = j; i < n; i++) { P->Hi[w] = i; P->Hx[w] = Hd[i + j * n]; w++; } } P->Hp[n] = w;
    P->Ap = malloc(sizeof(int) * (n + 1)); P->Ai = malloc(sizeof(int) * n * m + 1); P->Ax = malloc(sizeof(double) * n * m + 1);
    w = 0; for (int j = 0; j < n; j++) { P->Ap[j] = w; for (int i = 0; i < m; i++) if (urand() < 0.5) { P->Ai[w] = i; P->Ax[w] = nrand(); w++; } } P->Ap[n] = w;
    P->c = malloc(sizeof(double) * n); for (int j = 0; j < n; j++) P->c[j] = nrand() * 3;
    /* a point x0 inside the bounds: rows built around A x0 so that the QP is feasible */
    double *x0 = malloc(sizeof(double) * n); for (int j = 0; j < n; j++) x0[j] = nrand();
    double *ax = calloc(m + 1, sizeof(double)); for (int j = 0; j < n; j++) for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) ax[P->Ai[p]] += P->Ax[p] * x0[j];
    P->rl = malloc(sizeof(double) * (m + 1)); P->ru = malloc(sizeof(double) * (m + 1));
    for (int i = 0; i < m; i++) {
        const int t = i % 4;
        if (t == 0) { P->rl[i] = P->ru[i] = ax[i]; }
        else if (t == 1) { P->rl[i] = ax[i] - urand(); P->ru[i] = ax[i] + urand(); }
        else if (t == 2) { P->rl[i] = -QP_INF; P->ru[i] = ax[i] + urand(); }
        else { P->rl[i] = ax[i] - urand(); P->ru[i] = QP_INF; }
    }
    P->xl = malloc(sizeof(double) * n); P->xu = malloc(sizeof(double) * n);
    for (int j = 0; j < n; j++) {
        const int t = j % 3;
        P->xl[j] = t == 2 ? -QP_INF : x0[j] - 1 - urand(); P->xu[j] = t == 1 ? QP_INF : x0[j] + 1 + urand();
        if (!strict && t == 2) { P->xl[j] = x0[j] - 2; P->xu[j] = x0[j] + 2; }     /* bounded: a semidefinite QP stays bounded */
    }
    free(M); free(Hd); free(x0); free(ax);
}

/* P with one more equality row, 2 (row 0) - 3 (row 4), both equalities of rand_qp; its right-hand side the same
 * combination plus off (0: a dependent row the presolve drops; else an inconsistent one) */
static void add_dep_row(QpProb *P, double off) {
    const int n = P->n, m = P->m;
    int *Ap = malloc(sizeof(int) * (n + 1)), *Ai = malloc(sizeof(int) * (P->Ap[n] + n + 1)); double *Ax = malloc(sizeof(double) * (P->Ap[n] + n + 1));
    int w = 0;
    for (int j = 0; j < n; j++) {
        Ap[j] = w; double v = 0;
        for (int p = P->Ap[j]; p < P->Ap[j + 1]; p++) { Ai[w] = P->Ai[p]; Ax[w] = P->Ax[p]; w++; if (P->Ai[p] == 0) v += 2 * P->Ax[p]; if (P->Ai[p] == 4) v -= 3 * P->Ax[p]; }
        if (v != 0) { Ai[w] = m; Ax[w] = v; w++; }
    }
    Ap[n] = w;
    free(P->Ap); free(P->Ai); free(P->Ax); P->Ap = Ap; P->Ai = Ai; P->Ax = Ax;
    P->rl = realloc(P->rl, sizeof(double) * (m + 2)); P->ru = realloc(P->ru, sizeof(double) * (m + 2));
    P->rl[m] = P->ru[m] = 2 * P->rl[0] - 3 * P->rl[4] + off; P->m = m + 1;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL " __VA_ARGS__); printf("\n"); } } while (0)

/* depth-first branch and bound on the variables 0..ni-1 (integers), the nodes solved warm (from the parent) or cold */
typedef struct { long nodes, iters; double best; } BB;
static void bb(QpSolver *Q, const QpProb *P0, int ni, int warm, BB *st, int depth) {
    QpRes R; memset(&R, 0, sizeof(R));
    QpSolver *C = NULL;
    if (warm) qp_resolve(Q, &R);
    else { QpOpts o; qp_default_opts(&o); o.verbose = 0; o.method = QP_METHOD_ACTIVESET; C = qp_create(qp_problem(Q), &o); qp_resolve(C, &R); }
    st->nodes++; st->iters += R.iter;
    const int ok = R.status == QP_OPTIMAL || R.status == QP_REDUCED;
    if (!ok || R.obj >= st->best - 1e-9) { qp_res_free(&R); if (C) qp_destroy(C); return; }
    int jb = -1; double frac = 0;
    for (int j = 0; j < ni; j++) { const double f = fabs(R.x[j] - floor(R.x[j] + 0.5)); if (f > 1e-6 && f > frac) { frac = f; jb = j; } }
    if (jb < 0) { st->best = R.obj; qp_res_free(&R); if (C) qp_destroy(C); return; }
    const QpProb *P = qp_problem(Q);
    const double lo = P->xl[jb], hi = P->xu[jb], v = R.x[jb];
    qp_res_free(&R); if (C) qp_destroy(C);
    if (depth > 60) return;
    qp_set_var_bounds(Q, jb, lo, floor(v)); bb(Q, P0, ni, warm, st, depth + 1);
    qp_set_var_bounds(Q, jb, ceil(v), hi);  bb(Q, P0, ni, warm, st, depth + 1);
    qp_set_var_bounds(Q, jb, lo, hi);
}

int main(void) {
    QpOpts o; qp_default_opts(&o); o.verbose = 0;
    printf("1. the two methods on random QPs\n");
    for (int t = 0; t < 40; t++) {
        const int n = 5 + t % 25, m = 3 + (t * 7) % 20, strict = t % 3 != 2;
        QpProb P; rand_qp(&P, n, m, strict);
        QpRes A, B; o.method = QP_METHOD_IPM; qp_solve(&P, &o, &A); o.method = QP_METHOD_ACTIVESET; qp_solve(&P, &o, &B);
        CHECK(A.status == QP_OPTIMAL, "QP %d (n %d, m %d, %s): interior point %s", t, n, m, strict ? "definite" : "semidefinite", qp_status_name(A.status));
        CHECK(B.status == QP_OPTIMAL, "QP %d (n %d, m %d, %s): active set %s", t, n, m, strict ? "definite" : "semidefinite", qp_status_name(B.status));
        if (A.status == QP_OPTIMAL && B.status == QP_OPTIMAL) CHECK(fabs(A.obj - B.obj) <= 1e-6 * (1 + fabs(A.obj)), "QP %d: objectives %.10g and %.10g", t, A.obj, B.obj);
        qp_res_free(&A); qp_res_free(&B); qp_prob_free(&P);
    }
    printf("2. branch and bound on random mixed-integer QPs: warm against cold node solves\n");
    long tw = 0, tc = 0;
    for (int t = 0; t < (getenv("QPTEST_QUICK") ? 0 : 12); t++) {
        const int n = 10 + t * 3, m = 4 + t, ni = n / 2;
        QpProb P; rand_qp(&P, n, m, 1);
        for (int j = 0; j < ni; j++) { if (P.xl[j] <= -QP_INF) P.xl[j] = -5; if (P.xu[j] >= QP_INF) P.xu[j] = 5; }
        o.method = QP_METHOD_ACTIVESET;
        QpSolver *Qw = qp_create(&P, &o), *Qc = qp_create(&P, &o);
        BB w = { 0, 0, INFINITY }, c = { 0, 0, INFINITY };
        bb(Qw, &P, ni, 1, &w, 0); bb(Qc, &P, ni, 0, &c, 0);
        CHECK(fabs(w.best - c.best) <= 1e-6 * (1 + fabs(c.best)) || (!isfinite(w.best) && !isfinite(c.best)), "MIQP %d: optimal values %.10g (warm) and %.10g (cold)", t, w.best, c.best);
        printf("  MIQP %2d (n %2d, %2d integers): %4ld nodes, iterations %6ld warm, %6ld cold, value %.8g\n", t, n, ni, w.nodes, w.iters, c.iters, w.best);
        tw += w.iters; tc += c.iters;
        qp_destroy(Qw); qp_destroy(Qc); qp_prob_free(&P);
    }
    printf("  all nodes: %ld iterations warm, %ld cold (%.2f)\n", tw, tc, (double)tw / (double)(tc ? tc : 1));
    printf("3. infeasible and unbounded QPs (the interior-point method's certificates)\n");
    for (int t = 0; t < 10; t++) {
        const int n = 6 + t, m = 4 + t % 5, strict = t % 2;
        QpProb P; rand_qp(&P, n, m, strict);
        int want;
        if (t % 3 != 2) {          /* infeasible: row 1 (ranged) moved above row 1's bounds by a copy of it */
            /* rows 1 and 2 forced apart: row 2 := row 1 as a column pattern is costly; instead bound x_0 against row 0's
             * equality: x_0 fixed outside the box the other rows allow -- simplest: two bounds that contradict a row */
            for (int j = 0; j < n; j++) { P.xl[j] = 0; P.xu[j] = 1; }
            /* the row 0 (an equality a'x = b) with b beyond sum |a_j|: no x in [0,1]^n meets it */
            double sa = 0; for (int j = 0; j < n; j++) for (int p = P.Ap[j]; p < P.Ap[j + 1]; p++) if (P.Ai[p] == 0) sa += fabs(P.Ax[p]);
            P.rl[0] = P.ru[0] = sa + 1.0;
            want = QP_PRIMAL_INFEASIBLE;
        } else {                   /* unbounded: H = 0 on a free variable without rows, with a cost */
            for (int j = 0; j < n; j++) { P.xl[j] = -1; P.xu[j] = 1; }
            P.xl[0] = -QP_INF; P.xu[0] = QP_INF; P.c[0] = 1.0;
            for (int p = P.Hp[0]; p < P.Hp[n]; p++) if (P.Hi[p] == 0) P.Hx[p] = 0;      /* row and column 0 of H: zero */
            for (int p = P.Hp[0]; p < P.Hp[1]; p++) P.Hx[p] = 0;
            for (int j = 0; j < n; j++) for (int p = P.Ap[j]; p < P.Ap[j + 1]; p++) if (j == 0) P.Ax[p] = 0;
            /* (24 October 2026) the rows rebuilt around x = 0, inside the new bounds: rand_qp's rows were made around a
             * point that the bounds [-1, 1] and column 0's removal leave out, and two of these QPs were primal infeasible
             * as well (Gurobi: infeasible), which the Farkas proof now says first */
            for (int i = 0; i < m; i++) {
                const int tt = i % 4;
                if (tt == 0) P.rl[i] = P.ru[i] = 0;
                else if (tt == 1) { P.rl[i] = -urand(); P.ru[i] = urand(); }
                else if (tt == 2) { P.rl[i] = -QP_INF; P.ru[i] = urand(); }
                else { P.rl[i] = -urand(); P.ru[i] = QP_INF; }
            }
            want = QP_DUAL_INFEASIBLE;
        }
        QpRes A; o.method = QP_METHOD_IPM; qp_solve(&P, &o, &A);
        CHECK(A.status == want, "QP %d (n %d, m %d): %s, expected %s (%d iterations)", t, n, m, qp_status_name(A.status), qp_status_name(want), A.iter);
        /* (24 October 2026) every variable bounded: the multipliers prove the infeasibility within a few iterations
         * (qp_farkas_proof), not only once they have grown beyond 1e6 */
        if (want == QP_PRIMAL_INFEASIBLE) CHECK(A.iter <= 5, "QP %d: primal infeasibility proven after %d iterations (at most 5 expected)", t, A.iter);
        qp_res_free(&A); qp_prob_free(&P);
    }
    printf("4. a linearly dependent equality row (dropped, the same optimum) and an inconsistent one (primal infeasible)\n");
    for (int t = 0; t < 6; t++) {
        const int n = 8 + t, m = 6 + t % 3, strict = t % 2;
        QpProb P; rand_qp(&P, n, m, strict);
        QpRes A0, A1; o.method = QP_METHOD_IPM; qp_solve(&P, &o, &A0);
        add_dep_row(&P, 0.0); qp_solve(&P, &o, &A1);
        CHECK(A0.status == QP_OPTIMAL && A1.status == QP_OPTIMAL && fabs(A0.obj - A1.obj) <= 1e-6 * (1 + fabs(A0.obj)),
              "QP %d with a dependent row: %s %.10g, without: %s %.10g", t, qp_status_name(A1.status), A1.obj, qp_status_name(A0.status), A0.obj);
        qp_res_free(&A1); qp_prob_free(&P);
        rand_qp(&P, n, m, strict); add_dep_row(&P, 1.0); qp_solve(&P, &o, &A1);
        CHECK(A1.status == QP_PRIMAL_INFEASIBLE, "QP %d with an inconsistent dependent row: %s (%d iterations)", t, qp_status_name(A1.status), A1.iter);
        qp_res_free(&A0); qp_res_free(&A1); qp_prob_free(&P);
    }
    printf("5. QPs that are not convex: refused (INPUT ERROR) by both methods and by the solver object\n");
    {   /* min x1 x2 over the box [-1, 1]^2 with x1 + x2 <= 1.5: a product without square terms (the minimum -1 at (1, -1),
         * where the interior-point method reported OPTIMAL at 0 until 24 October 2026); then min -x1^2/2 + x2^2/2 */
        int Hp[3] = { 0, 1, 1 }, Hi[1] = { 1 }; double Hx[1] = { 1.0 };
        int Ap[3] = { 0, 1, 2 }, Ai[2] = { 0, 0 }; double Ax[2] = { 1, 1 }, c[2] = { 0, 0 }, rl[1] = { -QP_INF }, ru[1] = { 1.5 }, xl[2] = { -1, -1 }, xu[2] = { 1, 1 };
        QpProb P; memset(&P, 0, sizeof P); P.n = 2; P.m = 1; P.Hp = Hp; P.Hi = Hi; P.Hx = Hx; P.Ap = Ap; P.Ai = Ai; P.Ax = Ax; P.c = c; P.rl = rl; P.ru = ru; P.xl = xl; P.xu = xu;
        for (int v = 0; v < 2; v++) {
            int Hp2[3] = { 0, 1, 2 }, Hi2[2] = { 0, 1 }; double Hx2[2] = { -1.0, 1.0 };
            if (v == 1) { P.Hp = Hp2; P.Hi = Hi2; P.Hx = Hx2; }
            for (int meth = QP_METHOD_AUTO; meth <= QP_METHOD_ACTIVESET; meth++) {
                QpRes A; o.method = meth; qp_solve(&P, &o, &A);
                CHECK(A.status == QP_INPUT, "not convex (%s), method %d: %s", v ? "a negative square term" : "a product without squares", meth, qp_status_name(A.status));
                qp_res_free(&A);
            }
            o.method = QP_METHOD_ACTIVESET;
            QpSolver *Q = qp_create(&P, &o); QpRes A; memset(&A, 0, sizeof A);
            qp_resolve(Q, &A);
            CHECK(A.status == QP_INPUT, "not convex (%s), the solver object: %s", v ? "a negative square term" : "a product without squares", qp_status_name(A.status));
            qp_res_free(&A); qp_destroy(Q);
        }
    }
    printf("6. flows with quadratic arc costs on random graphs: the normal equations by conjugate gradients (BRISK_QPCG=2) against\n   their factor\n");
    for (int t = 0; t < 4; t++) {
        /* nn nodes, a path through them and 3 nn random arcs; supplies summing to zero; min sum c_e f_e + h_e f_e^2 / 2,
         * -u_e <= f_e <= u_e; the node balances as equality rows (one of them dependent) */
        const int nn = 30 + 40 * t, na = nn - 1 + 3 * nn;
        int *tl = malloc(sizeof(int) * na), *hd = malloc(sizeof(int) * na);
        for (int e = 0; e < nn - 1; e++) { tl[e] = e; hd[e] = e + 1; }
        for (int e = nn - 1; e < na; e++) { tl[e] = (int)(urand() * nn) % nn; do hd[e] = (int)(urand() * nn) % nn; while (hd[e] == tl[e]); }
        QpProb P; memset(&P, 0, sizeof P); P.n = na; P.m = nn;
        P.Ap = malloc(sizeof(int) * (na + 1)); P.Ai = malloc(sizeof(int) * 2 * na); P.Ax = malloc(sizeof(double) * 2 * na);
        P.Hp = malloc(sizeof(int) * (na + 1)); P.Hi = malloc(sizeof(int) * na); P.Hx = malloc(sizeof(double) * na);
        P.c = malloc(sizeof(double) * na); P.xl = malloc(sizeof(double) * na); P.xu = malloc(sizeof(double) * na);
        P.rl = malloc(sizeof(double) * nn); P.ru = malloc(sizeof(double) * nn);
        for (int e = 0; e < na; e++) {
            const int lo = tl[e] < hd[e] ? tl[e] : hd[e], hi = tl[e] < hd[e] ? hd[e] : tl[e];
            P.Ap[e] = 2 * e; P.Ai[2 * e] = lo; P.Ai[2 * e + 1] = hi; P.Ax[2 * e] = lo == tl[e] ? -1.0 : 1.0; P.Ax[2 * e + 1] = -P.Ax[2 * e];
            P.Hp[e] = e; P.Hi[e] = e; P.Hx[e] = 0.1 + urand(); P.c[e] = urand(); P.xu[e] = 2 + 3 * urand(); P.xl[e] = -P.xu[e];
        }
        P.Ap[na] = 2 * na; P.Hp[na] = na;
        double sb = 0; for (int i = 0; i < nn; i++) { P.rl[i] = P.ru[i] = (i < nn - 1) ? urand() - 0.5 : 0; sb += P.rl[i]; }
        P.rl[nn - 1] = P.ru[nn - 1] = -sb;
        QpRes A, B; o.method = QP_METHOD_IPM;
        unsetenv("BRISK_QPCG"); qp_solve(&P, &o, &A);
        setenv("BRISK_QPCG", "2", 1); qp_solve(&P, &o, &B); unsetenv("BRISK_QPCG");
        CHECK(A.status == QP_OPTIMAL && B.status == QP_OPTIMAL && fabs(A.obj - B.obj) <= 1e-7 * (1 + fabs(A.obj)),
              "flow %d (%d nodes, %d arcs): factor %s %.10g, conjugate gradients %s %.10g", t, nn, na, qp_status_name(A.status), A.obj, qp_status_name(B.status), B.obj);
        qp_res_free(&A); qp_res_free(&B); qp_prob_free(&P); free(tl); free(hd);
    }
    printf("6. receding-horizon MPC QPs in the sparse form (the dynamics as equality rows): the QP model with its equality rows\n"
           "   eliminated against the model without and the interior-point method, over moving initial states\n");
    for (int t = 0; t < 6; t++) {
        const int nx = 2 + t % 3, nu = 1 + t % 2, N = 6 + 3 * t, n = (N + 1) * nx + N * nu, m = (N + 1) * nx;
        double *Ad = malloc(sizeof(double) * nx * nx), *Bd = malloc(sizeof(double) * nx * nu);
        for (int i = 0; i < nx * nx; i++) Ad[i] = (i % (nx + 1) == 0 ? 1.0 : 0.0) + 0.2 * nrand();
        for (int i = 0; i < nx * nu; i++) Bd[i] = nrand();
        QpProb P; memset(&P, 0, sizeof P); P.n = n; P.m = m;
        /* variables: x_0..x_N (nx each), then u_0..u_{N-1}; rows: x_0 = x_init, x_{s+1} - A x_s - B u_s = 0 */
        P.Ap = malloc(sizeof(int) * (n + 1)); P.Ai = malloc(sizeof(int) * (n * (nx + 1) + 1)); P.Ax = malloc(sizeof(double) * (n * (nx + 1) + 1));
        int w = 0;
        for (int j = 0; j < n; j++) {
            P.Ap[j] = w;
            if (j < (N + 1) * nx) {
                const int s0 = j / nx, a = j % nx;
                P.Ai[w] = j; P.Ax[w] = 1.0; w++;                                  /* its own row */
                if (s0 < N) for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Ad[i + a * nx]; w++; }
            } else {
                const int s0 = (j - (N + 1) * nx) / nu, b = (j - (N + 1) * nx) % nu;
                for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Bd[i + b * nx]; w++; }
            }
        }
        P.Ap[n] = w;
        P.Hp = malloc(sizeof(int) * (n + 1)); P.Hi = malloc(sizeof(int) * n); P.Hx = malloc(sizeof(double) * n);
        P.c = calloc(n, sizeof(double)); P.xl = malloc(sizeof(double) * n); P.xu = malloc(sizeof(double) * n);
        for (int j = 0; j < n; j++) { P.Hp[j] = j; P.Hi[j] = j; P.Hx[j] = j < (N + 1) * nx ? 1.0 : 0.1; P.xl[j] = j < (N + 1) * nx ? -5 : -1; P.xu[j] = -P.xl[j]; }
        P.Hp[n] = n;
        P.rl = calloc(m, sizeof(double)); P.ru = calloc(m, sizeof(double));
        double *x0 = malloc(sizeof(double) * nx); for (int i = 0; i < nx; i++) x0[i] = 2 * nrand();
        for (int i = 0; i < nx; i++) P.rl[i] = P.ru[i] = x0[i];
        o.method = QP_METHOD_ACTIVESET;
        QpSolver *Qe = qp_create(&P, &o);
        setenv("BRISK_QPASELIM", "0", 1); QpSolver *Qf = qp_create(&P, &o); unsetenv("BRISK_QPASELIM");
        for (int step = 0; step < 8; step++) {
            for (int i = 0; i < nx; i++) { x0[i] = 0.7 * x0[i] + 0.5 * nrand(); qp_set_row_bounds(Qe, i, x0[i], x0[i]); qp_set_row_bounds(Qf, i, x0[i], x0[i]); P.rl[i] = P.ru[i] = x0[i]; }
            QpRes A, B, C; memset(&A, 0, sizeof A); memset(&B, 0, sizeof B);
            qp_resolve(Qe, &A); qp_resolve(Qf, &B);
            QpOpts oi = o; oi.method = QP_METHOD_IPM; qp_solve(&P, &oi, &C);
            {   /* the single solve by the active-set method (qp_solve: through a model that eliminates the equalities) */
                QpRes D; QpOpts oa = o; oa.method = QP_METHOD_ACTIVESET; qp_solve(&P, &oa, &D);
                CHECK(D.status == C.status && (C.status != QP_OPTIMAL || fabs(D.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj))),
                      "MPC %d, step %d: the single active-set solve %s %.10g, interior point %s %.10g", t, step, qp_status_name(D.status), D.obj, qp_status_name(C.status), C.obj);
                qp_res_free(&D); }
            CHECK(A.status == C.status && B.status == C.status && (C.status != QP_OPTIMAL || (fabs(A.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj)) && fabs(B.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj)))),
                  "MPC %d (n %d, %d equalities), step %d: eliminated %s %.10g, full %s %.10g, interior point %s %.10g", t, n, m, step,
                  qp_status_name(A.status), A.obj, qp_status_name(B.status), B.obj, qp_status_name(C.status), C.obj);
            qp_res_free(&A); qp_res_free(&B); qp_res_free(&C);
        }
        qp_destroy(Qe); qp_destroy(Qf); qp_prob_free(&P); free(Ad); free(Bd); free(x0);
    }
    printf("7. MPC QPs with wider stages: the interior-point method's stage-wise factorizations against its general path; a\n"
           "   persistent model's interior-point re-solves (the levels, the scaling and the last solution kept) and its own\n"
           "   choice of method over a stream, against fresh solves\n");
    for (int t = 0; t < 4; t++) {
        extern void qp_ipm_stage_mode(int mode);
        const int nx = 9 + 2 * t, nu = 3 + t % 2, N = 10 + 4 * t, n = (N + 1) * nx + N * nu, m = (N + 1) * nx;
        double *Ad = malloc(sizeof(double) * nx * nx), *Bd = malloc(sizeof(double) * nx * nu);
        for (int i = 0; i < nx * nx; i++) Ad[i] = (i % (nx + 1) == 0 ? 0.7 : 0.0) + 0.2 / sqrt((double)nx) * nrand();      /* (stable: the states' elimination stays accurate over the horizon) */
        for (int i = 0; i < nx * nu; i++) Bd[i] = nrand();
        QpProb P; memset(&P, 0, sizeof P); P.n = n; P.m = m;
        P.Ap = malloc(sizeof(int) * (n + 1)); P.Ai = malloc(sizeof(int) * (n * (nx + 1) + 1)); P.Ax = malloc(sizeof(double) * (n * (nx + 1) + 1));
        int w = 0;
        for (int j = 0; j < n; j++) {
            P.Ap[j] = w;
            if (j < (N + 1) * nx) {
                const int s0 = j / nx, a = j % nx;
                P.Ai[w] = j; P.Ax[w] = 1.0; w++;
                if (s0 < N) for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Ad[i + a * nx]; w++; }
            } else {
                const int s0 = (j - (N + 1) * nx) / nu, b = (j - (N + 1) * nx) % nu;
                for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Bd[i + b * nx]; w++; }
            }
        }
        P.Ap[n] = w;
        P.Hp = malloc(sizeof(int) * (n + 1)); P.Hi = malloc(sizeof(int) * n); P.Hx = malloc(sizeof(double) * n);
        P.c = calloc(n, sizeof(double)); P.xl = malloc(sizeof(double) * n); P.xu = malloc(sizeof(double) * n);
        for (int j = 0; j < n; j++) { P.Hp[j] = j; P.Hi[j] = j; P.Hx[j] = j < (N + 1) * nx ? 1.0 + (j % 3) : 0.1; P.xl[j] = j < (N + 1) * nx ? -4 : -0.6; P.xu[j] = -P.xl[j]; P.c[j] = 0.3 * nrand(); }
        P.Hp[n] = n;
        P.rl = calloc(m, sizeof(double)); P.ru = calloc(m, sizeof(double));
        double *x0 = malloc(sizeof(double) * nx); for (int i = 0; i < nx; i++) x0[i] = nrand();
        for (int i = 0; i < nx; i++) P.rl[i] = P.ru[i] = x0[i];
        QpOpts oi = o; oi.method = QP_METHOD_IPM;
        /* one solve by each factorization: the general path (0), the normal equations by stage (3), the augmented system by stage (2) */
        QpRes G, S3, S2; qp_ipm_stage_mode(0); qp_solve(&P, &oi, &G); qp_ipm_stage_mode(3); qp_solve(&P, &oi, &S3); qp_ipm_stage_mode(2); qp_solve(&P, &oi, &S2); qp_ipm_stage_mode(-1);
        CHECK(G.status == QP_OPTIMAL && S3.status == QP_OPTIMAL && S2.status == QP_OPTIMAL && fabs(S3.obj - G.obj) <= 1e-7 * (1 + fabs(G.obj)) && fabs(S2.obj - G.obj) <= 1e-7 * (1 + fabs(G.obj)),
              "MPC %d (n %d, %d states): the general path %s %.10g, the normal equations by stage %s %.10g, the augmented system by stage %s %.10g", t, n, nx,
              qp_status_name(G.status), G.obj, qp_status_name(S3.status), S3.obj, qp_status_name(S2.status), S2.obj);
        qp_res_free(&G); qp_res_free(&S3); qp_res_free(&S2);
        /* a stream: the model with the interior-point method, the model that chooses, and a fresh solve by the general path */
        QpOpts oa = o; oa.method = QP_METHOD_AUTO;
        QpSolver *Qi = qp_create(&P, &oi), *Qa = qp_create(&P, &oa);
        int nas = 0;
        for (int step = 0; step < 10; step++) {
            const double mv = step < 6 ? 0.02 : 0.3;            /* small moves of the initial state (a stream the active-set method serves), then large ones (a solve over its budget is handed to the interior-point method) */
            for (int i = 0; i < nx; i++) { x0[i] = 0.8 * x0[i] + mv * nrand(); qp_set_row_bounds(Qi, i, x0[i], x0[i]); qp_set_row_bounds(Qa, i, x0[i], x0[i]); P.rl[i] = P.ru[i] = x0[i]; }
            QpRes A, B, C; memset(&A, 0, sizeof A); memset(&B, 0, sizeof B);
            qp_resolve(Qi, &A); qp_resolve(Qa, &B);
            qp_ipm_stage_mode(0); qp_solve(&P, &oi, &C); qp_ipm_stage_mode(-1);
            nas += B.method == QP_METHOD_ACTIVESET;
            CHECK(A.status == C.status && B.status == C.status && (C.status != QP_OPTIMAL || (fabs(A.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj)) && fabs(B.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj)))),
                  "MPC %d, step %d: the interior-point model %s %.10g (%d iterations), the choosing model %s %.10g, a fresh solve %s %.10g (%d iterations)", t, step,
                  qp_status_name(A.status), A.obj, A.iter, qp_status_name(B.status), B.obj, qp_status_name(C.status), C.obj, C.iter);
            qp_res_free(&A); qp_res_free(&B); qp_res_free(&C);
        }
        CHECK(nas >= 6, "MPC %d (n %d, %d left after the elimination): the choosing model took the active-set method on %d of 10 solves (at least the 6 with small moves expected)", t, n, N * nu, nas);
        /* (15 November 2026) the interior-point model's kept reduction: steps that keep the fixed columns and the rows' kinds
         * (the kept reduction is used) and steps that change them (a column fixed and freed again, an equality row made a
         * range and back, bounds that contradict each other), each against a fresh solve */
        for (int step = 0; step < 9; step++) {
            const int ju = (N + 1) * nx + 1 % (N * nu);         /* an input */
            if (step == 1 || step == 2) { P.xl[ju] = P.xu[ju] = 0.1; qp_set_var_bounds(Qi, ju, 0.1, 0.1); }
            if (step == 3) { P.xl[ju] = -0.6; P.xu[ju] = 0.6; qp_set_var_bounds(Qi, ju, -0.6, 0.6); }
            if (step == 4 || step == 5) { P.rl[0] = x0[0] - 0.1; P.ru[0] = x0[0] + 0.1; qp_set_row_bounds(Qi, 0, P.rl[0], P.ru[0]); }
            if (step == 6) { P.rl[0] = P.ru[0] = x0[0]; qp_set_row_bounds(Qi, 0, x0[0], x0[0]); }
            if (step == 7) { P.xl[ju] = 0.5; P.xu[ju] = 0.3; qp_set_var_bounds(Qi, ju, 0.5, 0.3); }
            if (step == 8) { P.xl[ju] = -0.6; P.xu[ju] = 0.6; qp_set_var_bounds(Qi, ju, -0.6, 0.6); }
            for (int i = 1; i < nx; i++) { x0[i] = 0.8 * x0[i] + 0.05 * nrand(); qp_set_row_bounds(Qi, i, x0[i], x0[i]); P.rl[i] = P.ru[i] = x0[i]; }
            QpRes A, C; memset(&A, 0, sizeof A);
            qp_resolve(Qi, &A);
            qp_ipm_stage_mode(0); qp_solve(&P, &oi, &C); qp_ipm_stage_mode(-1);
            CHECK(A.status == C.status && (step == 7 ? A.status == QP_PRIMAL_INFEASIBLE : A.status == QP_OPTIMAL && fabs(A.obj - C.obj) <= 1e-6 * (1 + fabs(C.obj))),
                  "MPC %d, the kept reduction, step %d: the interior-point model %s %.10g, a fresh solve %s %.10g", t, step,
                  qp_status_name(A.status), A.obj, qp_status_name(C.status), C.obj);
            qp_res_free(&A); qp_res_free(&C);
        }
        qp_destroy(Qi); qp_destroy(Qa); qp_prob_free(&P); free(Ad); free(Bd); free(x0);
    }
    printf("8. An MPC QP with stages of 176 states: the stage-wise factorization's chunks of a wide level and its two ends on\n"
           "   the threads against the general path, and the same bits on one thread and on all\n");
    {
        extern void qp_ipm_stage_mode(int mode);
        const int nx = 176, nu = 20, N = 9, n = (N + 1) * nx + N * nu, m = (N + 1) * nx;
        double *Ad = malloc(sizeof(double) * nx * nx), *Bd = malloc(sizeof(double) * nx * nu);
        for (int i = 0; i < nx * nx; i++) Ad[i] = (i % (nx + 1) == 0 ? 0.7 : 0.0) + 0.2 / sqrt((double)nx) * nrand();
        for (int i = 0; i < nx * nu; i++) Bd[i] = nrand();
        QpProb P; memset(&P, 0, sizeof P); P.n = n; P.m = m;
        P.Ap = malloc(sizeof(int) * (n + 1)); P.Ai = malloc(sizeof(int) * ((size_t)n * (nx + 1) + 1)); P.Ax = malloc(sizeof(double) * ((size_t)n * (nx + 1) + 1));
        int w = 0;
        for (int j = 0; j < n; j++) {
            P.Ap[j] = w;
            if (j < (N + 1) * nx) {
                const int s0 = j / nx, a = j % nx;
                P.Ai[w] = j; P.Ax[w] = 1.0; w++;
                if (s0 < N) for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Ad[i + a * nx]; w++; }
            } else {
                const int s0 = (j - (N + 1) * nx) / nu, b = (j - (N + 1) * nx) % nu;
                for (int i = 0; i < nx; i++) { P.Ai[w] = (s0 + 1) * nx + i; P.Ax[w] = -Bd[i + b * nx]; w++; }
            }
        }
        P.Ap[n] = w;
        P.Hp = malloc(sizeof(int) * (n + 1)); P.Hi = malloc(sizeof(int) * n); P.Hx = malloc(sizeof(double) * n);
        P.c = calloc(n, sizeof(double)); P.xl = malloc(sizeof(double) * n); P.xu = malloc(sizeof(double) * n);
        for (int j = 0; j < n; j++) { P.Hp[j] = j; P.Hi[j] = j; P.Hx[j] = j < (N + 1) * nx ? 1.0 + (j % 3) : 0.1; P.xl[j] = j < (N + 1) * nx ? -4 : -0.6; P.xu[j] = -P.xl[j]; P.c[j] = 0.3 * nrand(); }
        P.Hp[n] = n;
        P.rl = calloc(m, sizeof(double)); P.ru = calloc(m, sizeof(double));
        for (int i = 0; i < nx; i++) P.rl[i] = P.ru[i] = 0.3 * nrand();
        QpOpts oi = o; oi.method = QP_METHOD_IPM;
        QpRes G, S1, Sn; int nt = 1;
        qp_ipm_stage_mode(0); qp_solve(&P, &oi, &G);
        qp_ipm_stage_mode(3);
#ifdef _OPENMP
        nt = omp_get_max_threads(); omp_set_num_threads(1);
#endif
        qp_solve(&P, &oi, &S1);
#ifdef _OPENMP
        omp_set_num_threads(nt);
#endif
        qp_solve(&P, &oi, &Sn); qp_ipm_stage_mode(-1);
        CHECK(G.status == QP_OPTIMAL && S1.status == QP_OPTIMAL && Sn.status == QP_OPTIMAL && fabs(S1.obj - G.obj) <= 1e-7 * (1 + fabs(G.obj)) && S1.iter == Sn.iter &&
              !memcmp(S1.x, Sn.x, sizeof(double) * n) && !memcmp(S1.y, Sn.y, sizeof(double) * m) && !memcmp(S1.z, Sn.z, sizeof(double) * n),
              "MPC with %d states (n %d): the general path %s %.10g, the stage-wise path on one thread %s %.10g (%d iterations), on %d threads %s %.10g (%d iterations), the same bits: %s",
              nx, n, qp_status_name(G.status), G.obj, qp_status_name(S1.status), S1.obj, S1.iter, nt, qp_status_name(Sn.status), Sn.obj, Sn.iter,
              S1.x && Sn.x && !memcmp(S1.x, Sn.x, sizeof(double) * n) ? "yes" : "no");
        qp_res_free(&G); qp_res_free(&S1); qp_res_free(&Sn); qp_prob_free(&P); free(Ad); free(Bd);
    }
    printf(fails ? "qp_test: %d failures\n" : "qp_test: all checks passed\n", fails);
    return fails != 0;
}

/* The solver's objects print through brisk_printf and brisk_fprintf (brisk.h), which main.c defines; this test links them
 * without main.o, with qpstub.c's plain forwarding versions (the merge for 1.6.5: they were also defined here). */
#include <stdarg.h>
