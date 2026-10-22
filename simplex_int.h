/* simplex_int.h - internal structures of the simplex method (simplex*.c): sparse work vectors and the
 * factorization of the basis with its updates (simplex_lu.c). Not part of the API (simplex.h).
 *
 * Rows of the basis are identified by their row index i (0..m-1). After a factorization the basis is
 * reordered so that the column in position i has its pivot in row i: solves take and return vectors
 * indexed by rows, and B^{-1} a gives in entry i the coefficient of the basic variable of position i. */
#ifndef BRISK_SIMPLEX_INT_H
#define BRISK_SIMPLEX_INT_H
#include <stddef.h>

/* a vector of length n kept dense (val) with the list of its nonzero positions (idx[0..cnt-1]);
 * cnt < 0: the list is not kept (the vector is treated as dense) */
typedef struct {
    int n, cnt;
    int *idx;
    double *val;
    unsigned char *mark;    /* scratch: a position is in idx (used while a list is built) */
} SxVec;
void sxv_init(SxVec *v, int n);
void sxv_free(SxVec *v);
void sxv_clear(SxVec *v);                   /* all zero (by the list when it is short) */
void sxv_reindex(SxVec *v);                 /* rebuild the list from val (drops exact zeros) */
void sxv_copy(SxVec *dst, const SxVec *src);

/* the factorization B = L U of the basis, with Forrest-Tomlin updates (row etas R):
 *   B^{-1} = U^{-1} R_k ... R_1 L^{-1}.
 * L: column etas in pivot order (lrow[k] the pivot row, entries (row, multiplier));
 * U: upper triangular in the order uord[] (row ids; -1 a pivot moved away by an update), diagonal
 *    udiag[row]; stored by columns (ucol: per row id r the entries (row id i, u) of the pivots before r)
 *    and by rows (urow: per row id r the entries (row id j, u) of the pivots after r);
 * R: row etas (rpiv[e]: the row they change; entries (row id j, eta)). */
typedef struct {
    int m;
    /* L */
    int nl;                         /* L columns (pivots with a nonempty eta) */
    int *lrow, *lstart;             /* lstart[nl] = end */
    int *lidx; double *lval; int lcap;
    int *lrstart, *lridx; double *lrval;   /* L by rows: per row id i the entries (row id of the pivot, multiplier) */
    int *lpos;                      /* per row id: its L column (-1: none) */
    int *lcs_row, *lcl_row;         /* per row id: start and length of its L column (for the depth-first search) */
    int *lrlen;                     /* per row id: length of its row of L */
    int *lrorder, nlrorder;         /* rows with entries in L, by decreasing pivot step (BTRAN order) */
    /* U */
    int *uord, nord, ordcap;        /* pivot order of U (row ids; -1 deleted) */
    int *uordpos;                   /* per row id: its position in uord */
    int *uz, nuz, uzcap, *uzpos;    /* the pivots of uord whose column is not empty or whose diagonal is not 1 (-1 deleted) */
    double *udiag;
    int *ucs, *ucl, *ucc;           /* column lists: start, length, capacity in (uci, ucv) */
    int *uci; double *ucv; int ucend, ucap;
    int *urs, *url, *urc;           /* row lists in (uri, urv) */
    int *uri; double *urv; int urend, urcap;
    /* R */
    int nr, rcapn;
    int *rpiv, *rstart;             /* rstart[nr] = end */
    int *ridx; double *rval; int rcap;
    /* statistics */
    long nnz_l, nnz_u0, nnz_r;      /* entries after the factorization; R entries */
    long nnz_u;                     /* entries of U now (the updates change it) */
    double factor_work;             /* multiply-adds of the last factorization (for the refactorization rule) */
    int nupd;                       /* updates since the factorization */
    int rank_deficient;             /* columns that could not be pivoted at the last factorization */
    /* work */
    int *stack, *wmark, *wlist, nwlist; double *wval;
    int *dfs_pos;
    /* density estimates of the results (for the choice between the hypersparse and the sparse solves) */
    double pivthr;                  /* threshold of the partial pivoting (<= 0: 0.1) */
    long nflops;                    /* work counter (entries touched), for the cost model of refactorization */
} SxLU;

/* the scratch space of a solve: one per solve that runs at the same time as another (the factorization's own, F->stack
 * and the others above, serves the solves called without one) */
typedef struct { int m; int *stack, *wmark, *wlist, *dfs_pos; long nflops; } SxWs;
void sxws_init(SxWs *W, int m);
void sxws_free(SxWs *W);

/* the basis: for position i, var[i] (< n: column of A; >= n: the logical of row var-n, column e_{var-n}).
 * Factorizes, reorders var[] (pivot of position i in row i). Columns that could not be pivoted are
 * replaced by logicals; their variables are returned in out[] (count returned), the logicals that took
 * their place have been put in var[]. Returns the number replaced (0: a regular basis). */
int  sxlu_factor(SxLU *F, int m, int n, const int *Ap, const int *Ai, const double *Ax, int *var, int *out);
void sxlu_free(SxLU *F);
/* B x = rhs in place (rhs indexed by rows; on return x indexed by positions = rows).
 * spike != NULL: the partial result after L and R is saved there (for the update) */
void sxlu_ftran(SxLU *F, SxVec *x, double expdens, SxVec *spike);
/* the same with the scratch space W (W->m == m): solves with different W may run at the same time */
void sxlu_ftran_ws(SxLU *F, SxWs *W, SxVec *x, double expdens, SxVec *spike);
/* B x = rhs wanted at the positions pat[0..npat-1] only: those entries are exact; the others may be zero or computed;
 * the list holds the computed positions. W2: a second scratch space (W2->m == m). Returns the nonzeros after L and R
 * (-1: not counted) */
int  sxlu_ftran_part(SxLU *F, SxWs *W2, SxVec *x, double expdens, const int *pat, int npat);
/* B' y = rhs in place; upart != NULL: the partial result after U' is saved there (for the update) */
void sxlu_btran(SxLU *F, SxVec *y, double expdens, SxVec *upart);
/* replace the column in position r by the entering column whose spike (from sxlu_ftran) and the
 * partial btran of e_r (upart) are given; alpha = (B^{-1} a_q)_r. Returns 0, or 1 if the update is
 * numerically doubtful (refactorize). */
int  sxlu_update(SxLU *F, int r, SxVec *spike, SxVec *upart, double alpha);
#endif
